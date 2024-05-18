import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal, Optional, Dict 
from collections import defaultdict
from scipy.special import j0
from tqdm import tqdm
import os
from utils.statistical_models import forecast_evaluation

import tensorflow as tf
import tensorflow_addons as tfa

from keras.utils import set_random_seed
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import regularizers
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import load_model
from keras_tuner.tuners import RandomSearch
import importlib.util
import subprocess
import sys

def check_and_install(package):
    """
    Check if a package is installed and install it if it's not.
    """
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")

class LSTMnetwork():
    def __init__(self,
                train:np.array,
                test:np.array,
                seq_length:int = 10,
                h:int = 1,
                batch_size:int = 8,
                stride:int = 1,
                **kwarg):
        
        """
        Parameters
        ----------
        train : np.array
            The training data.
        test : np.array
            The test data.
        seq_length : int, optional
            The sequence length use to train the neural net, by default 10. This is the number of previous
            timesteps used to predict the forecasts h steps ahead and, as such, it's one of the most important
            hyperparameters of the model and should be carefully chosen/tuned.
        h : int, optional
            The forecast horizon, by default 1. This is the number of steps ahead we want to forecast.
        batch_size : int, optional
            The batch size used to train the neural net, by default 8. This is the number of samples used in each
            iteration of the training process. It's also an important hyperparameter that should be carefully chosen/tuned.
        stride : int, optional
            The stride used to train the neural net, by default 1. This is the number of steps between each sequence
            used to train the model.
        """
        
        self.train = train
        self.train_ = train
        self.test = test
        self.seq_length = seq_length
        self.h = h
        self.batch_size = batch_size
        self.stride = stride
        self.additional_params = kwarg,
        self.cross_validated = False,
        self.fitted = False
        self.forecasted = False

        self.train_val_split(**kwarg)
        self.preprocess()

    def train_val_split(self, train_fraction:float = 0.67):
        """
        Split the training data into training and validation sets.
        Parameters
        ----------
        train_fraction : float, optional
            The fraction of the training data to use for training, by default 0.67.
        """
        self.train, self.val = self.train[0:int(len(self.train)*train_fraction)], \
            self.train[int(len(self.train)*train_fraction):]

    def preprocess(self):
        """
        Preprocess the data by creating the sequences and targets for training, validation and testing.
        This is done using the TimeseriesGenerator class from Keras, which is a generates batches of temporal data.
        """

        len_train = len(self.train)
        len_val = len(self.val)
        len_test = len(self.test)

        self.train_generator = self.sequence_generator(data = self.train, 
                                                       target = np.roll(self.train, -self.h+1), 
                                                       seq_length = self.seq_length, 
                                                       batch_size = self.batch_size, 
                                                       stride = self.stride,
                                                       end_index = (len_train - self.h))
        
        self.val_generator = self.sequence_generator(data = self.val, 
                                                     target = np.roll(self.val, -self.h+1), 
                                                     seq_length = self.seq_length, 
                                                     batch_size = self.batch_size, 
                                                     stride = self.stride,
                                                     end_index = (len_val -self.h))
        
        self.test_generator = self.sequence_generator(data = self.test, 
                                                      target = np.roll(self.test, -self.h+1), 
                                                      seq_length = self.seq_length, 
                                                      batch_size = self.batch_size, 
                                                      stride = self.stride,
                                                      end_index = (len_test - self.h))

        return None

    def build_model(self, hp):
        """
        This function is using Keras Tuner to define a hyperparameter search space for a LSTM model. 
        Keras Tuner is a library that helps you pick the optimal set of hyperparameters for your TensorFlow program.

        Here's a breakdown of what's happening in the build_model function:

            1) The hp.Choice function is used to specify a set of discrete choices for a hyperparameters
            
            2) The hp.Float function is used to specify a range of float values for a hyperparameter.

            3) The hp.Boolean function is used to specify a boolean hyperparameter. In this case, it's 
                used to decide whether to include a second hidden layer in the model.

        Parameters
        ----------
        hp : HyperParameters
            The hyperparameters to use when building the model.
        """

        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'elu'])

        hp_recurrent_layer_1 = hp.Choice('recurrent_layer_1', values=[16, 32, 64, 128])
        hp_recurrent_layer_2 = hp.Choice('recurrent_layer_2', values=[16, 32, 64, 128])
        hp_dense_layer_1 = hp.Choice('dense_layer_1', values=[16, 32, 64, 128])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        hp_optimizers = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        hp_drop_out_rate = hp.Float('drop_out_rate', min_value=0.0, max_value=0.5, step=0.1)
        hp_initializers = hp.Choice('initializer', values=['random_normal','glorot_uniform', 'he_normal'])
        l2_coeff = hp.Float('l2_coeff', min_value=0.0, max_value=0.5, step=0.1)

        model = Sequential()
        include_second_hidden_layer = hp.Boolean('with_second_hidden_layer', default=True)
        if include_second_hidden_layer:
            model.add(LSTM(units=hp_recurrent_layer_1, 
                            return_sequences=True,
                            input_shape=(self.seq_length, 1),
                            kernel_initializer=hp_initializers,
                            kernel_regularizer=regularizers.l2(l2_coeff)))
            model.add(Dropout(hp_drop_out_rate))
            model.add(LSTM(units=hp_recurrent_layer_2,
                           kernel_initializer=hp_initializers,
                           kernel_regularizer=regularizers.l2(l2_coeff)))
            model.add(Dropout(hp_drop_out_rate))
        else:
            model.add(LSTM(units=hp_recurrent_layer_1, 
                            input_shape=(self.seq_length, 1),
                            kernel_initializer=hp_initializers,
                            kernel_regularizer=regularizers.l2(l2_coeff)))
            model.add(Dropout(hp_drop_out_rate))
        model.add(Dense(units = hp_dense_layer_1, 
                        activation=hp_activation,
                        kernel_initializer=hp_initializers,
                        kernel_regularizer=regularizers.l2(l2_coeff)))
        model.add(Dense(1))

        if hp_optimizers == 'adam':
            opt = Adam(learning_rate=hp_learning_rate)
        elif hp_optimizers == 'sgd':
            opt = SGD(learning_rate=hp_learning_rate)
        elif hp_optimizers == 'rmsprop':
            opt = RMSprop(learning_rate=hp_learning_rate)

        model.compile(loss='mean_squared_error',
                        optimizer=opt,
                        metrics=['mean_absolute_error']
        )

        return model
        
    def cross_validate(self, DGP:str, **kwarg):
        """
        Cross validate the model using the RandomSearch tuner from Keras Tuner.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file where the results of the cross
              validation will be saved.
        """

        objetive = kwarg.get('objective', 'val_loss')
        max_trials = kwarg.get('max_trials', 10)
        executions_per_trial = kwarg.get('executions_per_trial', 3)
        seed = kwarg.get('seed', 42)
        directory = kwarg.get('directory', 'tuner/LSTM')
        epochs = kwarg.get('epochs', 10)

        filepath = f'{DGP}_LSTM_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride'
        
        self.tuner = RandomSearch(self.build_model,
                             objective=objetive,
                             max_trials=max_trials,
                             executions_per_trial=executions_per_trial,
                             directory=directory,
                             project_name=filepath,
                             seed = seed
                             )
        
        self.tuner.search(self.train_generator,
                        epochs=epochs,
                        validation_data=self.val_generator,)
        
        self.cross_validated = True

    def best_model(self):
        """
        Get the best model from the tuner.
        """

        if not self.cross_validated:
            self.cross_validate()
        
        self.best_hp = self.tuner.get_best_hyperparameters()[0]
        self.model = self.build_model(self.best_hp)

        return self.model

    def fit(self, DGP:str, epochs:int = 100, callbacks:Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'] = 'all', verbose:int=1, **kwarg):

        """
        Fit the model to the training data. It'll use the best model found during the cross validation process.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file with the model trained.
        epochs : int, optional
            The number of epochs to train the model, by default 100. An epoch is one complete pass through the training data.
        callbacks : Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'], optional
            The callbacks to use during the training process, by default 'all'. Callbacks are functions that are called during
            the training process to perform specific actions. For example, the EarlyStopping callback stops the training process
            if the validation loss stops improving, while the ModelCheckpoint callback saves the best model found during the training
            process. The ReduceLROnPlateau callback reduces the learning rate if the validation loss stops improving.

        """

        ES_min_delta = kwarg.get('ES_min_delta', 1e-3)
        ES_patience = kwarg.get('ES_patience', 10)
        RLROP_factor = kwarg.get('RLROP_factor', 0.1)
        RLROP_patience = kwarg.get('RLROP_patience', 5)

        if not self.cross_validated:
            raise ValueError('You must cross validate the model before fitting.')

        checkpoint_dir = 'checkpoints/LSTM'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_filepath = f'{checkpoint_dir}/{DGP}_best_model_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride.h5'

        if os.path.exists(model_filepath):
            self.model = load_model(model_filepath)
            print('Loading existing model in filepath {}'.format(model_filepath))
        else:
            callbacks_list = []
            if callbacks == 'EarlyStopping' or callbacks == 'all':
                callbacks_list.append(EarlyStopping(monitor='val_loss', min_delta=ES_min_delta, patience=ES_patience, verbose=verbose, mode='min', restore_best_weights=True))
            if callbacks == 'ModelCheckpoint' or callbacks == 'all':
                callbacks_list.append(ModelCheckpoint(filepath=model_filepath, 
                                                    monitor='val_loss', 
                                                    mode = 'min',
                                                    save_best_only=True))
            if callbacks == 'ReduceLROnPlateau' or callbacks == 'all':
                callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=RLROP_factor, patience=RLROP_patience, verbose=verbose, mode='min'))

            self.best_model()
            self.history = self.model.fit(self.train_generator, 
                                        epochs=epochs, 
                                        verbose=verbose, 
                                        validation_data=self.val_generator, 
                                        callbacks=[callbacks_list])

        self.fitted = True

    def predict(self, n_runs:int, strategy:Literal['one_shot', 'iterative','all']='all', **kwarg) -> defaultdict:

        """
        Forecast the test data using the best model found during the cross validation process.
        Parameters
        ----------
        strategy : Literal['one_shot', 'iterative','all'], optional
            The strategy to use when forecasting the test data, by default 'all'.
        """
        
        if not self.fitted:
            raise ValueError('You must fit the model before predicting.')

        self.predictions = defaultdict()

        if strategy == 'one_shot' or strategy == 'all':
            # To do: implement several runs with one_shot strategy
            previous_timesteps = self.train_[-self.seq_length:]
            forecasts = []

            for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using one_shot strategy:'):
                forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][0]
                forecasts.append(forecast)
                previous_timesteps = np.append(previous_timesteps, forecast)[-self.seq_length:]

            forecasts = np.array(forecasts).reshape(-1,1)
            self.predictions['one_shot'] = forecasts

        if strategy == 'iterative' or strategy == 'all':

            forecast_runs = []

            for run in range(n_runs):
            
                previous_timesteps = self.train_[-self.seq_length:]
                forecasts = []

                for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using iterative strategy:'):
                    forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][0]
                    forecasts.append(forecast)
                    previous_timesteps = np.append(previous_timesteps, self.test[i])[-self.seq_length:]

                forecasts = np.array(forecasts).reshape(-1,1)
                forecast_runs.append(forecasts)

                self.fitted = False
                
                set_random_seed(run)
                dgp = kwarg.get('dgp', {})
                self.fit(DGP = '{}_run_{}'.format(dgp, run+1), verbose=0)

            self.predictions['iterative'] = np.mean(forecast_runs, axis=0)
                
        self.forecasted = True

    def evaluate_forecast(self, h:int) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the forecast using different metrics.
        Parameters
        ----------
        h : int
            The forecast horizon.
        Returns
        -------
        Dict[str, Dict[str, float]]
            The evaluation results.
        """

        if not self.forecasted:
            raise ValueError('You must predict first.')

        self.evaluation_results = defaultdict()

        for key, value in self.predictions.items():
            self.evaluation_results[key] = forecast_evaluation(self.h, self.test, value)

        return self.evaluation_results
    
    @staticmethod
    def sequence_generator(
            data:np.array,
            target:np.array,
            seq_length:int, 
            batch_size:int = 128, 
            stride:int = 1, 
            end_index:int = None,
            ) -> TimeseriesGenerator:
        """
        Generate sequences of data and targets for training, validation and testing.
        Parameters
        ----------
        data : np.array
            The data to generate sequences from.
        target : np.array
            The target to generate sequences from. Generally, it's the same as the data.
        seq_length : int
            The sequence length.
        batch_size : int, optional
            The batch size, by default 128.
        stride : int, optional
            The stride, by default 1.
        end_index : int
            The end index, by default None.
        """
        
        generator = TimeseriesGenerator(data, 
                                        target, 
                                        length=seq_length, 
                                        batch_size=batch_size,
                                        stride=stride, 
                                        end_index=end_index,
                                        )
        
        return generator

class PeepholedLSTMnetwork():
    def __init__(self,
                train:np.array,
                test:np.array,
                seq_length:int = 10,
                h:int = 1,
                batch_size:int = 8,
                stride:int = 1,
                **kwarg):
        
        """
        Parameters
        ----------
        train : np.array
            The training data.
        test : np.array
            The test data.
        seq_length : int, optional
            The sequence length use to train the neural net, by default 10. This is the number of previous
            timesteps used to predict the forecasts h steps ahead and, as such, it's one of the most important
            hyperparameters of the model and should be carefully chosen/tuned.
        h : int, optional
            The forecast horizon, by default 1. This is the number of steps ahead we want to forecast.
        batch_size : int, optional
            The batch size used to train the neural net, by default 8. This is the number of samples used in each
            iteration of the training process. It's also an important hyperparameter that should be carefully chosen/tuned.
        stride : int, optional
            The stride used to train the neural net, by default 1. This is the number of steps between each sequence
            used to train the model.
        """
        
        self.train = train
        self.train_ = train
        self.test = test
        self.seq_length = seq_length
        self.h = h
        self.batch_size = batch_size
        self.stride = stride
        self.additional_params = kwarg,
        self.cross_validated = False,
        self.fitted = False
        self.forecasted = False

        self.train_val_split(**kwarg)
        self.preprocess()

        # Check and install tensorflow_addons, package that will be used to implement the PeepholedLSTM
        check_and_install("tensorflow_addons")

    def train_val_split(self, train_fraction:float = 0.67):
        """
        Split the training data into training and validation sets.
        Parameters
        ----------
        train_fraction : float, optional
            The fraction of the training data to use for training, by default 0.67.
        """
        self.train, self.val = self.train[0:int(len(self.train)*train_fraction)], \
            self.train[int(len(self.train)*train_fraction):]

    def preprocess(self):
        """
        Preprocess the data by creating the sequences and targets for training, validation and testing.
        This is done using the TimeseriesGenerator class from Keras, which is a generates batches of temporal data.
        """

        len_train = len(self.train)
        len_val = len(self.val)
        len_test = len(self.test)
        
        self.train_generator = LSTMnetwork.sequence_generator(data = self.train, 
                                                                target = np.roll(self.train, -self.h+1), 
                                                                seq_length = self.seq_length, 
                                                                batch_size=self.batch_size, 
                                                                stride=self.stride,
                                                                end_index = (len_train - self.h))
        
        self.val_generator = LSTMnetwork.sequence_generator(data = self.val, 
                                                            target = np.roll(self.val, -self.h+1), 
                                                            seq_length = self.seq_length, 
                                                            batch_size = self.batch_size, 
                                                            stride = self.stride,
                                                            end_index = (len_val -self. h))
        
        self.test_generator = LSTMnetwork.sequence_generator(data = self.test, 
                                                            target = np.roll(self.test, -self.h+1), 
                                                            seq_length = self.seq_length, 
                                                            batch_size = self.batch_size, 
                                                            stride = self.stride,
                                                            end_index = len_test - self.h)

        return None

    def build_model(self, hp):
        """
        This function is using Keras Tuner to define a hyperparameter search space for a LSTM model. 
        Keras Tuner is a library that helps you pick the optimal set of hyperparameters for your TensorFlow program.

        Here's a breakdown of what's happening in the build_model function:

            1) The hp.Choice function is used to specify a set of discrete choices for a hyperparameters
            
            2) The hp.Float function is used to specify a range of float values for a hyperparameter.

            3) The hp.Boolean function is used to specify a boolean hyperparameter. In this case, it's 
                used to decide whether to include a second hidden layer in the model.

        As could be seen, this method differs from the one implemented in LSTMnetwork because it 
            used TensorFlow backend directly instead of Keras.

        Parameters
        ----------
        hp : HyperParameters
            The hyperparameters to use when building the model.
        """

        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'elu'])

        hp_recurrent_layer_1 = hp.Choice('recurrent_layer_1', values=[16, 32, 64, 128])
        hp_recurrent_layer_2 = hp.Choice('recurrent_layer_2', values=[16, 32, 64, 128])
        hp_dense_layer_1 = hp.Choice('dense_layer_1', values=[16, 32, 64, 128])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        hp_optimizers = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        hp_drop_out_rate = hp.Float('drop_out_rate', min_value=0.0, max_value=0.5, step=0.1)
        hp_initializers = hp.Choice('initializer', values=['random_normal','glorot_uniform', 'he_normal'])
        l2_coeff = hp.Float('l2_coeff', min_value=0.0, max_value=0.5, step=0.1)

        model = tf.keras.Sequential()
        include_second_hidden_layer = hp.Boolean('with_second_hidden_layer', default=True)
        if include_second_hidden_layer:
            LSTMCell1 = tfa.rnn.PeepholeLSTMCell(units=hp_recurrent_layer_1, 
                                                kernel_initializer=hp_initializers,
                                                kernel_regularizer=regularizers.l2(l2_coeff))
            model.add(tf.keras.layers.RNN(LSTMCell1, return_sequences=True))
            model.add(tf.keras.layers.Dropout(hp_drop_out_rate))
            LSTMCell2 = tfa.rnn.PeepholeLSTMCell(units=hp_recurrent_layer_1, 
                                    kernel_initializer=hp_initializers,
                                    kernel_regularizer=regularizers.l2(l2_coeff))
            model.add(tf.keras.layers.RNN(LSTMCell2, return_sequences=True))
            model.add(tf.keras.layers.Dropout(hp_drop_out_rate))
        else:
            LSTMCell1 = tfa.rnn.PeepholeLSTMCell(units=hp_recurrent_layer_1, 
                                                kernel_initializer=hp_initializers,
                                                kernel_regularizer=regularizers.l2(l2_coeff))
            model.add(tf.keras.layers.RNN(LSTMCell1, return_sequences=True))
            model.add(tf.keras.layers.Dropout(hp_drop_out_rate))
        model.add(tf.keras.layers.Dense(units = hp_dense_layer_1, 
                                activation=hp_activation,
                                kernel_initializer=hp_initializers,
                                kernel_regularizer=regularizers.l2(l2_coeff)))
        model.add(tf.keras.layers.Dense(1))

        if hp_optimizers == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        elif hp_optimizers == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
        elif hp_optimizers == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)

        model.compile(loss='mean_squared_error',
                        optimizer=opt,
                        metrics=['mean_absolute_error']
        )

        model.build(input_shape=(None, self.seq_length, 1))

        return model
        
    def cross_validate(self, DGP:str, **kwarg):
        """
        Cross validate the model using the RandomSearch tuner from Keras Tuner.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file where the results of the cross
              validation will be saved.
        """

        objetive = kwarg.get('objective', 'val_loss')
        max_trials = kwarg.get('max_trials', 10)
        executions_per_trial = kwarg.get('executions_per_trial', 3)
        seed = kwarg.get('seed', 42)
        directory = kwarg.get('directory', 'tuner/PeepholedLSTM')
        epochs = kwarg.get('epochs', 10)

        filepath = f'{DGP}_LSTM_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride'
        
        self.tuner = RandomSearch(self.build_model,
                             objective=objetive,
                             max_trials=max_trials,
                             executions_per_trial=executions_per_trial,
                             directory=directory,
                             project_name=filepath,
                             seed = seed
                             )
        
        self.tuner.search(self.train_generator,
                        epochs=epochs,
                        validation_data=self.val_generator,)
        
        self.cross_validated = True

    def best_model(self):
        """
        Get the best model from the tuner.
        """

        if not self.cross_validated:
            self.cross_validate()
        
        self.best_hp = self.tuner.get_best_hyperparameters()[0]
        self.model = self.build_model(self.best_hp)

        return self.model

    def fit(self, DGP:str, epochs:int = 100, callbacks:Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'] = 'all', verbose:int=1, **kwarg):

        """
        Fit the model to the training data. It'll use the best model found during the cross validation process.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file with the model trained.
        epochs : int, optional
            The number of epochs to train the model, by default 100. An epoch is one complete pass through the training data.
        callbacks : Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'], optional
            The callbacks to use during the training process, by default 'all'. Callbacks are functions that are called during
            the training process to perform specific actions. For example, the EarlyStopping callback stops the training process
            if the validation loss stops improving, while the ModelCheckpoint callback saves the best model found during the training
            process. The ReduceLROnPlateau callback reduces the learning rate if the validation loss stops improving.

        """

        ES_min_delta = kwarg.get('ES_min_delta', 1e-3)
        ES_patience = kwarg.get('ES_patience', 10)
        RLROP_factor = kwarg.get('RLROP_factor', 0.1)
        RLROP_patience = kwarg.get('RLROP_patience', 5)

        if not self.cross_validated:
            raise ValueError('You must cross validate the model before fitting.')

        checkpoint_dir = 'checkpoints/PeepholedLSTM'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_filepath = f'{checkpoint_dir}/{DGP}_best_model_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride.h5'

        if os.path.exists(model_filepath):
            self.model = load_model(model_filepath)
            print('Loading existing model in filepath {}'.format(model_filepath))
        else:
            callbacks_list = []
            if callbacks == 'EarlyStopping' or callbacks == 'all':
                callbacks_list.append(EarlyStopping(monitor='val_loss', min_delta=ES_min_delta, patience=ES_patience, verbose=verbose, mode='min', restore_best_weights=True))
            if callbacks == 'ModelCheckpoint' or callbacks == 'all':
                callbacks_list.append(ModelCheckpoint(filepath=model_filepath, 
                                                    monitor='val_loss', 
                                                    mode = 'min',
                                                    save_best_only=True))
            if callbacks == 'ReduceLROnPlateau' or callbacks == 'all':
                callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=RLROP_factor, patience=RLROP_patience, verbose=verbose, mode='min'))

            self.best_model()
            self.history = self.model.fit(self.train_generator, 
                                        epochs=epochs, 
                                        verbose=verbose, 
                                        validation_data=self.val_generator, 
                                        callbacks=[callbacks_list])

        self.fitted = True

    def predict(self, n_runs:int, strategy:Literal['one_shot', 'iterative','all']='all', **kwarg) -> defaultdict:

        """
        Forecast the test data using the best model found during the cross validation process.
        Parameters
        ----------
        strategy : Literal['one_shot', 'iterative','all'], optional
            The strategy to use when forecasting the test data, by default 'all'.
        """
        
        if not self.fitted:
            raise ValueError('You must fit the model before predicting.')

        self.predictions = defaultdict()

        if strategy == 'one_shot' or strategy == 'all':
            
            previous_timesteps = self.train_[-self.seq_length:]
            forecasts = []

            for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using one_shot strategy:'):
                forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][0]
                forecasts.append(forecast)
                previous_timesteps = np.append(previous_timesteps, forecast)[-self.seq_length:]

            forecasts = np.array(forecasts).reshape(-1,1)
            self.predictions['one_shot'] = forecasts

        if strategy == 'iterative' or strategy == 'all':

            forecast_runs = []

            for run in range(n_runs):
            
              previous_timesteps = self.train_[-self.seq_length:]
              forecasts = []

              for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using iterative strategy:'):
                  forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][0]
                  forecasts.append(forecast)
                  previous_timesteps = np.append(previous_timesteps, self.test[i])[-self.seq_length:]

              forecasts = np.array(forecasts).reshape(-1,1)
              forecast_runs.append(forecasts)

              self.fitted = False

              set_random_seed(run)

              dgp = kwarg.get('dgp', {})
              self.fit(DGP = '{}_run_{}'.format(dgp, run+1), verbose=0)

            self.predictions['iterative'] = np.mean(forecast_runs, axis=0)
                
        self.forecasted = True

    def evaluate_forecast(self, h:int) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the forecast using different metrics.
        Parameters
        ----------
        h : int
            The forecast horizon.
        Returns
        -------
        Dict[str, Dict[str, float]]
            The evaluation results.
        """

        if not self.forecasted:
            raise ValueError('You must predict first.')

        self.evaluation_results = defaultdict()

        for key, value in self.predictions.items():
            self.evaluation_results[key] = forecast_evaluation(self.h, self.test, value)

        return self.evaluation_results

class Seq2SeqLSTM():
    def __init__(self,
                train:np.array,
                test:np.array,
                seq_length:int = 10,
                h:int = 1,
                batch_size:int = 8,
                stride:int = 1,
                **kwarg):

        """
        Parameters
        ----------
        train : np.array
            The training data.
        test : np.array
            The test data.
        seq_length : int, optional
            The sequence length use to train the neural net, by default 10. This is the number of previous
            timesteps used to predict the forecasts h steps ahead and, as such, it's one of the most important
            hyperparameters of the model and should be carefully chosen/tuned.
        h : int, optional
            The forecast horizon, by default 1. This is the length of the output sequence in the decoder.
        batch_size : int, optional
            The batch size used to train the neural net, by default 8. This is the number of samples used in each
            iteration of the training process. It's also an important hyperparameter that should be carefully chosen/tuned.
        stride : int, optional
            The stride used to train the neural net, by default 1. This is the number of steps between each sequence
            used to train the model.
        """

        self.train = train
        self.train_ = train
        self.test = test
        self.seq_length = seq_length
        self.h = h
        self.batch_size = batch_size
        self.stride = stride
        self.additional_params = kwarg,
        self.cross_validated = False,
        self.fitted = False
        self.forecasted = False

        self.train_val_split(**kwarg)
        self.preprocess()

    def train_val_split(self, train_fraction:float = 0.67):

        """
        Split the training data into training and validation sets.
        Parameters
        ----------
        train_fraction : float, optional
            The fraction of the training data to use for training, by default 0.67.
        """

        self.train, self.val = self.train[0:int(len(self.train)*train_fraction)], \
            self.train[int(len(self.train)*train_fraction):]
        
    def preprocess(self):

        """
        Preprocess the data by creating the sequences and targets for training, validation and testing.
        Unlike the other models, this model creates both the input and output sequences for the training, 
        validation and testing sets.
        """
        
        self.train_in, self.train_out = self.data_to_seq(data = self.train, 
                                                    seq_length=self.seq_length, 
                                                    h=self.h, 
                                                    stride=self.stride
                                                    )
        
        self.val_in, self.val_out = self.data_to_seq(data = self.val, 
                                                seq_length=self.seq_length, 
                                                h=self.h, 
                                                stride=self.stride
                                                )
        
        self.test_in, self.test_out = self.data_to_seq(data = self.test,
                                                seq_length=self.seq_length,
                                                h=self.h,
                                                stride=self.stride
                                                )
    
    def build_model(self, hp):

        """
        This function is using Keras Tuner to define a hyperparameter search space for a S2S architecture with LSTM cells.
        Unlike previous classes, this model uses Keras' functional API to build the model and it's based upon the following
        implementation https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb
        """

        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid','elu'])

        hp_layer_1_dim = hp.Choice('recurrent_layer_1', values=[16, 32, 64, 128])
        hp_layer_2_prop = hp.Choice('recurrent_layer_2', values=[0.5, 0.25, 0.125])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
        hp_optimizers = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
        hp_drop_out_rate = hp.Float('drop_out_rate', min_value=0.0, max_value=0.5, step=0.1)
        hp_initializers = hp.Choice('initializer', values=['random_normal','glorot_uniform', 'he_normal'])
        l2_coeff = hp.Float('l2_coeff', min_value=0.0, max_value=0.5, step=0.1)

        input_train = Input(shape=(self.train_in.shape[1], self.train_in.shape[2]))
        output_train = Input(shape=(self.train_out.shape[1], self.train_out.shape[2]))

        include_second_hidden_layer = hp.Boolean('with_second_hidden_layer', default=True)
        if include_second_hidden_layer:
            encoder = LSTM(units = hp_layer_1_dim, 
                            activation = hp_activation, 
                            dropout = hp_drop_out_rate, 
                            kernel_initializer=hp_initializers,
                            kernel_regularizer=regularizers.l2(l2_coeff),
                            recurrent_dropout = hp_drop_out_rate, 
                            return_sequences=True, 
                            return_state=False)(input_train)
            encoder_last_h1, _, encoder_last_c = LSTM(units = int(hp_layer_1_dim*hp_layer_2_prop), 
                                                      activation=hp_activation, 
                                                      dropout=hp_drop_out_rate, 
                                                      kernel_initializer=hp_initializers,
                                                      kernel_regularizer=regularizers.l2(l2_coeff),
                                                      recurrent_dropout=hp_drop_out_rate, 
                                                      return_sequences=False, 
                                                      return_state=True)(encoder)
            encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
            encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
            decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
            decoder = LSTM(units = int(hp_layer_1_dim*hp_layer_2_prop), 
                           activation=hp_activation, 
                           dropout=hp_drop_out_rate, 
                           kernel_initializer=hp_initializers,
                           kernel_regularizer=regularizers.l2(l2_coeff),
                           recurrent_dropout=hp_drop_out_rate, 
                           return_state=False, 
                           return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
            decoder = LSTM(units = hp_layer_1_dim, 
                            activation = hp_activation, 
                            dropout = hp_drop_out_rate, 
                            kernel_initializer=hp_initializers,
                            kernel_regularizer=regularizers.l2(l2_coeff),
                            recurrent_dropout = hp_drop_out_rate, 
                            return_sequences=True, 
                            return_state=False)(decoder)
            out = TimeDistributed(Dense(output_train.shape[2]))(decoder)                
        else:
            encoder_last_h1, _, encoder_last_c = LSTM(units = hp_layer_1_dim, 
                                                    activation = hp_activation, 
                                                    dropout = hp_drop_out_rate, 
                                                    kernel_initializer=hp_initializers,
                                                    kernel_regularizer=regularizers.l2(l2_coeff),
                                                    recurrent_dropout = hp_drop_out_rate, 
                                                    return_sequences=False, 
                                                    return_state=True)(input_train)
            encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
            encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
            decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
            decoder = LSTM(units = hp_layer_1_dim, 
                            activation = hp_activation, 
                            dropout = hp_drop_out_rate, 
                            kernel_initializer=hp_initializers,
                            kernel_regularizer=regularizers.l2(l2_coeff),
                            recurrent_dropout = hp_drop_out_rate, 
                            return_state=False, 
                            return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
            out = TimeDistributed(Dense(output_train.shape[2]))(decoder)  

        model = Model(inputs=input_train, outputs=out)

        if hp_optimizers == 'adam':
            opt = Adam(learning_rate=hp_learning_rate, clipnorm=1)
        elif hp_optimizers == 'sgd':
            opt = SGD(learning_rate=hp_learning_rate, clipnorm=1)
        elif hp_optimizers == 'rmsprop':
            opt = RMSprop(learning_rate=hp_learning_rate, clipnorm=1)

        model.compile(loss='mean_squared_error',
                      optimizer=opt, 
                      metrics=['mean_absolute_error'])

        return model

    def cross_validate(self, DGP:str, **kwarg):

        """
        Cross validate the model using the RandomSearch tuner from Keras Tuner.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file where the results of the cross
              validation will be saved.
        """

        objetive = kwarg.get('objective', 'val_loss')
        max_trials = kwarg.get('max_trials', 10)
        executions_per_trial = kwarg.get('executions_per_trial', 3)
        seed = kwarg.get('seed', 42)
        directory = kwarg.get('directory', 'tuner/S2SLSTM')
        epochs = kwarg.get('epochs', 10)

        filepath = f'{DGP}_S2SLSTM_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride'
        
        self.tuner = RandomSearch(self.build_model,
                             objective=objetive,
                             max_trials=max_trials,
                             executions_per_trial=executions_per_trial,
                             directory=directory,
                             project_name=filepath,
                             seed = seed
                             )
        
        self.tuner.search(self.train_in[:, :, :1], 
                          self.train_out[:, :, :1], 
                            validation_data=[self.val_in[:, :, :1], self.val_out[:, :, :1]],
                            epochs=epochs,
        )
        
        self.cross_validated = True

    def best_model(self):

        """
        Get the best model from the tuner.
        """

        if not self.cross_validated:
            self.cross_validate()
        
        self.best_hp = self.tuner.get_best_hyperparameters()[0]
        self.model = self.build_model(self.best_hp)

        return self.model

    def fit(self, DGP:str, 
            epochs:int = 100, 
            callbacks:Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'] = 'all',
            verbose:int=1,
            **kwarg):
        
        """
        Fit the model to the training data. It'll use the best model found during the cross validation process.
        Parameters
        ----------
        DGP : str
            The name of the DGP. It's used to name the file with the model trained.
        epochs : int, optional
            The number of epochs to train the model, by default 100. An epoch is one complete pass through the training data.
        callbacks : Literal['EarlyStopping', 'ModelCheckpoint', 'ReduceLROnPlateau','all'], optional
            The callbacks to use during the training process, by default 'all'. Callbacks are functions that are called during
            the training process to perform specific actions. For example, the EarlyStopping callback stops the training process
            if the validation loss stops improving, while the ModelCheckpoint callback saves the best model found during the training
            process. The ReduceLROnPlateau callback reduces the learning rate if the validation loss stops improving.

        """

        ES_min_delta = kwarg.get('ES_min_delta', 1e-3)
        ES_patience = kwarg.get('ES_patience', 10)
        RLROP_factor = kwarg.get('RLROP_factor', 0.1)
        RLROP_patience = kwarg.get('RLROP_patience', 5)

        if not self.cross_validated:
            raise ValueError('You must cross validate the model before fitting.')

        checkpoint_dir = 'checkpoints/S2SLSTM'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_filepath = f'{checkpoint_dir}/{DGP}_best_model_{self.seq_length}_seq_{self.h}_h_{self.batch_size}_batch_{self.stride}_stride.h5'

        if os.path.exists(model_filepath):
            self.model = load_model(model_filepath)
            print('Loading existing model in filepath {}'.format(model_filepath))
        else:
            callbacks_list = []
            if callbacks == 'EarlyStopping' or callbacks == 'all':
                callbacks_list.append(EarlyStopping(monitor='val_loss', min_delta=ES_min_delta, patience=ES_patience, verbose=verbose, mode='min', restore_best_weights=True))
            if callbacks == 'ModelCheckpoint' or callbacks == 'all':
                callbacks_list.append(ModelCheckpoint(filepath=model_filepath, 
                                                    monitor='val_loss', 
                                                    mode = 'min',
                                                    save_best_only=True))
            if callbacks == 'ReduceLROnPlateau' or callbacks == 'all':
                callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=RLROP_factor, patience=RLROP_patience, verbose=verbose, mode='min'))

            self.best_model()
            self.history = self.model.fit(
                                        self.train_in[:, :, :1], 
                                        self.train_out[:, :, :1], 
                                        validation_data=[self.val_in[:, :, :1], self.val_out[:, :, :1]],
                                        epochs=epochs, 
                                        verbose=verbose, 
                                        batch_size=self.batch_size,
                                        callbacks=[callbacks_list])

        self.fitted = True

    def predict(self, n_runs:int, strategy:Literal['one_shot', 'iterative','all']='all', **kwarg) -> defaultdict:

        """
        Forecast the test data using the best model found during the cross validation process.
        Parameters
        ----------
        strategy : Literal['one_shot', 'iterative','all'], optional
            The strategy to use when forecasting the test data, by default 'all'.
        """
        
        if not self.fitted:
            raise ValueError('You must fit the model before predicting.')

        self.predictions = defaultdict()

        if strategy == 'one_shot' or strategy == 'all':
            
            previous_timesteps = self.train_[-self.seq_length:]
            forecasts = []

            for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using one_shot strategy:'):
                forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][self.h-1][0]
                forecasts.append(forecast)
                previous_timesteps = np.append(previous_timesteps, forecast)[-self.seq_length:]

            forecasts = np.array(forecasts).reshape(-1,1)
            self.predictions['one_shot'] = forecasts

        if strategy == 'iterative' or strategy == 'all':

            forecast_runs = []

            for run in range(n_runs):
            
              previous_timesteps = self.train_[-self.seq_length:]
              forecasts = []

              for i in tqdm(range(len(self.test)-self.h+1), desc='Forecasting using iterative strategy:'):
                  forecast = self.model.predict(previous_timesteps.reshape(1, self.seq_length, 1), verbose=0)[0][self.h-1][0]
                  forecasts.append(forecast)
                  previous_timesteps = np.append(previous_timesteps, self.test[i])[-self.seq_length:]

              forecasts = np.array(forecasts).reshape(-1,1)
              forecast_runs.append(forecasts)

              self.fitted = False

              set_random_seed(run)

              dgp = kwarg.get('dgp', {})
              self.fit(DGP = '{}_run_{}'.format(dgp, run+1), verbose=0)

            self.predictions['iterative'] = np.mean(forecast_runs, axis=0)
                
        self.forecasted = True

    def evaluate_forecast(self, h:int) -> Dict[str, Dict[str, float]]:

        """
        Evaluate the forecast using different metrics.
        Parameters
        ----------
        h : int
            The forecast horizon.
        Returns
        -------
        Dict[str, Dict[str, float]]
            The evaluation results.
        """

        if not self.forecasted:
            raise ValueError('You must predict first.')

        self.evaluation_results = defaultdict()

        for key, value in self.predictions.items():
            self.evaluation_results[key] = forecast_evaluation(self.h, self.test, value)

        return self.evaluation_results

    @staticmethod
    def data_to_seq(data:np.array, seq_length:int=10, h:int=1, stride:int=0):

        """
        Convert the data to sequences. This is done to create the input and output sequences 
        for the training, validation and testing sets in a sequence-to-sequence model.
        """
        in_, out_, = [], []
        j = 0
        for i in range(len(data)-seq_length-h+1):
            if (i+j*stride+seq_length+h) > len(data):
                break
            in_.append(data[i+j*stride:(i+j*stride+seq_length)].tolist())
            out_.append(data[(i+j*stride+seq_length):(i+j*stride+seq_length+h)].tolist())
            j += 1
        return np.array(in_), np.array(out_)

