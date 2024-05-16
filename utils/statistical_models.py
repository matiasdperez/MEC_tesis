import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
import pmdarima as pm
from pmdarima.metrics import smape
from itertools import product
from collections import defaultdict
import pandas as pd
import tqdm
import copy as cp
from typing import Dict, Literal, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ast

def forecast_evaluation(h:int,y_true:np.array, y_pred:np.array) -> Dict[str, float]:
    """
    Evaluate forecast using different forecasting metrics
    Parameters
    ----------
    h : int
        The forecast horizon.
    y_true : np.array
        The true values.
    y_pred : np.array
        The predicted values.
    """
    y_true = y_true[h-1:]
    y_pred = y_pred

    metrics = {}
    
    metrics['MSE'] = np.round(mean_squared_error(y_true, y_pred), 5)
    metrics['RMSE'] = np.round(np.sqrt(metrics['MSE']), 5)
    #metrics['RMSPE'] = np.round(np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)), 5)
    metrics['MAE'] = np.round(mean_absolute_error(y_true, y_pred), 5)
    metrics['MAPE'] = np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 5)
    metrics['SMAPE'] = np.round(smape(y_true, y_pred), 5)
    
    return metrics

class ArimaModel():
    def __init__(self,
                 train:np.array,
                 test:np.array,
                 **kwargs):
        
        self.train = train
        self.test = test
        self.cross_validated = False
        self.fitted = False
        self.forecasted = False
        self.additional_params = kwargs

    def cross_validate(self, 
                       ARIMA_configs:product, 
                       h:int=1,
                       stride:int=25,
                       window_length:int=None,
                       eval_metric:Literal['smape', 'mean_squared_error', 'mean_absolute_error']='mean_squared_error') -> None:
        """
        Cross validate the ARIMA model with the given configurations.
        Parameters
        ----------
        ARIMA_configs : product
            The configurations to cross validate.
        h : int, optional
            The forecast horizon, by default 1
        stride : int, optional
            The stride size between folds, by default 25
        window_length : int, optional
            The window size for sliding window cross validation, by default None
        eval_metric : Literal['smape', 'mean_squared_error', 'mean_absolute_error'], optional
            The metric to use for evaluation, by default 'mean_squared_error'
        """

        if not self.cross_validated:
            self.cross_val_results = defaultdict()

            for conf in tqdm.tqdm(ARIMA_configs):
                # ARIMA model
                model = pm.ARIMA(order=conf,
                                suppress_warnings=True,
                                **self.additional_params)
                
                # Cross-validation objects
                cv_rolling = pm.model_selection.RollingForecastCV(h=h, step=stride)
                cv_sliding = pm.model_selection.SlidingWindowForecastCV(h=h, step=stride, window_size=window_length)

                # Fit the model
                model_rolling_scores = pm.model_selection.cross_val_score(model, self.train, cv=cv_rolling, scoring=eval_metric)
                model_sliding_scores = pm.model_selection.cross_val_score(model, self.train, cv=cv_sliding, scoring=eval_metric)

                self.cross_val_results[conf] = {'rolling': np.median(model_rolling_scores), 
                                                'sliding': np.median(model_sliding_scores)}
                
            self.cross_validation_df =pd.DataFrame.from_dict(self.cross_val_results, orient='index').reset_index()\
                .rename(columns={'level_0': 'AR_order',
                                'level_1': 'I_order',
                                'level_2': 'MA_order',
                                'rolling': 'rolling_median_MSE',
                                'sliding': 'sliding_median_MSE'})
        
        self.cross_validated = True

    def best_model(self, cross_validation_strategy:Literal['rolling', 'sliding']='rolling') -> pd.Series:
        """
        Get the best model configuration based on the cross validation results.
        Parameters
        ----------
        cross_validation_strategy : Literal['rolling', 'sliding'], optional
            The cross validation strategy to use, by default 'rolling'
        Returns
        -------
        pd.Series
            The best model configuration.
        """

        if not self.cross_validated:
            raise ValueError('Cross validation must be done first.')
        
        best_config = self.cross_validation_df.loc[self.cross_validation_df[f'{cross_validation_strategy}_median_MSE'].idxmin()]

        return tuple(best_config[['AR_order', 'I_order', 'MA_order']].values.astype(int))

    def fit(self, use_best_config:bool=True, order:Optional[tuple]=None) -> None:
        """
        Fit the ARIMA model.
        Parameters
        ----------
        use_best_config : bool, optional
            Whether to use the best configuration from cross validation, by default True
        order : tuple, optional
            The ARIMA configuration to use, by default self.best_model()
        """

        if use_best_config:
            order = self.best_model()
        elif order is None:
            raise ValueError('You must provide an ARIMA configuration to use if not using the best configuration.')
        else:
            order = order  

        self.model = pm.ARIMA(order=order,
                            suppress_warnings=True,
                            **self.additional_params)
        self.model.fit(self.train)

        self.fitted = True

        

    def predict(self, h:int, strategy:Literal['one_shot', 'iterative', 'iterative_with_update','all']='all') -> defaultdict:
        """
        Predict the next steps.
        Parameters
        ----------
        h : int
            The forecast horizon.
        strategy : Literal['one_shot', 'iterative', 'iterative_with_update', 'all'], optional
            The prediction strategy to use, by default 'all'
                'one_shot': Predict all steps at once. Number of steps = test length.
                'iterative': Predict one step at a time, including the last realized value in the next prediction
                    but freezing parameters.
                'iterative_with_update': Predict one step at a time, including the last realized value in the next prediction
                    and updating parameters.
                'all': Predict using all strategies.

        """

        if not self.fitted:
            raise ValueError('You must fit the model before predicting.')
            
        self.predictions = defaultdict()

        if strategy == 'one_shot' or strategy == 'all':
            self.predictions['one_shot'] = self.model.predict(n_periods=len(self.test),
                                                              return_conf_int=True)[0][h-1:]

        if strategy == 'iterative' or strategy == 'all':
            model_copy = cp.deepcopy(self.model)
            forecasts = []
            forecasts.append(self.model.predict(n_periods=h, return_conf_int=True)[0][-1])
            for i in range(len(self.test[:-h])):
                model_copy.update([self.test[i]], maxiter=0)
                model_copy.arima_res_.params = self.model.params()
                new_forecast = model_copy.predict(n_periods=h, return_conf_int=True)[0]
                forecasts.append(new_forecast[-1])

            self.predictions['iterative'] = np.array(forecasts)

        # if strategy == 'iterative' or strategy == 'all':
        #     model_copy = cp.deepcopy(self.model)
        #     forecasts = []
        #     forecasts.append(self.model.predict(n_periods=h, return_conf_int=True)[0][-1])
        #     for i in range(len(self.test)):
        #         model_copy.update([self.test[i]], maxiter=0)
        #         model_copy.arima_res_.params = self.model.params()
        #         new_forecast = model_copy.predict(n_periods=h, return_conf_int=True)[0]
        #         forecasts.append(new_forecast[-1])

        #     self.predictions['iterative'] = np.array(forecasts[h-1:])

        if strategy == 'iterative_with_update' or strategy == 'all':
            model_copy = cp.deepcopy(self.model)
            forecasts = []
            forecasts.append(self.model.predict(n_periods=h, return_conf_int=True)[0][-1])
            for i in range(len(self.test[:-h])):
                model_copy.update([self.test[i]], maxiter=0)
                new_forecast = model_copy.predict(n_periods=h, return_conf_int=True)[0]
                forecasts.append(new_forecast[-1])

            self.predictions['iterative_with_update'] = np.array(forecasts)

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
            self.evaluation_results[key] = forecast_evaluation(h, self.test, value)

        return self.evaluation_results
    
class EtsModel:
    def __init__(self,
                 train:np.array,
                 test:np.array,
                 **kwargs):
        
        self.train = train
        self.test = test
        self.cross_validated = False
        self.fitted = False
        self.forecasted = False
        self.additional_params = kwargs

    def cross_validate(self, 
                       ETS_configs:List[dict], 
                       h:int=1,
                       stride:int=25,
                       window_length:int=None,
                       eval_metric:Literal['smape', 'mean_squared_error', 'mean_absolute_error']='mean_squared_error') -> None:
        """
        Cross validate the ETS model with the given configurations.
        Parameters
        ----------
        ETS_configs : product
            The configurations to cross validate.
        h : int, optional
            The forecast horizon, by default 1
        stride : int, optional
            The stride size between folds, by default 25
        window_length : int, optional
            The window size for sliding window cross validation, by default None
        eval_metric : Literal['smape', 'mean_squared_error', 'mean_absolute_error'], optional
            The metric to use for evaluation, by default 'mean_squared_error'
        """

        if eval_metric == 'smape' or eval_metric == 'mean_absolute_error':
            raise ValueError('This metric is not yet supported. Please use mean_squared_error instead.')

        self.cross_val_results = defaultdict()

        for conf in tqdm.tqdm(ETS_configs):
            
            squared_error_rolling = []
            squared_error_sliding = []

            cv_sliding = pm.model_selection.SlidingWindowForecastCV(h=h, step=stride, window_size=window_length)
            cv_rolling = pm.model_selection.RollingForecastCV(h=h, step=stride)

            for train_index, val_index in cv_rolling.split(self.train):
                train_, val_ = self.train[train_index], self.train[val_index]
                try:
                    model = ETSModel(train_, **conf, **self.additional_params).fit(disp=False)
                    pred = model.forecast(steps=1)

                    squared_error_rolling.append((val_ - pred)**2)
                except:
                    pass

            for train_index, val_index in cv_sliding.split(self.train):
                train_, val_ = self.train[train_index], self.train[val_index]
                try:
                    model = ETSModel(train_, **conf, **self.additional_params).fit(disp=False)
                    pred = model.forecast(steps=1)

                    squared_error_sliding.append((val_ - pred)**2)
                except:
                    pass

            if len(squared_error_rolling) > 0 and len(squared_error_sliding) > 0:
                #Median of MSE ==> calculated using only 1 value in validation set
                self.cross_val_results[str(conf)] = {'rolling_median_MSE':np.round(np.median(squared_error_rolling),5),
                                                'sliding_median_MSE':np.round(np.median(squared_error_sliding),5)}
                
        self.cross_validation_df  = pd.DataFrame.from_dict(self.cross_val_results, \
                                                                      orient='index').reset_index()
        
        self.cross_validated = True

    def best_model(self, cross_validation_strategy:Literal['rolling', 'sliding']='rolling') -> pd.Series:
        """
        Get the best model configuration based on the cross validation results.
        Parameters
        ----------
        cross_validation_strategy : Literal['rolling', 'sliding'], optional
            The cross validation strategy to use, by default 'rolling'
        Returns
        -------
        pd.Series
            The best model configuration.
        """

        if not self.cross_validated:
            raise ValueError('Cross validation must be done first.')

        return self.cross_validation_df.loc[self.cross_validation_df[f'{cross_validation_strategy}_median_MSE'].idxmin()]
    
    def fit(self, use_best_config:bool=True, config:Optional[dict]=None) -> None:
        """
        Fit the ETS model.
        Parameters
        ----------
        h : int
            The forecast horizon.
        use_best_config : bool, optional
            Whether to use the best configuration from cross validation, by default True
        config : dict, optional
            The ETS configuration to use, by default self.best_model()
        """

        if use_best_config:
            self._config = ast.literal_eval(self.best_model()['index'])
        elif config is None:
            raise ValueError('You must provide an ETS configuration to use if not using the best configuration.')
        else:
            self._config = config

        self.model = ETSModel(self.train, **self._config, **self.additional_params).fit(disp=False)

        self.fitted = True

    def predict(self, h:int, strategy:Literal['one_shot', 'iterative', 'iterative_with_update','all']='all') -> defaultdict:
        """
        Predict the next steps.
        Parameters
        ----------
        h : int
            The forecast horizon.
        strategy : Literal['one_shot', 'iterative', 'iterative_with_update', 'all'], optional
            The prediction strategy to use, by default 'all'
                'one_shot': Predict all steps at once. Number of steps = test length.
                'iterative': Predict one step at a time, including the last realized value in the next prediction
                    but freezing parameters.
                'iterative_with_update': Predict one step at a time, including the last realized value in the next prediction
                    and updating parameters.
                'all': Predict using all strategies.

        """

        if not self.fitted:
            raise ValueError('You must fit the model first.')
            
        self.predictions = defaultdict()

        if strategy == 'one_shot' or strategy == 'all':
            self.predictions['one_shot'] = self.model.forecast(steps=len(self.test))[h-1:]

        if strategy == 'iterative' or strategy == 'all':
            forecasts = []
            best_model_params = {key:value for key, value in \
                                 zip(self.model.model.param_names, self.model.params)}
            for i in range(len(self.test)-h+1):
                new_model = ETSModel(np.r_[self.train,self.test[:i]], 
                    **self._config, **self.additional_params)
                with new_model.fix_params(best_model_params):
                    #print(new_model.fit(disp=False).summary())
                    new_forecast = new_model.fit(disp=False).forecast(steps=h)[-1]
                forecasts.append(new_forecast)

            self.predictions['iterative'] = np.array(forecasts)

        if strategy == 'iterative_with_update' or strategy == 'all':
            forecasts = []
            for i in range(len(self.test)-h+1):
                new_model = ETSModel(np.r_[self.train,self.test[:i]], 
                    **self._config, **self.additional_params).fit(disp=False)
                new_forecast = new_model.forecast(steps=h)[-1]
                forecasts.append(new_forecast)

            self.predictions['iterative_with_update'] = np.array(forecasts)
        
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
            self.evaluation_results[key] = forecast_evaluation(h, self.test, value)

        return self.evaluation_results