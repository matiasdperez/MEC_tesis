import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Literal, Union
import statsmodels.api as sm
from tqdm import tqdm
import scipy.linalg as la

sns.set_style("darkgrid")

class Simulator:
    """
    Base class for simulation of time series data.
    """
    def __init__(self, n_steps:int = 1000, initial_state:float = 0.0 ,random_seed:int = 42):
        """
        Parameters:
        -----------
        n_steps: int
            Number of steps to simulate
        initial_state: float
            Initial value of the time series
        random_seed: int
            Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.random_seed = random_seed
        self.initial_state = initial_state

    def setting(self):
        """
        Set the x and y values for the simulation
        """
        self.x = np.arange(self.n_steps)
        self.y = np.zeros(self.n_steps)
        self.y[0] = self.initial_state

    def simulate(self):
        # Basic implementation, to be overridden by child classes
        pass

    def plot(self):
        # Basic implementation, to be overridden by child classes
        pass

class RandomWalk(Simulator):
    """
    Class for simulating a random walk process.
    """
    def __init__(self, n_steps:int = 1000, initial_state:float = 0.0, drift:float = 0.0, trend:float=0.0, random_seed:int = 42,):
        """
        Parameters:
        -----------
        n_steps: int
            Number of steps to simulate
        initial_state: float
            Initial value of the time series
        drift: float
            Drift parameter
        trend: float
            Trend parameter
        random_seed: int
            Random seed for reproducibility
        """
        super().__init__()
        self.n_steps = n_steps
        self.random_seed = random_seed
        self.initial_state = initial_state
        self.drift = drift
        self.trend = trend
        self.simulated = False
        self.setting()

    def simulate(self, distribution:str = 'Gaussian', **kwargs):
        """
        Simulate the random walk process.
        Parameters:
        -----------
        distribution: str
            Distribution to use for the noise term. So far, only 'Gaussian' is supported.
        **kwargs: dict
            Additional parameters for the noise term.
            - loc: float
                Mean of the distribution
            - scale: float
                Standard deviation of the distribution
        """

        np.random.seed(self.random_seed)

        if distribution == 'Gaussian':
            for i in range(1, self.n_steps):
                self.y[i] = self.y[i-1] + self.drift + self.trend*self.x[i] + np.random.normal(**kwargs)
        else:
            raise ValueError('Distribution not supported yet. Please use Gaussian.')
        
        self.simulated = True
        
    def regime_switch(self, t_switch:int, new_drift: Optional[float] = None, new_trend:Optional[float] = None ,**kwargs):

        """
        Change the drift and trend parameters of the random walk process at a given time step.
        Parameters:
        -----------
        t_switch: int
            Time step at which to change the drift and trend parameters
        new_drift: float
            New drift parameter
        new_trend: float
            New trend parameter
        **kwargs: dict
            Additional parameters for the noise term.
            - loc: float
                Mean of the distribution
            - scale: float
                Standard deviation of the distribution
        """

        assert t_switch < self.n_steps, 't_switch must be less than n_steps'
        if not self.simulated:
            raise Exception('Please simulate the process first.')
        
        new_drift = self.drift if new_drift is None else new_drift
        new_trend = self.trend if new_trend is None else new_trend
        
        np.random.seed(self.random_seed)

        for i in range(t_switch, self.n_steps):
            self.y[i] = self.y[i-1] + new_drift + new_trend*self.x[i] + np.random.normal(**kwargs)

    def plot(self):

        """
        Plot the simulated random walk process.
        """     
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x = self.x, y = self.y, lw=0.3)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.axhline(0, color='red', lw=0.3, ls='--')
        if self.trend != 0 and self.drift != 0:
            plt.title(f'Random Walk with trend = {self.trend} and drift = {self.drift}')
        else:
            plt.title('Random Walk')

class ARIMA(Simulator):
    """
    Class for simulating ARIMA processes. This class allows for simulating ARIMA processes with different orders of AR, I (only 0,1 and 2 are admitted), and MA components.
    """
    def __init__(self, n_steps:int = 1000, ARIMA_model:List[int] = [1,0,1], AR_params:List[float] = [0.5], MA_params:List[float] = [0.5], \
                 initial_y_value:float = 0.0, random_seed:int = 42, **kwargs):
        """
        Parameters:
        -----------
        n_steps: int
            Number of steps to simulate.
        ARIMA_model: List[int], default=[1,0,1]
            List of integers specifying the AR, I, and MA orders of the ARIMA model. It must have length 3.
        AR_params: List[float], default=[0.5]
            List of floats specifying the AR parameters of the ARIMA model. It must have length equal to the AR order.
        MA_params: List[float], default=[0.5]
            List of floats specifying the MA parameters of the ARIMA model. It must have length equal to the MA order.
        initial_y_value: float, default=0.0
            Initial value of the time series.
        random_seed: int, default=42
            Random seed for reproducibility.
        **kwargs: dict
            Additional arguments for the random number generator.
            - 'constant': float
                Intercept parameter for y_t
            - 'trend': float
                Deterministic for y_t
            - 'sigma_y': float
                Standard deviation for y_t
            
        """
        super().__init__()
        self.n_steps = n_steps
        self.initial_y_value = initial_y_value 
        if len(ARIMA_model) != 3:
            raise ValueError("ARIMA model specification must have length 3.")
        
        self.AR_order, self.I_order, self.MA_order = ARIMA_model

        if len(AR_params) != self.AR_order:
            raise ValueError("AR parameters must have length equal to ARIMA model's AR order.")
        self.AR_params = AR_params

        if len(MA_params) != self.MA_order:
            raise ValueError("MA parameters must have length equal to ARIMA model's MA order.")
        self.MA_params = MA_params

        self.additional_parameters = kwargs
        
        self.random_seed = random_seed
        self.simulated = False
        self.regime_switched = False
        self.setting()
        
    def _simulate(self, distribution:str = 'Gaussian', initial_values:Optional[List[float]]=[], burnin:bool = True, **kwargs):

        """
        Simulates the ARIMA process.
        Parameters:
        -----------
        distribution: str, default='Gaussian'
            Distribution of the errors. So far, only Gaussian is supported.
        initial_values: List[float], default=[]
            Initial values of the series. If empty, the series is generated from the ARIMA model.
        burnin: bool, default=True
            If True, a burn-in period is used to generate initial values of the series but then discarded.
        """

        np.random.seed(self.random_seed)

        if len(initial_values) == 0:
            y = self.y.copy()
        else:
            y = self.y.copy()
            y[0] = initial_values[0]
        delta_y = np.zeros(self.n_steps-self.I_order) # used in integrated models
        x = np.arange(len(y))

        constant = kwargs.get('constant', 0)
        trend = kwargs.get('trend', 0)
        sigma_y = kwargs.get('sigma_y', 1)
        initial_t = kwargs.get('initial_t', 1)

        if distribution == 'Gaussian':
            elf = np.random.normal(size=self.n_steps, loc=0, scale=sigma_y) # iid errors
            if self.I_order == 0:
                    for i in tqdm(range(initial_t, self.n_steps), initial=1, desc='Simulating'):
                        burnin_period = max(self.AR_order, self.MA_order)
                        if i < burnin_period: # burn-in period, used to generate initial values of the series
                            if len(initial_values) > 0:
                                y[i] = initial_values[i]
                            else:
                                y[i] = np.random.normal(y[i-1], 0.1) # initial values are generated as a random walk. Then, they are discarded.
                        else:
                            y[i] = constant + trend*i + y[i-self.AR_order:i].T@self.AR_params[::-1] + elf[i-self.MA_order:i+1].T@np.r_[self.MA_params[::-1], 1]
            elif self.I_order <= 2:
                for i in tqdm(range(initial_t, self.n_steps), initial=1, desc='Simulating'):
                    burnin_period = max(self.AR_order, self.MA_order) + self.I_order
                    if i < burnin_period-1:
                        if len(initial_values) > 0:
                            y[i] = initial_values[i]
                        else:
                            y[i] = np.random.normal(y[i-1], 0.1)
                    elif i == burnin_period-1:
                        if len(initial_values) > 0:
                            y[i] = initial_values[i-1]
                        else:
                            y[i] = np.random.normal(y[i-1], 0.1)
                            delta_y = np.diff(y, n=self.I_order)
                            delta_y[burnin_period-self.I_order:] = 0
                    else:
                        if self.AR_order == 0 and self.MA_order == 0:
                            raise ValueError('Only models with AR or MA orders greater than 0 are supported.\n Please use RandomWalk class for models with no AR or MA orders.')
                        else:
                            delta_y[i-self.I_order] = delta_y[i-self.I_order-self.AR_order:i-self.I_order].T@self.AR_params[::-1] + \
                                elf[i-self.MA_order:i+1].T@np.r_[self.MA_params[::-1], 1]
                        y[i] = constant + trend*i + self.I_order*y[i-1] - (self.I_order-1)*y[i-2] + delta_y[i-self.I_order]
            else:
                raise ValueError('Only models with an integration order less than or equal to 2 are supported.')
        else:   
            raise ValueError('Distribution not supported yet. Please use Gaussian.')
        
        if burnin: # remove burn-in period
            y = y[burnin_period:]
            x = np.arange(len(y))
            try:
                delta_y = delta_y[burnin_period:]
            except:
                pass

        self.simulated = True

        return x, y, delta_y

    def simulate(self, distribution:str = 'Gaussian', initial_values:Optional[List[float]]=[], burnin:bool = False):
        self.x, self.y, self.delta_y = self._simulate(distribution, initial_values, burnin, **self.additional_parameters)

    def regime_switch(self, 
                      t_switch:int, 
                      switch_type:Literal['ARIMA_parameters','error_distribution' ,'both'],
                      new_ARIMA_model:Optional[List[int]]=[1,0,1],
                      new_AR_params:Optional[List[float]]=[0.5],
                      new_MA_params:Optional[List[float]]=[0.5],
                      **kwargs):
        
        """
        This functions apply an idiosyncratic regime switch to the ARIMA process already simulated.
        Parameters:
        -----------
        t_switch: int
            Time period when the regime switch occurs. It must be less than n_steps.
        switch_type: str
            Type of regime switch. It can be 'ARIMA_parameters','error_distribution' or 'both'. In 'ARIMA_parameters', the ARIMA parameters are changed. 
            In 'error_distribution', the distribution of the errors is changed (so far, only changes in the variance of y are admitted). In 'both', both 
            the ARIMA parameters and the distribution of the errors are changed.
        new_ARIMA_model: List[int], default=[1,0,1]
            New ARIMA model specification. It must have length 3.
        new_AR_params: List[float], default=[0.5]
            New AR parameters. It must have length equal to the AR order of the new ARIMA model.
        new_MA_params: List[float], default=[0.5]
            New MA parameters. It must have length equal to the MA order of the new ARIMA model.
        **kwargs: dict
            - 'constant': float
                Intercept parameter for y_t
            - 'trend': float
                Deterministic for y_t
            - 'sigma_y': float
                Standard deviation for y_t
        """
        
        if not self.simulated:
            raise Exception('Please simulate the process first.')
        if len(new_ARIMA_model) != 3:
            raise ValueError("ARIMA model specification must have length 3.")
        if len(new_AR_params) != new_ARIMA_model[0]:
            raise ValueError("AR parameters must have length equal to ARIMA model's AR order.")
        if len(new_MA_params) != new_ARIMA_model[2]:
            raise ValueError("MA parameters must have length equal to ARIMA model's MA order.")
        if t_switch >= self.n_steps:    
            raise ValueError('t_switch must be less than n_steps.')
        
        np.random.seed(self.random_seed)

        if switch_type == 'ARIMA_parameters':

            self.new_AR_order, self.new_I_order, self.new_MA_order = new_ARIMA_model
            self.new_AR_params = new_AR_params
            self.new_MA_params = new_MA_params

            burnin_period = max(self.new_AR_order, self.new_MA_order) + self.new_I_order
            initial_values = self.y[t_switch-burnin_period:t_switch]
            
            self.x, self.new_y, self.new_delta_y = self._simulate('Gaussian', initial_values, burnin=False, **kwargs)
            self.y[t_switch:] = self.new_y[burnin_period+1:self.n_steps-t_switch+burnin_period+1]
            length_delta_y = len(self.delta_y)
            self.delta_y[t_switch:] = self.new_delta_y[burnin_period+1:length_delta_y-t_switch+burnin_period+1]

        elif switch_type == 'error_distribution':

            burnin_period = max(self.AR_order, self.MA_order) + self.I_order
            initial_values = self.y[t_switch-burnin_period:t_switch]

            self.x, self.new_y, self.new_delta_y = self._simulate('Gaussian', initial_values, burnin=False, **kwargs)
            self.y[t_switch:] = self.new_y[burnin_period+1:self.n_steps-t_switch+burnin_period+1]
            length_delta_y = len(self.delta_y)
            self.delta_y[t_switch:] = self.new_delta_y[burnin_period+1:length_delta_y-t_switch+burnin_period+1]

        else:
            self.AR_order, self.I_order, self.MA_order = new_ARIMA_model
            self.AR_params = new_AR_params
            self.MA_params = new_MA_params

            burnin_period = max(self.AR_order, self.MA_order) + self.I_order
            initial_values = self.y[t_switch-burnin_period:t_switch]
            
            self.x, self.new_y, self.new_delta_y = self._simulate('Gaussian', initial_values, burnin=False, **kwargs)
            self.y[t_switch:] = self.new_y[burnin_period+1:self.n_steps-t_switch+burnin_period+1]
            length_delta_y = len(self.delta_y)
            self.delta_y[t_switch:] = self.new_delta_y[burnin_period+1:length_delta_y-t_switch+burnin_period+1]

        self.regime_switched = True

    def plot(self):
        """
        Plot the simulated time series.
        """
        if self.I_order == 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', lw=0.3)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=ax1, lw=0.3)
            ax1.axhline(0, color='black', lw=0.5, ls='--')
            ax1.set_xlabel('Time')
            sns.lineplot(x = self.x, y = np.r_[[0]*(len(self.x)-len(self.delta_y)),self.delta_y], label = f'$Δ^{self.I_order}y$', ax=ax2, color='red', lw=0.3)
            ax2.axhline(0, color='black', lw=0.5, ls='--')
            ax2.set_xlabel('Time')
        if not self.regime_switched:
            plt.suptitle(f'ARIMA {(self.AR_order, self.I_order, self.MA_order)} model')
        else:
            plt.suptitle('Simulated data with regime switch')
        plt.tight_layout()

class MarkovSwitchingModel(Simulator):
    """
    Class to simulate a Markov switching model with a given transition matrix and state space dimension. The dimension of the state space is finite and discrete and each
    state represents a different regime. The transition matrix is a square matrix with rows that sum to 1 (since it's a pmf). This class implements 4 different models:

    1) 'intercept' model: the model has a constant intercept μ_s for each regime s
        y_t = μ_s + ε_t, ε_t ~ N(0, σ^2) for t = 1, ..., T (default σ^2=1)
    2) 'AR_1' model: the model has an autoregressive parameter φ_s for each regime s
        y_t = φ_s*y_{t-1} + ε_t, ε_t ~ N(0, σ^2) for t = 1, ..., T (default σ^2=1)
    3) 'AR_1_with_intercept' model: the model has an autoregressive parameter φ_s and an intercept μ_s for each regime s
        y_t = μ_s + φ_s*y_{t-1} + ε_t, ε_t ~ N(0, σ^2) for t = 1, ..., T (default σ^2=1)
    4) 'variance' model: the model has a variance σ^2_s for each regime s
        y_t = ε_t, ε_t ~ N(0, σ^2_s) for t = 1, ..., T
    5) 'all' model: the model has all the parameters above
        y_t = μ_s + φ_s*y_{t-1} + ε_t, ε_t ~ N(0, σ^2_s) for t = 1, ..., T
    """
    def __init__(self, 
                 n_steps:int = 1000, 
                model_type:Literal['intercept','AR_1','AR_1_with_intercept','variance','all']='intercept',
                state_space_dim:int = 2,
                transition_matrix:List[List[float]] = [[0.9, 0.1], [0.1, 0.9]],
                intercept:List[float] = [0, 0],
                AR_1_parameter:Optional[List[float]] = None,
                variance:Optional[List[float]] = None,
                initial_y_value:float = 0,
                random_seed:int = 42,
                **kwargs):
        """
        Parameters:
        -----------
        n_steps: int
            Number of steps to simulate
        model_type: str
            Type of model to simulate. Choose among 'intercept', 'AR_1', 'AR_1_with_intercept', 'variance' or 'all'.
        state_space_dim: int, default=2
            Dimension of the state space. Each state (aka. regime) will be indexed from 0 to state_space_dim-1.
        transition_matrix: List[List[float]], default=[[0.9, 0.1], [0.1, 0.9]]
            Transition matrix of the Markov chain. It must be a square matrix of size state_space_dim with rows that sum to 1.
        intercept: List[float], default=[0, 0]
            List of intercepts for each regime. Must be of size state_space_dim.
        AR_1_parameter: Optional[List[float]], default=None
            List of autoregressive parameters for each regime. Must be of size state_space_dim.
        variance: Optional[List[float]], default=None
            List of variances for each regime. Must be of size state_space_dim.
        initial_y_value: float, default=0
            Initial value of the time series
        **kwargs: dict
            Additional and optional arguments to be passed to the base class. This dictionary can include the following keys:
            - 'constant': float
                Intercept parameter for y_t
            - 'trend': float
                Deterministic for y_t
            - 'sigma_y': float
                Standard deviation for y_t

            
        """
        super().__init__()
        self.n_steps = n_steps
        self.model_type = model_type
        self.state_space_dim = state_space_dim
        if len(transition_matrix) == self.state_space_dim and all(len(sub) == self.state_space_dim for sub in transition_matrix):
            if all(np.array(transition_matrix).sum(axis=1) == 1):
                self.transition_matrix = transition_matrix
            else:
                raise ValueError('Rows of the transition matrix must sum to 1 since they represent probabilities of transitioning to other states')
        else:
            raise ValueError('Transition matrix must be a square matrix of size state_space_dim')
        
        if len(intercept) == self.state_space_dim:
            self.intercept = intercept
        else:
            raise ValueError('Intercept must be a list of size state_space_dim')
        if AR_1_parameter is not None:
            if len(AR_1_parameter) == self.state_space_dim:
                self.AR_1_parameter = AR_1_parameter
            else:
                raise ValueError('AR_1_parameter must be a list of size state_space_dim')
        else:
            pass
        if variance is not None:
            if len(variance) == self.state_space_dim:
                self.variance = variance
            else:
                raise ValueError('Variance must be a list of size state_space_dim')
            
        self.x = np.arange(self.n_steps)
        self.y = np.zeros(self.n_steps)
        self.state = np.zeros(self.n_steps).astype(int)
        self.initial_y_value = initial_y_value
        self.additional_parameters = kwargs

        self.random_seed = random_seed
        self.setting()

    def _steady_state_distribution(self):
        """
        Compute the steady state distribution of the Markov chain given the transition matrix. The steady state distribution is the left eigenvector of the transition matrix
        corresponding to the eigenvalue 1. If the eigenvalue 1 is unique and the eigenvector is unique, then the Markov chain has a unique steady state distribution. Otherwise, 
        the steady state distribution does not exist or it's not unique.
        """
        
        eigvalues, eigvectors = la.eig(self.transition_matrix, right=False, left=True)
        idx = eigvalues.argsort()[::-1]  
        eigvalues = eigvalues[idx]
        eigvectors = eigvectors[:,idx]

        if max(eigvalues.real)==1 and all(eigvalues.real[1:]<1):
            print("Steady state distribution exists and it's unique.")
            self.steady_state_distribution = eigvectors[:,0]/sum(eigvectors[:,0])
            return self.steady_state_distribution
        else:   
            print("Steady state distribution does not exist or it's not unique.")
            return None
        
    def simulate(self, initial_state:Union[int, Literal['steady_state','random']], distribution:str='Gaussian'):

        """
        Simulate the time series according to the specified model type and parameters.
        Parameters:
        -----------
        initial_state: Union[int, Literal['steady_state','random']]
            Initial state of the Markov chain. If 'steady_state', the initial state will be sampled from the steady state distribution. If 'random', 
            the initial state will be sampled uniformly at random. If an integer is provided, the initial state will be set to that integer.
        distribution: str, default='Gaussian'
            Distribution of the noise term. So far, only 'Gaussian' is supported.
        """

        state_space = np.arange(self.state_space_dim)
        np.random.seed(self.random_seed)

        if initial_state == 'steady_state':
            self.steady_state_distribution = self._steady_state_distribution()
            if self.steady_state_distribution is not None:
                initial_state = np.random.choice(state_space, p=self.steady_state_distribution)
            else:
                raise ValueError('Steady state distribution does not exist or it is not unique. \n \
                                 Please provide initial_state as an integer or use option "random"')
        elif initial_state == 'random':
            initial_state = np.random.choice(state_space)
        else:
            initial_state = initial_state

        self.state[0] = initial_state
        self.y[0] = self.initial_y_value
        constant = self.additional_parameters.get('constant', 0)
        trend = self.additional_parameters.get('trend', 0)
        sigma_y = self.additional_parameters.get('sigma_y', 1) 

        if distribution == 'Gaussian':
            if self.model_type == 'intercept':
                for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
                    self.state[t] = np.random.choice(state_space, p=self.transition_matrix[self.state[t-1]])
                    self.y[t] = np.random.normal(loc=constant + trend*t + self.intercept[self.state[t]], scale=sigma_y)
            elif self.model_type == 'AR_1':
                for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
                    self.state[t] = np.random.choice(state_space, p=self.transition_matrix[int(self.state[t-1])])
                    self.y[t] = np.random.normal(loc= constant + trend*t + self.AR_1_parameter[self.state[t]]*self.y[t-1], scale=sigma_y)        
            elif self.model_type == 'AR_1_with_intercept':
                for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
                    self.state[t] = np.random.choice(state_space, p=self.transition_matrix[int(self.state[t-1])])
                    self.y[t] = np.random.normal(loc=constant + trend*t + self.intercept[self.state[t]] + self.AR_1_parameter[self.state[t]]*self.y[t-1], scale=sigma_y)
            elif self.model_type == 'variance':
                for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
                    self.state[t] = np.random.choice(state_space, p=self.transition_matrix[self.state[t-1]])
                    self.y[t] = np.random.normal(loc= constant + trend*t, scale=self.variance[self.state[t]])
            else:
                for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
                    self.state[t] = np.random.choice(state_space, p=self.transition_matrix[self.state[t-1]])
                    self.y[t] = np.random.normal(loc=constant + trend*t + self.intercept[self.state[t]] + self.AR_1_parameter[self.state[t]]*self.y[t-1], \
                                                scale=self.variance[self.state[t]])
        else:
            raise ValueError('Only Gaussian distribution is supported.')
        

    def plot(self):
        """
        Plot the simulated time series and the state sequence.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        sns.lineplot(x = self.x, y = self.y, label = 'y', ax=ax1, lw=0.3)
        ax1.set_xlabel('Time')
        sns.lineplot(x = self.x, y = self.state, label = 'state', ax=ax2, color='red', lw=0.3)
        ax2.set_xlabel('Time')
        plt.suptitle(f'Simulated data with Markov Switching process in {self.model_type} parameter')
        plt.tight_layout()

class StateSpaceModel(Simulator):
    """
    Class to simulate a state space model with time-varying parameters. In all these models, the state space is assumed to be
    continuous and time is discrete. Depending of the model type chose, the state space model is defined by the following equations:

    1) 'intercept' model (aka. Local Linear Trend model):
        Observation equation: y_t = α + β*t + μ_t + ε_t / ε_t~IID N(0,σ_y) (default: σ_y=1, α=0, β=0)
        State equation: μ_t = ω + κ*t + AR(p) + η_t / η_t~IID N(0,σ_μ) (default: σ_μ=1, ω=0, κ=0) ==> μ_t follows a AR(p) process

    2) 'AR' model (AR(1) model on the observed variable y):
        Observation equation: y_t = α + β*t + ϕ_t*y_{t-1} + ε_t / ε_t~IID N(0,σ_y) (default: σ_y=1, α=0, β=0)
        State equation: ϕ_t = ω + κ*t + AR(p) + η_t / η_t~IID N(0,σ_ϕ) (default: σ_ϕ=1, ω=0, κ=0) ==> ϕ_t follows a AR(p) process

    3) 'AR_with_intercept' model:
        Observation equation: y_t = α + β*t + μ_t + ϕ_t*y_{t-1} + ε_t / ε_t~IID N(0,σ_y) (default: σ_y=1, α=0, β=0)
        State equations: μ_t = ω_1 + κ_1*t + AR(p) + η_t / η_t~IID N(0,σ_μ) (default: σ_μ=1, ω_1=κ_1=0) ==> μ_t follows a AR(p) process
                         ϕ_t = ω_2 + κ_2*t + AR(p) + ζ_t / ζ_t~IID N(0,σ_ϕ) (default: σ_ϕ=1, ω_2=κ_2=0) ==> ϕ_t follows a AR(p) process

    4) 'variance' model (aka. GARCH model):
        Observation equation: y_t = α + β*t + μ_t + ε_t / ε_t~IID N(0,σ_t) ==> σ_t is time-varying (default: α=0, β=0)
        State equations: σ^2_t = ω + AR(σ^2_t, p) + AR(ε^2_t, q) ==> both σ^2_t and ε^2_t follow an AR(p) and 
                                                                    AR(q) process respectively (default p=q=1, ω=0)
                        ε_t = μ_t*σ_t / μ_t~IID N(0,1)

    5) 'all' model:
        Observation equation: y_t = α + β*t + μ_t + ϕ_t*y_{t-1} + ε_t / ε_t~IID N(0,σ_t) (default: α=0, β=0)
        State equations: μ_t = ω_1 + κ_1*t + AR(p) + η_t / η_t~IID N(0,σ_μ) (default ω_1=κ_1=0)
                         ϕ_t = ω_2 + κ_2*t + AR(p) + ζ_t / ζ_t~IID N(0,σ_ϕ) (default ω_2=κ_2=0)
                         σ^2_t = ω_3 + AR(σ^2_t, p) + AR(ε^2_t, q)          (default ω_3=0)
                         ε_t = μ_t*σ_t / μ_t~IID N(0,1)
    """

    def __init__(self, 
            n_steps:int = 1000, 
            time_varying_parameters:Literal['intercept','AR','AR_with_intercept','variance','all']='intercept',
            intercept_parameters:Optional[List[float]] = None,
            AR_parameters:Optional[List[float]] = None,
            variance_parameters:Optional[List[float]] = None,
            error_parameters:Optional[List[float]] = None,
            initial_y_value:float = 0,
            random_seed:int = 42,
            **kwargs):

        """
        Parameters:
        -----------
        n_steps: int
            Number of steps to simulate the model.
        time_varying_parameters: str
            Type of model to simulate. Choose between 'intercept', 'AR', 'AR_with_intercept', 'variance', 'all'.
        intercept_parameters: Optional[List[float]]
            Parameters to build the state equation for the intercept model. The number of parameters included in the list
            will determine the order of the AR process that dictates the evolution of the intercept parameter.
        AR_parameters: Optional[List[float]]
            Parameters to build the state equation for the AR(1) model (aka. ϕ_t) that follows y_t. The number of parameters 
            included in the list will determine the order of the AR process that dictates the evolution of the ϕ_t.
        variance_parameters: Optional[List[float]]
            Parameters to build the first state equation for the variance model (aka. GARCH model). The number of parameters included 
            in the list will determine the order of the AR process that dictates the evolution of the variance parameter.
        error_parameters: Optional[List[float]]
            Parameters to build the second state equation for the variance model (aka. GARCH model). The number of parameters included 
            in the list will determine the order of the AR process that dictates the evolution of the error parameter.
        initial_y_value: float
            Initial value for the observed variable y.
        random_seed: int
            Random seed to reproduce the results.
        **kwargs: dict
            Additional and optional arguments to be passed to the base class. This dictionary can include the following keys:
                - 'constant': float
                    Intercept parameter for the observation equation.
                - 'trend': float
                    Deterministic for the observation equation.
                - 'sigma_y': float
                    Standard deviation for the observation equation (only used when time_varying_parameters is not 'variance' or 'all').
                - 'intercept_constant': float
                    Intercept parameter for the state equation of the intercept model.
                - 'intercept_trend': float
                    Deterministic trend for the state equation of the intercept model.
                - 'intercept_sigma': float
                    Standard deviation for the state equation of the intercept model.
                - 'AR_constant': float
                    Intercept parameter for the state equation of the AR model.
                - 'AR_trend': float
                    Deterministic trend for the state equation of the AR model.
                - 'AR_sigma': float
                    Standard deviation for the state equation of the AR model.
                - 'variance_constant': float
                    Intercept parameter for the state equation of the variance model.
        """
        
        super().__init__()
        self.n_steps = n_steps
        self.time_varying_parameters = time_varying_parameters

        if self.time_varying_parameters not in ('intercept','AR','AR_with_intercept','variance','all'):
            raise ValueError('Model not implemented yet. Please choose one of the following: intercept, AR, AR_with_intercept, variance, all.')
        if self.time_varying_parameters in ('intercept','AR_with_intercept','all'):
            if intercept_parameters is not None:
                self.intercept_parameters = intercept_parameters
            else:
                raise ValueError('Intercept parameter must be provided to build the state equation for this model.')
        if self.time_varying_parameters in ('AR','AR_with_intercept','all'):
            if AR_parameters is not None:
                self.AR_parameters = AR_parameters
            else:
                raise ValueError('AR parameter must be provided to build the state equation for this model.')
        if self.time_varying_parameters in ('variance','all'):
            if variance_parameters is not None and error_parameters is not None:
                self.variance_parameters = variance_parameters
                self.error_parameters = error_parameters
            else:
                raise ValueError('Variance parameters and errors parameters must be provided to build the state equation for this model.')

        self.additional_parameters = kwargs

        self.x = np.arange(self.n_steps)
        self.y = np.zeros(self.n_steps)
        self.y[0] = initial_y_value


    def simulate(self, 
                 intercept_initial_values:Optional[List[float]]=None, 
                 AR_initial_values:Optional[List[float]]=None,
                 variance_initial_values:Optional[List[float]]=None,
                 error_initial_values:Optional[List[float]]=None,
                 ):

        """
        Simulate the time series model.
        Parameters:
        -----------
        intercept_initial_values: Optional[List[float]]
            Initial values for the intercept model. Must be a list of size equal to intercept_parameters.
        AR_initial_values: Optional[List[float]]
            Initial values for the AR model. Must be a list of size equal to AR_parameters.
        variance_initial_values: Optional[List[float]]
            Initial values for σ^2_t. Must be a list of size equal to variance_parameters.
        error_initial_values: Optional[List[float]]
            Initial values for ε^2_t. Must be a list of size equal to error_parameters.
        """

        np.random.seed(self.random_seed)

        ## State Equations
        ### Intercept
        if self.time_varying_parameters in ('intercept','AR_with_intercept','all'):
            if intercept_initial_values is None:
                raise ValueError('Initial values for time-varying intercept must be provided and it must be a list of size equal to intercept_parameters.')
            elif len(intercept_initial_values) != len(self.intercept_parameters):
                raise ValueError('Initial values for time-varying intercept must be of equal size than intercept_parameters.')
            else:
                self.intercept = np.zeros(self.n_steps + len(self.intercept_parameters)-1)
                self.intercept[:len(self.intercept_parameters)] = intercept_initial_values
                for t in range(len(self.intercept_parameters), self.n_steps):
                    constant = self.additional_parameters.get('intercept_constant', 0)
                    deterministic_trend = self.additional_parameters.get('intercept_trend', 0)
                    sigma = self.additional_parameters.get('intercept_sigma', 1)
                    self.intercept[t] = constant + deterministic_trend*(t-len(self.intercept_parameters)) +\
                        self.intercept[t-len(self.intercept_parameters):t].T@self.intercept_parameters + np.random.normal(loc=0, scale=sigma)
                    
        ### AR(1) parameter
        if self.time_varying_parameters in ('AR','AR_with_intercept','all'):
            if AR_initial_values is None:
                raise ValueError('Initial value for time-varying AR parameter must be provided and it must be a list of size equal to AR_parameters.')
            elif len(AR_initial_values) != len(self.AR_parameters):
                raise ValueError('Initial value for time-varying AR parameter must be of equal size than AR_parameters.')
            else:
                self.AR = np.zeros(self.n_steps + len(self.AR_parameters)-1)
                self.AR[:len(self.AR_parameters)] = AR_initial_values
                for t in range(len(self.AR_parameters), self.n_steps):
                    constant = self.additional_parameters.get('AR_constant', 0)
                    deterministic_trend = self.additional_parameters.get('AR_trend', 0)
                    sigma = self.additional_parameters.get('AR_sigma', 1)
                    self.AR[t] = constant + deterministic_trend*(t-len(self.AR_parameters)) +\
                        self.AR[t-len(self.AR_parameters):t].T@self.AR_parameters + np.random.normal(loc = 0, scale = sigma)
                    
        ### Variance ==> State Space Model with GARCH errors
        if self.time_varying_parameters in ('variance','all'):
            if variance_initial_values is None or error_initial_values is None:
                raise ValueError('Initial values for time-varying variance parameter and error term must be \
                                 provided and they must be a list of size equal to variance_parameters and error_parameters respectively.')
            elif len(variance_initial_values) != len(self.variance_parameters) or len(error_initial_values) != len(self.error_parameters):
                raise ValueError('Mismatch in the size of initial values and the number of parameters passed.')
            else:
                self.errors = np.zeros(self.n_steps + len(self.variance_parameters)-1)
                self.sigma2 = np.zeros(self.n_steps + len(self.variance_parameters)-1)

                self.sigma2[:len(self.variance_parameters)] = variance_initial_values
                self.errors[:len(self.error_parameters)] = error_initial_values

                for t in range(len(self.variance_parameters), self.n_steps):
                    constant = self.additional_parameters.get('variance_constant', 0)
                    self.sigma2[t] = constant + self.sigma2[t-len(self.variance_parameters):t].T@self.variance_parameters \
                        + ((self.errors[t-len(self.error_parameters):t].T)**2)@self.error_parameters
                    self.errors[t] = np.random.normal()*np.sqrt(self.sigma2[t])
                    
        ## Observation Equation
        constant = self.additional_parameters.get('constant', 0)
        deterministic_trend = self.additional_parameters.get('trend', 0)
        sigma_y = self.additional_parameters.get('sigma_y', 1)
        for t in tqdm(range(1, self.n_steps), initial=1, desc='Simulating'):
            if self.time_varying_parameters == 'intercept':
                self.y[t] = constant + deterministic_trend*(t-1) + self.intercept[t+len(self.intercept_parameters)-1] + np.random.normal(loc=0, scale=sigma_y)
            elif self.time_varying_parameters == 'AR':
                self.y[t] = constant + deterministic_trend*(t-1) + self.AR[t+len(self.AR_parameters)-1]*self.y[t-1] + np.random.normal(loc=0, scale=sigma_y)
            elif self.time_varying_parameters == 'AR_with_intercept':
                self.y[t] = constant + deterministic_trend*(t-1) + self.intercept[t+len(self.intercept_parameters)-1] + \
                    self.AR[t+len(self.AR_parameters)-1]*self.y[t-1] + np.random.normal(loc=0, scale=sigma_y)
            elif self.time_varying_parameters == 'variance':
                self.y[t] = constant + deterministic_trend*(t-1) + self.errors[t+len(self.error_parameters)-1]
            elif self.time_varying_parameters == 'all':
                self.y[t] = constant + deterministic_trend*(t-1) + self.intercept[t+len(self.intercept_parameters)-1] + \
                    self.AR[t+len(self.AR_parameters)-1]*self.y[t-1] + self.errors[t+len(self.error_parameters)-1]
            else:
                raise ValueError(f'{self.time_varying_parameters} is not implemented yet.')

    def plot(self):
        """
        Plot the generated time series data.
        """
        if self.time_varying_parameters == 'intercept':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=ax1, lw=0.3)
            ax1.set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.intercept[len(self.intercept_parameters)-1:], label = '$μ_t$', ax=ax2, color='red', lw=0.3)
            ax2.set_xlabel('Time')
        elif self.time_varying_parameters == 'AR':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=ax1, lw=0.3)
            ax1.set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.AR[len(self.AR_parameters)-1:], label = '$β_t$', ax=ax2, color='red', lw=0.3)
            ax2.set_xlabel('Time')
        elif self.time_varying_parameters == 'AR_with_intercept':
            fig, axs = plt.subplots(3, 1, figsize=(10, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=axs[0], lw=0.3)
            axs[0].set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.intercept[len(self.intercept_parameters)-1:], label = '$μ_t$', ax=axs[1], color='red', lw=0.3)
            axs[1].set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.AR[len(self.AR_parameters)-1:], label = '$β_t$', ax=axs[2], color='green', lw=0.3)
            axs[2].set_xlabel('Time')
        elif self.time_varying_parameters == 'variance':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=ax1, lw=0.3)
            ax1.set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.sigma2[len(self.variance_parameters)-1:], label = '$σ^2_t$', ax=ax2, color='red', lw=0.3)
            ax2.set_xlabel('Time')
        elif self.time_varying_parameters == 'all':
            fig, axs = plt.subplots(4, 1, figsize=(20, 15))
            sns.lineplot(x = self.x, y = self.y, label = 'y', ax=axs[0], lw=0.3)
            axs[0].set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.intercept[len(self.intercept_parameters)-1:], label = '$μ_t$', ax=axs[1], color='red', lw=0.3)
            axs[1].set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.AR[len(self.AR_parameters)-1:], label = '$β_t$', ax=axs[2], color='green', lw=0.3)
            axs[2].set_xlabel('Time')
            sns.lineplot(x = self.x, y = self.sigma2[len(self.variance_parameters)-1:], label = '$σ^2_t$', ax=axs[3], color='blue', lw=0.3)
            axs[3].set_xlabel('Time')
        plt.suptitle(f'Simulated data with a State Space Model.')
        plt.tight_layout()
        
