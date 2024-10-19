import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from dataclasses import dataclass, field

class EWMA:
    def __init__(self, lam):
        '''
        Initialize the EWMA model with a given lambda parameter.

        Parameters:
        lam (float): The smoothing parameter (lambda), between 0 and 1.
        '''
        self.lam = lam
        self.EWMA_volatility = None

    def fit(self, sq_rets):
        '''
        Fits the EWMA model to a series of squared returns.

        Parameters:
        sq_rets (pd.Series): A pandas Series of squared returns.

        Returns:
        pd.Series: A pandas Series of the annualized EWMA volatility with the same index as sq_rets.
        '''
        sq_ret = sq_rets.values
        EWMA_var = np.zeros(len(sq_ret))
        EWMA_var[0] = sq_ret[0]  # set initial variance based on the first squared return

        for r in range(1, len(sq_ret)):
            EWMA_var[r] = (1 - self.lam) * sq_ret[r] + self.lam * EWMA_var[r-1]  # compute EWMA variance

        EWMA_vol = np.sqrt(EWMA_var * 365)
        self.EWMA_volatility = pd.Series(EWMA_vol, index=sq_rets.index, name=f'EWMA Vol {self.lam}')
        return self.EWMA_volatility

    @staticmethod
    def compute_log_likelihood(residuals):
        '''
        Computes the log-likelihood of a set of residuals assuming they are normally distributed.

        Parameters:
        residuals (np.ndarray or pd.Series): An array or Series of residuals.

        Returns:
        float: The log-likelihood of the residuals.
        '''
        N = len(residuals)  # number of residuals
        sigma2 = np.var(residuals)  # variance of the residuals

        # the log-likelihood formula for normally distributed residuals
        log_likelihood = -0.5 * N * np.log(2 * np.pi) - 0.5 * N * np.log(sigma2) - (0.5 / sigma2) * np.sum(residuals ** 2)

        return log_likelihood

    @staticmethod
    def calculate_aic_bic(log_likelihood, N, k=1):
        '''
        Calculates the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC).

        Parameters:
        log_likelihood (float): The log-likelihood value.
        N (int): The number of observations.
        k (int, optional): The number of parameters estimated (default is 1).

        Returns:
        tuple: AIC and BIC values.
        '''
        # AIC formula: 2*k - 2*log_likelihood
        AIC = 2 * k - 2 * log_likelihood

        # BIC formula: k*log(N) - 2*log_likelihood
        BIC = k * np.log(N) - 2 * log_likelihood

        return AIC, BIC


class BEKK:
    def __init__(self, returns):
        self.returns = returns
        self.T = len(returns)
        self.m = 1  # Number of series (only one series 'returns' in this case)
        self.H0 = np.cov(returns, rowvar=False)  # Initial covariance matrix (sample covariance)

        # Initialize parameters (random initialization for illustration)
        self.Omega = np.eye(self.m)  # Constant covariance matrix
        self.A = np.random.normal(size=(self.m, self.m))  # Matrix A
        self.B = np.random.normal(size=(self.m, self.m))  # Matrix B

        # Number of parameters to estimate
        self.n_params = self.m**2 + 2 * self.m**2  # Omega + A + B parameters

        # Initial guess for parameters (flatten into a vector)
        self.initial_guess = np.concatenate([self.Omega.flatten(), self.A.flatten(), self.B.flatten()])

    def _likelihood(self, parameters):
        # Unpack parameters
        omega_params = parameters[:self.m**2]
        A_params = parameters[self.m**2:self.m**2 + self.m**2]
        B_params = parameters[self.m**2 + self.m**2:]

        # Reshape parameters to matrices
        Omega = np.reshape(omega_params, (self.m, self.m))
        A = np.reshape(A_params, (self.m, self.m))
        B = np.reshape(B_params, (self.m, self.m))

        # Initialize log-likelihood
        log_likelihood = 0

        # Iterate through time periods to compute likelihood
        Ht = self.H0.copy()  # Initialize Ht as H0
        for t in range(self.T):
            # Calculate current conditional variance
            Ht = Omega + np.outer(A, A) + np.dot(np.dot(B, Ht), B.T)

            # Calculate log likelihood contribution for this time period
            log_likelihood += -0.5 * (np.log(np.linalg.det(Ht)) + np.dot(np.dot(self.returns[t], np.linalg.inv(Ht)), self.returns[t]))

        return -log_likelihood  # Return negative log likelihood for minimization

    def fit(self):
        # Minimize negative log-likelihood
        res = minimize(self._likelihood, self.initial_guess, method='BFGS')

        # Extract estimated parameters
        estimated_params = res.x

        # Reshape estimated parameters
        self.Omega_est = np.reshape(estimated_params[:self.m**2], (self.m, self.m))
        self.A_est = np.reshape(estimated_params[self.m**2:self.m**2 + self.m**2], (self.m, self.m))
        self.B_est = np.reshape(estimated_params[self.m**2 + self.m**2:], (self.m, self.m))

        # Compute log-likelihood with estimated parameters
        self.log_likelihood_value = -res.fun  # Minimization function value is negative log-likelihood

    def simulate_volatility(self):
        # Simulate volatility using the estimated parameters
        Ht = self.H0.copy()
        volatility = np.zeros(self.T)

        for t in range(self.T):
            Ht = self.Omega_est + np.outer(self.A_est, self.A_est) + np.dot(np.dot(self.B_est, Ht), self.B_est.T)
            volatility[t] = np.sqrt(Ht)

        return volatility

    def calculate_var(self, alpha=0.05):
        # Calculate Value at Risk (VaR) at a given alpha level
        volatility = self.simulate_volatility()
        VaR = - np.percentile(self.returns, alpha * 100)
        return VaR

    def print_results(self):
        # Print estimated parameters
        print("Estimated Omega:")
        print(self.Omega_est)
        print("Estimated A:")
        print(self.A_est)
        print("Estimated B:")
        print(self.B_est)

        # Print log-likelihood value
        print(f"Log-Likelihood: {self.log_likelihood_value}")

        # Compute AIC and BIC
        k = self.n_params  # Number of parameters
        aic = -2 * self.log_likelihood_value + 2 * k
        bic = -2 * self.log_likelihood_value + k * np.log(self.T)

        print(f"AIC: {aic}")
        print(f"BIC: {bic}")

class DCC:
    def __init__(self, returns):
        self.returns = returns
        self.T = len(returns)
        self.m = 1  # Number of series (assuming one series for simplicity)
        self.Ht = np.cov(returns, rowvar=False)  # Initial covariance matrix (sample covariance)
        self.Q = np.eye(self.m)  # Q matrix (identity for simplicity)

    def likelihood(self, params):
        # Unpack parameters
        omega = params[0]
        alpha = params[1]
        beta = params[2]

        # Initialize log-likelihood
        log_likelihood = 0

        # Initialize DCC parameters
        Rt = np.zeros((self.m, self.m))
        Qt = np.zeros((self.m, self.m))

        for t in range(self.T):
            # Update Qt
            Qt = self.Q * (1 - alpha - beta) + alpha * np.outer(self.returns[t], self.returns[t]) + beta * self.Ht

            # Update Rt
            Rt = np.diag(np.diag(Qt)) ** (-0.5) @ Qt @ np.diag(np.diag(Qt)) ** (-0.5)

            # Calculate log-likelihood contribution for this time period
            log_likelihood += -0.5 * (np.log(np.linalg.det(Qt)) + np.dot(np.dot(self.returns[t], np.linalg.inv(Qt)), self.returns[t]))

            # Update Ht for the next iteration
            self.Ht = Qt

        return -log_likelihood  # Return negative log-likelihood for minimization

    def fit(self):
        # Initial guess for parameters (omega, alpha, beta)
        initial_guess = [0.1, 0.1, 0.8]

        # Minimize negative log-likelihood
        res = minimize(self.likelihood, initial_guess, method='BFGS')

        # Extract estimated parameters
        estimated_params = res.x
        omega_est = estimated_params[0]
        alpha_est = estimated_params[1]
        beta_est = estimated_params[2]

        return omega_est, alpha_est, beta_est, res.fun

    def calculate_var(self, alpha=0.05):
        # Compute conditional volatility using estimated DCC parameters
        conditional_volatility = np.sqrt(np.diag(self.Ht))

        # Calculate VaR using normal distribution quantile
        VaR = - norm.ppf(alpha) * conditional_volatility

        return VaR

class ADCC:
    def __init__(self, returns):
        self.returns = returns
        self.T = len(returns)
        self.m = 1  # Number of series (assuming one series for simplicity)
        self.Ht = np.cov(returns, rowvar=False)  # Initial covariance matrix (sample covariance)
        self.Q = np.eye(self.m)  # Q matrix (identity for simplicity)

    def likelihood(self, params):
        # Unpack parameters
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        gamma = params[3]

        # Initialize log-likelihood
        log_likelihood = 0

        # Initialize ADCC parameters
        Rt = np.zeros((self.m, self.m))
        Qt = np.zeros((self.m, self.m))

        for t in range(self.T):
            # Update Qt
            Qt = self.Q * (1 - alpha - beta - gamma) + alpha * np.outer(self.returns[t], self.returns[t]) + beta * self.Ht

            # Update Rt
            Rt = np.diag(np.diag(Qt)) ** (-0.5) @ Qt @ np.diag(np.diag(Qt)) ** (-0.5)

            # Calculate log-likelihood contribution for this time period
            log_likelihood += -0.5 * (np.log(np.linalg.det(Qt)) + np.dot(np.dot(self.returns[t], np.linalg.inv(Qt)), self.returns[t]))

            # Update Ht for the next iteration
            self.Ht = Qt

        return -log_likelihood  # Return negative log-likelihood for minimization

    def fit(self):
        # Initial guess for parameters (omega, alpha, beta, gamma)
        initial_guess = [0.1, 0.1, 0.8, 0.1]

        # Minimize negative log-likelihood
        res = minimize(self.likelihood, initial_guess, method='BFGS')

        # Extract estimated parameters
        estimated_params = res.x
        omega_est = estimated_params[0]
        alpha_est = estimated_params[1]
        beta_est = estimated_params[2]
        gamma_est = estimated_params[3]

        return omega_est, alpha_est, beta_est, gamma_est, res.fun

    def calculate_var(self, alpha=0.05):
        # Compute conditional volatility using estimated ADCC parameters
        conditional_volatility = np.sqrt(np.diag(self.Ht))

        # Calculate VaR using normal distribution quantile
        VaR = -norm.ppf(alpha) * conditional_volatility

        return VaR

class cDCC:
    def __init__(self, returns):
        self.returns = returns
        self.T = len(returns)
        self.m = 1  # Number of series (assuming one series for simplicity)
        self.H0 = np.cov(returns, rowvar=False)  # Initial covariance matrix (sample covariance)
        self.Omega = np.eye(self.m)  # Constant correlation matrix
        self.A = np.random.normal(size=(self.m, self.m))  # Matrix A for conditional volatility
        self.B = np.random.normal(size=(self.m, self.m))  # Matrix B for conditional correlation

    def likelihood(self, params):
        # Unpack parameters
        omega_params = params[:self.m**2]
        A_params = params[self.m**2:self.m**2 + self.m**2]
        B_params = params[self.m**2 + self.m**2:]

        # Reshape parameters to matrices
        Omega = np.reshape(omega_params, (self.m, self.m))
        A = np.reshape(A_params, (self.m, self.m))
        B = np.reshape(B_params, (self.m, self.m))

        # Initialize log-likelihood
        log_likelihood = 0

        # Iterate through time periods to compute likelihood
        Ht = self.H0.copy()  # Initialize Ht as H0
        for t in range(self.T):
            # Calculate current conditional variance
            Ht = Omega + np.outer(A, A) + np.dot(np.dot(B, Ht), B.T)

            # Calculate log likelihood contribution for this time period
            try:
                log_likelihood += -0.5 * (np.log(np.linalg.det(Ht)) + np.dot(np.dot(self.returns[t], np.linalg.inv(Ht)), self.returns[t]))
            except Exception as e:
                print(f"Exception encountered at time {t}: {e}")

        return -log_likelihood  # Return negative log likelihood for minimization

    def fit(self):
        # Initial guess for parameters (Omega, A, B)
        initial_guess = np.concatenate([self.Omega.flatten(), self.A.flatten(), self.B.flatten()])

        # Minimize negative log-likelihood
        res = minimize(self.likelihood, initial_guess, method='BFGS')

        # Extract estimated parameters
        estimated_params = res.x
        self.Omega_est = np.reshape(estimated_params[:self.m**2], (self.m, self.m))
        self.A_est = np.reshape(estimated_params[self.m**2:self.m**2 + self.m**2], (self.m, self.m))
        self.B_est = np.reshape(estimated_params[self.m**2 + self.m**2:], (self.m, self.m))

        return self.Omega_est, self.A_est, self.B_est, res.fun
    
    def calculate_var(self, alpha=0.05):
        # Simulate conditional volatility using the estimated parameters
        Ht = self.H0.copy()
        volatility = np.zeros(self.T)

        for t in range(self.T):
            Ht = self.Omega_est + np.outer(self.A_est, self.A_est) + np.dot(np.dot(self.B_est, Ht), self.B_est.T)
            volatility[t] = np.sqrt(np.diag(Ht))

        # Calculate VaR using normal distribution quantile
        VaR = -norm.ppf(alpha) * volatility

        return VaR

