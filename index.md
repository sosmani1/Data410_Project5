# Advanced Machine Learning Project 5

### A Comparison of Different Regularization and Variable Selection Techniques
### In this project, we will apply and compare the different regularization techniques including Ridge, LASSO, Elastic Net, SCAD, and Square Root Lasso.

This project is interesting because it focuses more on the coding aspect rather than an explanatory aspect. Given the advanced nature of our courses progress I will assume any readers here will fully grasp the mechanics of Ridge, LASSO, Elastic Net, SCAD, and Square Root Lasso. If you do not please feel free to contact me with further questions. 


```

# Relevant installs and imports:

!pip install pyswarms
!pip install --upgrade statsmodels==0.13.2

import pyswarms as ps
from numba import jit, prange
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 150

# general imports
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer
from numba import njit
from sklearn.preprocessing import StandardScaler
```

## Now that all important basic fundamentals are established in our coding base. We will proceed to create an Sklearn compliant Square Root Lasso Function:

```

class SQRTLasso:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.array((-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)).flatten()
          return output
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))

        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
          setattr(self, parameter, value)
        return self
```
## Now we will create a Sklearn compliant SCAD function:
```
@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```

```
from sklearn.base import BaseEstimator, RegressorMixin
class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

## Now we will be Simulating 100 datasets, applying variable selection methods + GridSearchCV, and calculating final results:

```

def make_correlated_features(num_samples,p,rho):
  vcor = [] 
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  return X
  
MSE_Ridge = 0 
MSE_Lasso = 0 
MSE_Net = 0 
MSE_SQRT = 0
MSE_SCAD = 0
nonzero_Ridge = 0 
nonzero_Lasso = 0 
nonzero_Net = 0 
nonzero_SQRT = 0
nonzero_SCAD = 0
L2_Ridge = 0 
L2_Lasso = 0 
L2_Net = 0 
L2_SQRT = 0
L2_SCAD = 0
beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))

for x in range(100): 
  n = 200
  p = 1200
  X = make_correlated_features(200,p,0.8)
  np.corrcoef(np.transpose(X))
  beta =np.array([-1,2,3,0,0,0,0,2,-1,4])
  beta.shape
  beta = beta.reshape(-1,1)
  betas = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)
  n = 200
  sigma = 3.5
  y = X.dot(betas) + sigma*np.random.normal(0,1,n).reshape(-1,1)
  
  model = Ridge()
  ridge_params = {'alpha':[200, 230, 250,265, 270, 275, 290, 300, 500]}
  gs = GridSearchCV(estimator=model,cv=2,scoring='neg_mean_squared_error',param_grid=ridge_params)
  gs_results = gs.fit(X,y)
  coefs = gs_results.best_estimator_.coef_
  MSE_Ridge += np.abs(gs_results.best_score_)
  nonzero_Ridge += np.count_nonzero(coefs)
  L2_Ridge += np.linalg.norm(coefs-beta_star,ord=2)

  model = Lasso()
  lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}
  gs = GridSearchCV(estimator=model,cv=2,scoring='neg_mean_squared_error',param_grid=lasso_params)
  gs_results = gs.fit(X,y)
  coefs = gs_results.best_estimator_.coef_
  MSE_Lasso += np.abs(gs_results.best_score_)
  nonzero_Lasso += np.count_nonzero(coefs)
  L2_Lasso += np.linalg.norm(coefs-beta_star,ord=2)

  model = ElasticNet(max_iter=1000000)
  params = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}
  gs = GridSearchCV(estimator=model,cv=2,scoring='neg_mean_squared_error',param_grid=params)
  gs_results = gs.fit(X,y)
  coefs = gs_results.best_estimator_.coef_
  MSE_Net += np.abs(gs_results.best_score_)
  nonzero_Net += np.count_nonzero(coefs)
  L2_Net += np.linalg.norm(coefs-beta_star,ord=2)

  model = SQRTLasso()
  parametersGrid = {'alpha':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]}
  gs = GridSearchCV(estimator=model,cv=2,scoring='neg_mean_squared_error',param_grid=parametersGrid)
  gs_results = gs.fit(X,y)
  coefs = gs_results.best_estimator_.coef_
  MSE_SQRT += np.abs(gs_results.best_score_)
  nonzero_SQRT += np.count_nonzero(coefs)
  L2_SQRT += np.linalg.norm(coefs-beta_star,ord=2)

  model = SCAD()
  params = {'a':[0.02, 0.024, 0.025, 0.026, 0.03]}
  gs = GridSearchCV(estimator=model,cv=2,scoring='neg_mean_squared_error',param_grid=params)
  gs_results = gs.fit(X,y)
  coefs = gs_results.best_estimator_.coef_
  MSE_SCAD += np.abs(gs_results.best_score_)
  nonzero_SCAD += np.count_nonzero(coefs)
  L2_SCAD += np.linalg.norm(coefs-beta_star,ord=2)
  ```
  
## Now we will be Displaying the final results:
```
  from math import sqrt
print('Final Results:')
print('\n')
print('Ridge + Grid Search:')
print('Average RMSE: ' + str(sqrt(MSE_Ridge/100)))
print('Average # non-zero coefficients: ' + str(nonzero_Ridge/100))
print('Average L2 distance: ' + str(L2_Ridge/100))
print('\n')
print('Lasso + Grid Search:')
print('Average RMSE: ' + str(sqrt(MSE_Lasso/100)))
print('Average # non-zero coefficients: ' + str(nonzero_Lasso/100))
print('Average L2 distance: ' + str(L2_Lasso/100))
print('\n')
print('ElasicNet + Grid Search:')
print('Average RMSE: ' + str(sqrt(MSE_Net/100)))
print('Average # non-zero coefficients: ' + str(nonzero_Net/100))
print('Average L2 distance: ' + str(L2_Net/100))
print('\n')
print('SQRT Lasso + Grid Search:')
print('Average RMSE: ' + str(sqrt(MSE_SQRT/100)))
print('Average # non-zero coefficients: ' + str(nonzero_SQRT/100))
print('Average L2 distance: ' + str(L2_SQRT/100))
print('\n')
print('SCAD + Grid Search:')
print('Average RMSE: ' + str(sqrt(MSE_SCAD/100)))
print('Average # non-zero coefficients: ' + str(nonzero_SCAD/100))
print('Average L2 distance: ' + str(L2_SCAD/100))
```

## Final Results:


Ridge + Grid Search:
Average RMSE: 6.538135915575019
Average # non-zero coefficients: 1200.0
Average L2 distance: 3.820927108613206


Lasso + Grid Search:
Average RMSE: 4.641382822119719
Average # non-zero coefficients: 185.1
Average L2 distance: 6.129713935549328


ElasicNet + Grid Search:
Average RMSE: 4.451569094754115
Average # non-zero coefficients: 47.9
Average L2 distance: 3.7681904829435817


SQRT Lasso + Grid Search:
Average RMSE: 4.757838554779526
Average # non-zero coefficients: 1200.0
Average L2 distance: 4.749864375747746


SCAD + Grid Search:
Average RMSE: 7.563598753366877
Average # non-zero coefficients: 1200.0
Average L2 distance: 3.7968680651667626
