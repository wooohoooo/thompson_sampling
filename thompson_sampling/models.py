# AUTOGENERATED! DO NOT EDIT! File to edit: 02_models.ipynb (unless otherwise specified).

__all__ = ['OnlineLogisticRegression', 'BayesLinReg', 'SimpleModel']

# Cell
from .multi_armed_bandits import contextual_categorical_bandit, contextual_categorical_get_optimal_arm, contextual_numerical_bandit
import matplotlib.pyplot as plt
import numpy as np



# Cell
from scipy.optimize import minimize
import scipy.stats as stats
import numpy as np



class OnlineLogisticRegression:
    """taken from https://gdmarmerola.github.io/ts-for-contextual-bandits/ """

    # initializing
    def __init__(self,n_dim, lambda_ = None, alpha = None):

        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_ or 1
        self.alpha = alpha or 1

        # initializing parameters of the model
        self.n_dim = n_dim,
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_

        # initializing weights
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    # method for sampling weights
    def get_weights(self):
      return stats.multivariate_normal(self.m, self.alpha * (self.q)**(-1.0)).rvs()
        #return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    # fitting method
    def fit(self, X, y):

#         print(X)

#         print(f'X {X.shape}')
#         print(f'y {y.shape}')
#         print(f'self w {self.w.shape}')
#         print(f'self m {self.m.shape}')

        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
        self.m = self.w

        # step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)


    def observe(self,X,y):
        self.fit(X,y)

    # probability output method, using weights sample
    def predict_proba(self, X, mode='sample'):

        # adding intercept to X
        #X = add_constant(X)

        # sampling weights after update
        self.w = self.get_weights()

        # using weight depending on mode
        if mode == 'sample':
            w = self.w # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')


        X = np.atleast_1d(X)
        w = np.atleast_1d(w)
        #print(f'X shape {X.shape}')
        #print(f'w shape {w.shape}')

        # calculating probabilities
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return np.array([1-proba , proba]).T

# Cell

"this is not working at the moment"
class BayesLinReg(object):

  def __init__(self, num_features,v):
    self.intercept = False
    if self.intercept:
      num_features += 1

    self.B = np.eye(num_features)
    self.Binv = np.linalg.inv(self.B)
    self.f = np.atleast_1d(np.zeros(num_features))
    self.v = v

    self.mu = np.zeros(num_features)

  def add_intercept(self,X):
    if self.intercept:
      X = np.insert(np.atleast_1d(X),0,1)
    X = np.atleast_1d(X)

    return X.T


  def observe(self,X,y):
    y = np.atleast_2d(y)
    X = self.add_intercept(X)
    self.B += np.outer(X,X)
    self.f += np.dot(X,y).T


  def get_mean_std(self):
      B_inv = np.linalg.inv(self.B)
      mu_t = B_inv.dot(self.f.T)

      return mu_t, B_inv



#helpers
  def train(self,X,y,shuffle = True):
      index = list(range(X.shape[0]))
      if shuffle ==True:
          np.random.shuffle(index)
      for i in index:
          self.observe(X[i],y[i])



  def predict_ML(self,X):
      beta= np.linalg.inv(self.B).dot(self.f.T)

      y = []
      try:
          for i in range(len(X)):
              x = X[i]
              y += [self.predict_ML_x(x)]
              return y
      except:
          X = self.add_intercept(X)
          return X.T.dot(beta)





  def predict_ML_x(self,x):
      beta = np.linalg.inv(self.B).dot(self.f.T)
      x = self.add_intercept(x)
      return x.T.dot(beta)[0][0]

  def draw(self):
      B_inv = np.linalg.inv(self.B)
      mu_t = B_inv.dot(self.f.T)
      dist = stats.multivariate_normal
      return dist.rvs(mean=mu_t.flatten(),cov=self.v**2*B_inv)


# Cell
import torch
from torch.autograd import Variable

class SimpleModel(torch.nn.Module):
    def __init__(self,num_input, num_hidden_units=100, p=0.05, decay=0.001, non_linearity=torch.nn.LeakyReLU):
        super(SimpleModel, self).__init__()
        self.dropout_p = p
        self.decay = decay
        self.f = torch.nn.Sequential(
            torch.nn.Linear(num_input,num_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_units,1)
        )
    def forward(self, X):
        X = Variable(torch.Tensor(X), requires_grad=False)
        return self.f(X)