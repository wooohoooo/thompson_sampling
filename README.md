# Project name here
> Summary description here.


```python
from thompson_sampling.helpers import plot_regret
```

```python
theta = [0.6, 0.3]

theta_contextual = [1.6, 0.4]


num_data = 1500
X = np.linspace(-6, 6, num_data)
X = shuffle(X)
noise = 0.1

```

This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

# Categorical Reward

This section deals with rewards of categorical nature, i.e. yes/no, 0/1, ...

Note that our approaches are only built to take two categories. 

### non-contextual case

The non-contextual case does.. well... not use context.

```python
from thompson_sampling.multi_armed_bandits import non_contextual_categorical_bandit
```

```python
from thompson_sampling.solvers import BetaBandit
```

```python

bb = BetaBandit()

y_optimal_list = []
y_hat_list= []
for i in range(250):
    if i in [1,5,10,75,250]:
        plt.figure()
        arm = bb.choose_arm_and_plot()
    else:
        arm = bb.choose_arm()
    reward = non_contextual_categorical_bandit(arm,theta)
    bb.update(arm,reward)
    y_hat_list.append(int(reward))
        
    y_optimal = non_contextual_categorical_bandit(np.argmax(theta),theta)
    y_optimal_list.append(y_optimal)

```


![png](docs/images/output_12_0.png)



![png](docs/images/output_12_1.png)



![png](docs/images/output_12_2.png)



![png](docs/images/output_12_3.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_13_0.png)


## contextual Case

```python
from thompson_sampling.multi_armed_bandits import contextual_categorical_bandit, contextual_categorical_get_optimal_arm
```

#### BONUS: contextual bandit with noncontextual solver

```python
bb = BetaBandit()

y_hat_list = []
y_optimal_list = []

for i in progress_bar(range(num_data)):
    context = X[i]
    if i  in [0,5,10, 100, num_data//2, num_data]:
        plt.figure()
        arm = bb.choose_arm_and_plot()
    else:
        arm = bb.choose_arm()
    y_hat = contextual_categorical_bandit(context,arm, theta_contextual,noise)[0]
    y_hat_list += [y_hat]
    bb.update(arm, y_hat)
    y_optimal = contextual_categorical_bandit(context, contextual_categorical_get_optimal_arm(context), theta_contextual, noise)[0]
    y_optimal_list.append(y_optimal)


plt.figure()
plt.scatter(X,np.array(y_hat_list),c=range(len(y_hat_list)))



```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:03<00:00]
</div>






    <matplotlib.collections.PathCollection at 0x7f0f52ba1c90>




![png](docs/images/output_17_2.png)



![png](docs/images/output_17_3.png)



![png](docs/images/output_17_4.png)



![png](docs/images/output_17_5.png)



![png](docs/images/output_17_6.png)



![png](docs/images/output_17_7.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_18_0.png)


### contextual solver

```python
from thompson_sampling.models import OnlineLogisticRegression
from thompson_sampling.solvers import LogisticThompsonSampler
```

```python
lts = LogisticThompsonSampler(OnlineLogisticRegression, num_arms=2, num_context = 1)
```

```python
y_hat_list = []
y_optimal_list = []
arms = []

for i in progress_bar(range(num_data)):
    context = X[i]
    
#     if i % (num_data/10) == 0:
#         plt.figure()
#         arm = lts.choose_arm_and_plot(context)
#     else:
    arm = lts.choose_arm(context)
        
        
    #arm = lts.choose_arm(context)
    arms.append(arm)

    y_hat = contextual_categorical_bandit(context,arm, theta_contextual,noise)[0]
    y_hat_list += [y_hat]

    lts.update(arm, context, y_hat)

    y_optimal = contextual_categorical_bandit(context, contextual_categorical_get_optimal_arm(context), theta_contextual, noise)[0]
    y_optimal_list.append(y_optimal)
plt.figure()
plt.scatter(X,np.array(y_hat_list),c=range(len(y_hat_list)))


```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1500' class='' max='1500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1500/1500 00:03<00:00]
</div>






    <matplotlib.collections.PathCollection at 0x7f0f5018ba50>




![png](docs/images/output_22_2.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_23_0.png)


# nbdev stuff

```python

from nbdev.export import *
notebook2script()
```

    Converted 00_abstractions.ipynb.
    Converted 01_multi_armed_bandits.ipynb.
    Converted 02_models.ipynb.
    Converted 03_ensembles.ipynb.
    Converted 04_solvers.ipynb.
    Converted 99_helpers.ipynb.
    Converted Noncontextual_bandits.ipynb.
    Converted contextual_bandits.ipynb.
    Converted index.ipynb.


```python
! nbdev_build_docs
```

    converting: /home/thomas/Documents/GitHub/thompson_sampling/contextual_bandits.ipynb
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.99434D-01    |proj g|=  7.92917D-02
    
    At iterate    1    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      1      3      1     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      0      1      0     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      0      1      0     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      0      1      0     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  2.06845D+00
    
    At iterate    1    f=  5.91858D-01    |proj g|=  1.08896D-02
    
    At iterate    2    f=  5.91855D-01    |proj g|=  5.88083D-05
    
    At iterate    3    f=  5.91855D-01    |proj g|=  2.47834D-09
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      3      5      1     0     0   2.478D-09   5.919D-01
      F =  0.59185508486806593     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.67717D-01    |proj g|=  2.55916D-01
    
    At iterate    1    f=  6.66147D-01    |proj g|=  8.65716D-06
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      1      3      1     0     0   8.657D-06   6.661D-01
      F =  0.66614706612568608     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      0      1      0     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  5.85321D-01    |proj g|=  9.18341D-01
    
    At iterate    1    f=  5.66241D-01    |proj g|=  1.70708D-03
    
    At iterate    2    f=  5.66241D-01    |proj g|=  4.97163D-07
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      2      4      1     0     0   4.972D-07   5.662D-01
      F =  0.56624092975747142     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.93147D-01    |proj g|=  0.00000D+00
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      0      1      0     0     0   0.000D+00   6.931D-01
      F =  0.69314718055994529     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    RUNNING THE L-BFGS-B CODE
    
               * * *
    
    Machine precision = 2.220D-16
     N =            1     M =           10
     This problem is unconstrained.
    
    At X0         0 variables are exactly at the bounds
    
    At iterate    0    f=  6.80233D-01    |proj g|=  8.44645D-02
    
    At iterate    1    f=  6.80071D-01    |proj g|=  3.35521D-08
    
               * * *
    
    Tit   = total number of iterations
    Tnf   = total number of function evaluations
    Tnint = total number of segments explored during Cauchy searches
    Skip  = number of BFGS updates skipped
    Nact  = number of active bounds at final generalized Cauchy point
    Projg = norm of the final projected gradient
    F     = final function value
    
               * * *
    
       N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
        1      1      3      1     0     0   3.355D-08   6.801D-01
      F =  0.68007071329349689     
    
    CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
    Warning: Some of your Notebooks use the same title (['abstractions', 'multi_armed_bandits', 'models', 'ensembles', 'solvers', 'helpers', 'Title', 'Title']).
    converting /home/thomas/Documents/GitHub/thompson_sampling/index.ipynb to README.md


```python
! nbdev_install_git_hooks
```

    Executing: git config --local include.path ../.gitconfig
    Success: hooks are installed and repo's .gitconfig is now trusted


```python
# nbdev_fix_merge filename.ipynb
```
