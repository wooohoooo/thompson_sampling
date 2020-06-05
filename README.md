# thompson_sampling
> an educational ressource for Multi Armed Bandit Solutions with relative probability sampling.


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


np.random.seed(42)
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
from thompson_sampling.solvers import BetaBandit, GaussianCategoricalBandit
```

```python
%%time 
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

    CPU times: user 1.38 s, sys: 19.4 ms, total: 1.4 s
    Wall time: 1.47 s



![png](docs/images/output_12_1.png)



![png](docs/images/output_12_2.png)



![png](docs/images/output_12_3.png)



![png](docs/images/output_12_4.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_13_0.png)


GaussianCategorical comparisson

```python
%%time

gcb = GaussianCategoricalBandit()

y_optimal_list = []
y_hat_list= []
for i in range(250):
    if i in [1,5,10,75,250]:
        plt.figure()
        arm = gcb.choose_arm()
        gcb.plot_params()
    else:
        arm = gcb.choose_arm()
    reward = non_contextual_categorical_bandit(arm,theta)
    gcb.update(arm,reward)
    y_hat_list.append(int(reward))
        
    y_optimal = non_contextual_categorical_bandit(np.argmax(theta),theta)
    y_optimal_list.append(y_optimal)

```

    CPU times: user 57.1 s, sys: 177 ms, total: 57.3 s
    Wall time: 57.2 s



![png](docs/images/output_15_1.png)



![png](docs/images/output_15_2.png)



![png](docs/images/output_15_3.png)



![png](docs/images/output_15_4.png)


```python
gcb.plot_params()
```


![png](docs/images/output_16_0.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_17_0.png)


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
  100.00% [1500/1500 00:04<00:00]
</div>






    <matplotlib.collections.PathCollection at 0x7f3f6805fa50>




![png](docs/images/output_21_2.png)



![png](docs/images/output_21_3.png)



![png](docs/images/output_21_4.png)



![png](docs/images/output_21_5.png)



![png](docs/images/output_21_6.png)



![png](docs/images/output_21_7.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_22_0.png)


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
    

    arm = lts.choose_arm(context)
        
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
  100.00% [1500/1500 00:04<00:00]
</div>






    <matplotlib.collections.PathCollection at 0x7f3f69b60590>




![png](docs/images/output_26_2.png)


```python
plot_regret(y_optimal_list, y_hat_list)
```


![png](docs/images/output_27_0.png)


# nbdev stuff

```python
# ! nbdev_build_lib
```

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
    Converted contextual_bandits.ipynb.
    Converted index.ipynb.
    Converted noncontextual_bandits.ipynb.
    Converted nonlinear_bandits.ipynb.


```python
#  ! nbdev_build_docs
```

```python
# ! nbdev_install_git_hooks
```

```python
# def parse_requirements(filename):
#     """ load requirements from a pip requirements file """
#     lineiter = (line.strip() for line in open(filename))
#     return [line for line in lineiter if line and not line.startswith("#")]
# # parse_requirements() returns generator of pip.req.InstallRequirement objects
# install_reqs = parse_requirements('requirements.txt')

# install_reqs
```

!
