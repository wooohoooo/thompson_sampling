# AUTOGENERATED! DO NOT EDIT! File to edit: 99_helpers.ipynb (unless otherwise specified).

__all__ = ['plot_regret', 'showcase_code', 'iters', 'l2', 'n_std', 'plot_online_logreg']

# Cell
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


#ToDo: Propagate them through the methods
iters = 10
l2 = 1
n_std = 4
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython


def plot_regret(y_optimal_list,y_hat_list):
    y_optimal_array = np.array(y_optimal_list)
    y_hat_array = np.array(y_hat_list)
    regret_list = []


    regret = np.cumsum(y_optimal_array - y_hat_array)

    plt.plot(regret)


def showcase_code(pyfile,class_name = False, method_name = False):
    """shows content of py file"""


    with open(pyfile) as f:
        code = f.read()

    if class_name:
        #1. find beginning (class + <name>)
        index = code.find(f'class {class_name}')
        code = code[index:]

        #2. find end (class (new class!) or end of script)
        end_index = code[7:].find('class')
        code = code[:end_index]

    if method_name:
        #1. find beginning (class + <name>)
        index = code.find(f'def {method_name}')
        code = code[index:]

        #2. find end (class (new class!) or end of script)
        end_index = code[7:].find('def')
        code = code[:end_index]


    formatter = HtmlFormatter()
    return IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter)))


# Cell
import scipy.stats as stats

def plot_online_logreg(online_lr, wee_x, wee_y):
    # closing other figures
    plt.close('all')
    plt.figure(figsize=[9,3.5], dpi=150)

    # let us check the distribution of weights and uncertainty bounds
    plt.figure(figsize=[9,3.5], dpi=150)

    # plotting the pdf of the weight distribution
    X_pdf = np.linspace(-4, 4, 1000)
    pdf = stats.norm(loc=online_lr.m, scale=online_lr.q**(-1.0)).pdf(X_pdf)

    # range and resolution of probability plot
    X_prob = np.linspace(-6, 6, 1000)
    p_dist = 1/(1 + np.exp(-X_prob * online_lr.m))
    p_dist_plus = 1/(1 + np.exp(-X_prob * (online_lr.m + 2*online_lr.q**(-1.0))))
    p_dist_minus = 1/(1 + np.exp(-X_prob * (online_lr.m - 2*online_lr.q**(-1.0))))

    # opening subplots
    ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=3, rowspan=1)

    # plotting distriution of weights
    ax1.plot(X_pdf, pdf, color='b', linewidth=2, alpha=0.5)
    #ax1.plot([cmab.weights[0][1], cmab.weights[0][1]], [0, max(pdf)], 'k--', label='True $\\beta$', linewidth=1)
    ax1.fill_between(X_pdf, pdf, 0, color='b', alpha=0.2)

    # plotting probabilities
    ax2.plot(X_prob, p_dist, color='b', linewidth=2, alpha=0.5)
    ax2.fill_between(X_prob, p_dist_plus, p_dist_minus, color='b', alpha=0.2)
    ax2.scatter(wee_x, wee_y, c='k')

    # title and comments
    ax1.set_title('OLR estimate for $\\beta$', fontsize=10)
    ax1.set_xlabel('$\\beta$', fontsize=10); ax1.set_ylabel('$density$', fontsize=10)
    ax2.set_title('OLR estimate for $\\theta(x)$', fontsize=10)
    ax2.set_xlabel('$x$', fontsize=10); ax2.set_ylabel('$\\theta(x)$', fontsize=10)

    ax1.legend(fontsize=10)
    plt.tight_layout()
    plt.show()