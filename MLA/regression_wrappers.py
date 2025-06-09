import numpy as np


def logistic(x, y, color=None, ax=None, title=None, reg=None):
    import statsmodels.api as sm
    model = sm.Logit(y, sm.add_constant(x))
    if reg is None:
        re = model.fit()
    else:
        re = model.fit_regularized(alpha=0.01)
    if ax != None:
        if color==None:
            color='grey'
        x = np.linspace(min(x), max(x), 10)
        xx =  sm.add_constant(x)
        y = re.predict(xx)
        ax.plot(x, y, color=color)
        if title!=None:
            ax.set_title(title)
        # estimate confidence interval for predicted probabilities
        cov = re.cov_params()
        gradient = (y * (1 - y) * xx.T).T # matrix of gradients for each observation
        std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
        c = 1.96 # multiplier for confidence interval
        upper = np.maximum(0, np.minimum(1, y + std_errors * c))
        lower = np.maximum(0, np.minimum(1, y - std_errors * c))
        
        # ax.text(0.05, 0.9, "P=%.e"%re.pvalues[0], transform=ax.transAxes, fontsize=fontsize-1)
        # print(re.pvalues)
        ax.plot(x, upper, '--', color=color)
        ax.plot(x, lower, '--', color=color)
    # ax.plot(xx, re.predict(xx), color='red')
        return re.params 
#         return re.summary2().tables[1]


def linear(x, y, color=None, ax=None, label=None, log=False, linewidth=1, alpha=1):
    import statsmodels.api as sm
    model = sm.OLS(y.values, sm.add_constant(x.values) )
    re = model.fit()
    if ax != None:
        if color==None:
            color='grey'
        x = np.linspace(min(x), max(x), 10)
        xx =  sm.add_constant(x)
        # print(re.get_prediction(xx).summary_frame().head())
        ci = re.get_prediction(xx).summary_frame()[["mean", "mean_ci_lower", "mean_ci_upper"]].values
        if log:
            ci = np.exp(ci)
        ax.plot(x, ci[:,0], color=color, label=label, linewidth=linewidth, alpha=alpha)
        # ax.text(0.05, 0.9, "P=%.e"%re.pvalues[0], transform=ax.transAxes, fontsize=fontsize-1)
        # print(re.pvalues)
        ax.plot(x, ci[:,1], '--', color=color, linewidth=linewidth * 0.85, alpha=alpha)
        ax.plot(x, ci[:,2], '--', color=color, linewidth=linewidth * 0.85, alpha=alpha)
        ax.legend()
    # ax.plot(xx, re.predict(xx), color='red')
    return re.summary2().tables[1]


def plot_points(ax, x, y, color, ylim=(0, 96), label=None, alpha=1, s=4):
#     marker = y.apply(lambda y: 'triangle_up' if (y < ylim[0]) & (y > ylim[1]) else '.').astype(str).values
    idx1 = y.apply(lambda y: True if y > ylim[1] else False)
    idx2 = y.apply(lambda y: True if y < ylim[0] else False)
    y = y.apply(lambda y: y if y < ylim[1] else ylim[1])
    y = y.apply(lambda y: y if y > ylim[0] else ylim[0])
    ax.scatter(x[idx1], y[idx1], color=color, s=s * 1.1, alpha=alpha, label=None, marker='^')
    ax.scatter(x[idx2], y[idx2], color=color, s=s * 1.1, alpha=alpha, label=None, marker='v')
    ax.scatter(x[~(idx1 & idx2)], y[~(idx1 & idx2)], color=color, s=s, alpha=alpha, label=None)    
