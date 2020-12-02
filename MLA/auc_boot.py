import numpy as np

def bootstrapped_auc(y, prob, size=1000, ax=None, color='blue'):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.utils import resample
    if color is None:
        color = 'tab:blue'
    aucs = np.empty(1000, dtype=float)
    for i in range(size):
        sample = resample(y, prob)
        aucs[i] = roc_auc_score(sample[0], sample[1])
        if ax is not None:
            fpr, tpr, _ = roc_curve(sample[0], sample[1])
            ax.plot(fpr, tpr, alpha=1/(size/10), color=color)
    CI = np.quantile(aucs, [0.025, 0.975])

    return CI[0], roc_auc_score(y, prob), CI[1]

if __name__ == '__main__':

    # define a simple classification problem
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    from sklearn.linear_model import SGDClassifier 
    from sklearn.model_selection import train_test_split

    x = data['data']
    y = data['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    clf = SGDClassifier()

    clf.fit(x_train, y_train)

    # compute ROC-AUC


    prob_train = clf.decision_function(x_train)
    prob_test = clf.decision_function(x_test)


    print('train', bootstrapped_auc(y_train, prob_train))
    print('test', bootstrapped_auc(y_test, prob_test))

