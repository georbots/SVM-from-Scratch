import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.svm import SVC
from svm_module import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


def plot_svm_rbf(X, y, model, title, b, sv, sigma=1):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', s=50)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    xy = np.column_stack((xx.ravel(), yy.ravel()))

    # Evaluate decision function on grid
    if model.kernel == 'linear':
        decision_function = np.dot(xy, model.coef_.flatten()) + model.intercept_
    elif model.kernel == 'rbf':
        decision_function = np.sum(model.gaussian_kernel(sv, xy, sigma=sigma) * model.alphas * model.sv_y, axis=0) + b
    else:
        raise ValueError("Invalid kernel specified.")

    Z = decision_function.reshape(xx.shape)

    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none', edgecolors='k', marker='o')

    plt.title(f'{title} Set - SVM {model.kernel.capitalize()} Kernel')
    plt.show()


def plot_svm(X, y, model, sample_set):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()

    w = model.coef_
    b = model.intercept_
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy_decision = - (w[0] * xx + b) / w[1]
    yy_upper = - (w[0] * xx + b - 1) / w[1]
    yy_lower = - (w[0] * xx + b + 1) / w[1]

    plt.plot(xx, yy_decision, 'k-', label='Decision Boundary')
    plt.plot(xx, yy_upper, 'k--', label='Upper Margin')
    plt.plot(xx, yy_lower, 'k--', label='Lower Margin')

    # Plot support vectors
    support_vectors = model.support_vectors
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary and Margins on {sample_set}')
    plt.legend()
    plt.show()


def create_dataset(mode):
    if mode == 'lin_sep':
        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, flip_y=0, random_state=3, class_sep=1)
    elif mode == 'non_lin_sep':
        X, y = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=3)
    else:
        raise KeyError('Wrong type of dataset mode')
    y = np.expand_dims(y, axis=1).astype(float)
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    plot_dataset(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


def plot_dataset(X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', edgecolors='k',
                    label='Training Data')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('Training Data')

    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, marker='o', edgecolors='k',
                    label='Test Data')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title('Test Data')
    plt.tight_layout()
    plt.show()


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name='Custom'):
    start_time = time.time()
    if model_name == 'Custom':
        model.train(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = -(start_time - end_time) * 1000

    if model_name == 'Custom' and model.kernel == 'rbf':
        plot_svm_rbf(X_train, y_train, model, 'Train', model.intercept_, model.support_vectors)
        plot_svm_rbf(X_test, y_test, model, 'Test', model.intercept_, model.support_vectors)
    elif model_name == 'Custom' and model.kernel == 'linear':
        plot_svm(X_train, y_train, model, 'Train')
        plot_svm(X_test, y_test, model, 'Test')

    prediction = model.predict(X_test)

    y_test = y_test.flatten().astype(int)
    test_accuracy = accuracy_score(prediction, y_test)
    print(f'{model_name} SVM - Train time (ms): {elapsed_time}')
    if model.kernel == 'linear':
        print(f'W: {model.coef_}, b: {model.intercept_}')

    prediction = model.predict(X_train)

    y_train = y_train.flatten().astype(int)
    train_accuracy = np.mean(prediction == y_train)
    print(f'Accuracy on test set: {test_accuracy:.2%}')
    print(f'Accuracy on train set: {train_accuracy:.2%}')

    return prediction


X_train, X_test, y_train, y_test = create_dataset('lin_sep')


# Custom model evaluation linear case
model = SVM(kernel='linear', C=1000)
train_and_evaluate(model, X_train, y_train, X_test, y_test)

# Scikit-learn model evaluation linear case
clf_sklearn = SVC(kernel='linear', C=1000, gamma=3, degree=2)
train_and_evaluate(clf_sklearn, X_train, y_train.flatten(), X_test, y_test, model_name='Sklearn')


X_train, X_test, y_train, y_test = create_dataset('non_lin_sep')

# Custom model evaluation non linear case
model = SVM(kernel='rbf', C=20, gamma=1)
train_and_evaluate(model, X_train, y_train, X_test, y_test)

# Scikit-learn model evaluation non linear case
clf_sklearn = SVC(kernel='rbf', C=0.1, gamma=1, degree=2, tol=1e-6)
train_and_evaluate(clf_sklearn, X_train, y_train.flatten(), X_test, y_test, model_name='Sklearn')


