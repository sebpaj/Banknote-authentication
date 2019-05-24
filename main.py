import numpy as np
from data_manager import DataManager
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from matplotlib.colors import ListedColormap
from adaline import AdalineGD


def plot_decision_regions(X, y, clasiifier, resolution=0.02):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = clasiifier.predict(np.array([xx1.ravel(), xx1.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


def main():
    data = DataManager.load_data("data/data_banknote_authentication.txt")
    # remove break lines
    data = np.array([item.strip("\n") for item in data])
    data = np.array([item.split(',') for item in data])
    data = data.astype(np.float)

    # 0 for authentic and 1 for inauthentic
    df = pd.DataFrame(data)
    df[4] = df[4].astype(int)
    authentic = df[df[4] == 0]
    inauthentic = df[df[4] == 1]
    X = df.iloc[np.r_[0:200, 1100:1300], [0, 3]].values
    y = [0 if x < 200 else 1 for x in range(400)]
    plt.scatter(X[:200, 0], X[:200, 1], color='red', marker='o', label='authentic')
    plt.scatter(X[200:400, 0], X[200:400, 1], color='blue', marker='x', label='inauthentic')
    plt.xlabel('variance of Wavelet Transformed image')
    plt.ylabel('entropy of image')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plot_decision_regions(X, y, clasiifier=ppn)
    plt.xlabel('variance of Wavelet Transformed image')
    plt.ylabel('entropy of image')
    plt.legend(loc='upper left')
    plt.show()

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineGD(n_iter=10, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, clasiifier=ada)
    plt.title('Adaline - gradient descent')
    plt.xlabel('variance of Wavelet Transformed image')
    plt.ylabel('entropy of image')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()

