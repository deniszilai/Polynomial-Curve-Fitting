import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

def main():
    n = 3  # gradul polinomului P(x)
    data = pd.read_csv('points2.csv') # citire
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    xs, xy = get_system_of_equations(x, y, n)  # \sum_{k=0}^{n}a_{k} \sum_{i=1}^{m}x_{i}^{j+k} = \sum_{i=1}^{m}y_{i}x_{i}^{j}, for j = 0,1,...,n

    xs = np.reshape(xs, ((n + 1), (n + 1)))  # remodelare la matricea xs pentru a printa sistemul de ecuatii
    xy = np.reshape(xy, ((n + 1), 1))
    print('XS:\n', xs, '\n\n', 'XY:\n', xy)

    a = np.linalg.solve(xs, xy)  # rezolvarea sistemului de ecuatii
    print('\nSOLUTIONS:\n', a)  # afisarea solutiilor

    error = find_error(y, np.array(fn(x, a)))  # determinarea erorii P(x)
    print("\nError =", error)

    plot(x, y, fn(x, a))  # afisarea punctelor si aproximarea functiei P(x)


def get_system_of_equations(x, y, n):
    xs = np.array([]);
    xy = np.array([])  # xs = suma valorilor x, xy = produsul valorilor dintre x si y
    for index in range(0, (n + 1)):
        for exp in range(0, (n + 1)):
            tx = np.sum(x ** (index + exp))  # \sum_{i=1}^{m}x_{i}^{j+k}
            xs = np.append(xs, tx)
        txy = np.sum(y * (x ** index))  # \sum_{i=1}^{m}y_{i}x_{i}^{j}
        xy = np.append(xy, txy)
    return xs, xy

def fn(x, a):
    px = 0
    for index in range(0, np.size(a)):
        px += (a[index] * (x ** index))  # evaluare P(x)
    # print("\npx =", px)
    return px

def find_error(y, fn):
    return np.sum((y - fn) ** 2)  # E = \sum_{i=1}^{m} (y_{i} - P(x_{i}))**2


def plot(x, y, fn):
    pl.figure(figsize=(8, 6), dpi=80)
    pl.subplot(1, 1, 1)
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    pl.subplot(1, 1, 1)
    pl.plot(x, fn, color='red', linewidth=3.0, linestyle='-', label='P(x)')
    pl.legend(loc='best')
    pl.grid()
    pl.show()


if __name__ == '__main__':
    main()

def plot(x, y):
    pl.figure(figsize=(8, 6), dpi=80)
    pl.subplot(1, 1, 1)
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    pl.legend(loc='best')
    pl.grid()
    pl.show()
