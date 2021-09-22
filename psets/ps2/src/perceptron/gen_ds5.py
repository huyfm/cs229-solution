import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(229)

SIZE = 200

def distribution_one():
    angle = np.random.uniform(0, math.pi * 2, SIZE)
    distance = np.random.normal(loc=0, scale=1, size=SIZE)

    points = np.stack([distance * np.sin(angle), distance * np.cos(angle)], axis=1)

    return points

def distribution_two():
    angle = np.random.uniform(0, math.pi * 2, SIZE)
    distance = np.random.normal(loc=4, scale=1, size=SIZE)

    points = np.stack([distance * np.sin(angle), distance * np.cos(angle)], axis=1)

    return points

if __name__ == '__main__':
    x_one = distribution_one()
    x_two = distribution_two()

    xs = np.concatenate([x_one, x_two])
    print(xs.shape)
    ys = np.array([[i < x_one.shape[0]] for i in range(xs.shape[0])])
    print(ys.shape)

    D = np.concatenate([ys, xs], axis=1)
    print(D.shape)

    np.random.shuffle(D)

    num_x_values = D.shape[1] - 1
    header = ','.join(['y'] + ['x{}'.format(i) for i in range(num_x_values)])
    np.savetxt('../data/ds5_train.csv', D[:SIZE,:], delimiter=',', header=header, comments='')
    np.savetxt('../data/ds5_test.csv', D[SIZE:,:], delimiter=',', header=header, comments='')


    plt.figure(figsize=(12, 8))
    plt.scatter(x_one[:,0], x_one[:,1], color='red')
    plt.scatter(x_two[:,0], x_two[:,1], color='blue')
    plt.savefig('./scripts/plot.png')