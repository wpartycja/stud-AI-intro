import matplotlib.pyplot as plt
import numpy as np
import argparse


def create_parser():
    """
    creates parser with two optional arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--start_x", type=float,
                        help='x coordinate of the starting point', default=0)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help='learning rate', default=0.2)
    return parser


def gradient_descent(func, gradient, start_x,
                     learning_rate, tolerance, max_iter):
    """
    finds the local minimum of one-variable function
    returns x coordinate of local minimum
    and list of searched points that show the path to the result
    """
    min = start_x
    ax, ay = [], []

    for _ in range(max_iter):
        ax.append(min)
        ay.append(func(min))
        diff = -learning_rate*gradient(min)
        if abs(diff) <= tolerance:
            break
        min += diff

    return min, ax, ay


if __name__ == "__main__":

    # parsing arguments
    parser = create_parser()
    args = parser.parse_args()
    start_x = args.start_x
    learning_rate = args.learning_rate

    # declaring functions
    func = lambda x: x**2 + 3*x + 8
    func_der = lambda x: 2*x + 3

    # calculating local minimum and the searched
    min, ax, ay = gradient_descent(func, func_der, start_x,
                                   learning_rate, 1e-6, 1000)
    print("calculated local minimum is: " +
          '({}, {})'.format(round(min, 2), round(func(min), 2)))

    # declaring plot paramteres (size od the plot, range of the axes)
    middle = (start_x + min)/2
    min_x = middle - 5*(abs(middle-min))
    max_x = middle + 5*(abs(middle-min))

    x = np.linspace(min_x, max_x, 100)
    y = func(x)

    # drawing the plots
    plt.plot(x, y, 'red')
    plt.plot(ax, ay, 'p', linestyle='solid')
    plt.plot(min, func(min), 'black')
    plt.grid()
    plt.annotate("Local min = " +
                 '({}, {})'.format(round(min, 2), round(func(min), 2)),
                 (min, func(min)))
    plt.title("Gradient descent")
    plt.show()


# Some nice functions:
# f(x) = x**2 + 3*x + 8
# f'(x) = 2*x + 3
#
# f(x) = x**4 - 5*x**2 - 3*x
# f'(x) = 4*x**3 - 10*x -3
