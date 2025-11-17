import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


LINSPACE_POINTS = 50 

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def nelder_mead_trajectory(x0):
    trajectory = []
    def callback(xk):
        trajectory.append(xk.copy())

    opt.minimize(lambda x: rosenbrock(x[0], x[1]), x0,
                 method='Nelder-Mead', callback=callback)

    return np.array(trajectory)

def plot_nelder_mead(X, Y, f):
    plt.figure(figsize=(7,6))
    plt.contour(X, Y, f, levels=50, cmap='viridis')
    for idx in range(3):
        x0 = np.random.uniform(-2, 2, size=2)
        path = nelder_mead_trajectory(x0)
        plt.scatter(x0[0], x0[1], label="Starting point")
        plt.plot(path[:,0], path[:,1], marker='o', markersize=2, label=f"Trajectory {idx+1}")
    plt.scatter(1, 1, color='red', label="End point")
    plt.title("Nelder–Mead Optimisation Paths")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x = np.linspace(-2, 2, LINSPACE_POINTS)
    y = np.linspace(-2, 2, LINSPACE_POINTS)
    X, Y = np.meshgrid(x, y)
    f = rosenbrock(X, Y)

    plot_nelder_mead(X, Y, f)
