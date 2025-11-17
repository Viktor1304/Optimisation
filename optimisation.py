import matplotlib.pyplot as plt
import numpy as np


LINSPACE_POINTS = 50 


def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def rosenbrock_grad(x: float, y: float):
    df_dx: float = -400 * x * (y - x**2) - 2 * (1 - x)
    df_dy: float = 200 * (y - x**2)

    return np.array([df_dx, df_dy])

def rosenbrock_hessian(x: float, y: float):
    d2f_dx2 = 1200 * (x**2) - 400 * y + 2
    d2f_dxdy: float = -400 * x
    d2f_dydx: float = -400 * x
    d2f_dy2: float = 200

    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dydx, d2f_dy2]])

def rosenbrock_jacobian(x: float):
    dr1_dx: float = -20.0 * x
    dr1_dy: float = 10.0
    dr2_dx: float = -1.0
    dr2_dy: float = 0.0

    return np.array([[dr1_dx, dr1_dy], [dr2_dx, dr2_dy]])

def gradient_descent(x0: float, y0: float, iterations: int = 10000):
    trajectory = [np.array([x0, y0])]
    for _ in range(iterations):
        grad = rosenbrock_grad(x0, y0)
        alpha = 0.001
        x0 = x0 - alpha * grad[0]
        y0 = y0 - alpha * grad[1]

        trajectory.append(np.array([x0, y0]))

    return np.array(trajectory)

def newton_method(x0: float, y0: float, iterations: int = 100):
    trajectory = [np.array([x0, y0])]
    for _ in range(iterations):
        H = rosenbrock_hessian(x0, y0)
        g = rosenbrock_grad(x0, y0)
        step_x, step_y = np.linalg.solve(H, g)
        x0 -= step_x
        y0 -= step_y

        trajectory.append(np.array([x0, y0]))

    return np.array(trajectory)

def gauss_newton(x0: float, y0: float, iterations: int = 100):
    trajectory = [np.array([x0, y0])]
    for _ in range(iterations):
        residual1: float = 10  * (y0 - x0**2)
        residual2: float = 1 - x0

        J = rosenbrock_jacobian(x0)
        step_x, step_y = np.linalg.solve(J.T @ J, J.T @ np.array([residual1, residual2]))
        x0 -= step_x
        y0 -= step_y

        trajectory.append(np.array([x0, y0]))

    return np.array(trajectory)

def plot_trajectories(X, Y, method, name):
    plt.figure(figsize=(7,6))
    plt.contour(X, Y, f, levels=50, cmap='viridis')
    for idx in range(3):
        x0: float = np.random.uniform(-1, 1)
        y0: float = np.random.uniform(-1, 1)
        path = method(x0, y0)
        plt.scatter(x0, y0, label="Starting point")
        plt.plot(path[:,0], path[:,1], marker='o', markersize=2, label=f"Trajectory {idx+1}")
    plt.scatter(1, 1, color='red', label="End point")
    plt.title(f"{name} Optimisation Paths")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x = np.linspace(-2, 2, LINSPACE_POINTS)
    y = np.linspace(-2, 2, LINSPACE_POINTS)
    X, Y = np.meshgrid(x, y)
    f = rosenbrock(X, Y)

    plot_trajectories(X, Y, gradient_descent, "Gradient Descent")
    plot_trajectories(X, Y, newton_method, "Newton's Method")
    plot_trajectories(X, Y, gauss_newton, "Gauss-Newton")

