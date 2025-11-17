import matplotlib.pyplot as plt
import numpy as np


LINSPACE_POINTS = 50 

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


x = np.linspace(-2, 2, LINSPACE_POINTS)
y = np.linspace(-2, 2, LINSPACE_POINTS)
X, Y = np.meshgrid(x, y)
f = rosenbrock(X, Y)

plt.figure(figsize=(8, 7))
cs = plt.contourf(X, Y, f, levels=70, cmap='viridis')
plt.scatter(1, 1, color='red', s=60, label='Global Minimum (1, 1)')
plt.legend()
plt.colorbar(cs)

plt.title("Rosenbrock Function 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(1, 1, color='red', s=60, label='Global Minimum (1, 1)')

surf = ax.plot_surface(X, Y, f, cmap='viridis')

fig.colorbar(surf, ax=ax, shrink=0.6, label="Function value")

ax.set_title("Rosenbrock Function 3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.legend()
plt.show()
