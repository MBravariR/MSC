# -*- coding: utf-8 -*-
"""PA Tarea 01.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YV66l9vWszghGLdTZfpyd1WU4HezZTWQ

# Gráficas de una elipse y una función 3D

## Librerias
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""## Ecuación de la elipse: $\frac{x^2}{9} + \frac{y^2}{25} = 1$"""

a = 3  # Raíz de 9
b = 5  # Raíz de 25
theta = np.linspace(0, 2*np.pi, 100)

x = a * np.cos(theta)
y = b * np.sin(theta)

"""## Gráfica de la elipse: $\frac{x^2}{9} + \frac{y^2}{25} = 1$"""

fig, ax = plt.subplots()
ax.set_title('Gráfica de la elipse: $\\frac{x^2}{9} + \\frac{y^2}{25} = 1$', fontsize=21)
ax.text(-0.6, 0, '$\\frac{x^2}{9} + \\frac{y^2}{25} = 1$', fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))
ax.plot(x, y, color='green')
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.grid(True)

plt.show()

"""## Función f(x,y) = 9x^2 − 5y^2"""

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
x, y = np.meshgrid(x, y)

z = 9*x**2 - 5*y**2

"""## Gráfica 3D de la función f(x,y) = 9x^2 − 5y^2"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, cmap='viridis')

ax.set_title('Gráfica de la función $f(x, y) = 9x^2 - 5y^2$', fontsize=21)
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.set_zlabel('Eje z')

plt.show()