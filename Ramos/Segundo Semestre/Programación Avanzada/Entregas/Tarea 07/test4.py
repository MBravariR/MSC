import numpy as np
from petsc4py import PETSc

# Configurar el comunicador global para paralelización
comm = PETSc.COMM_WORLD
n = 1000  # Dimensión de la matriz

# Crear la matriz A de tamaño nxn con valores aleatorios
A = PETSc.Mat().createAIJ([n, n], comm=comm)  # Matriz dispersa en paralelo
A.setFromOptions()
A.setUp()

# Generar valores aleatorios y fijar semilla para reproducibilidad
np.random.seed(0)
random_values = np.random.rand(n, n) * 10

# Insertar los valores en la matriz PETSc
start, end = A.getOwnershipRange()
for i in range(start, end):
    for j in range(n):
        A.setValue(i, j, random_values[i, j])

A.assemble()

# Configuración del solucionador Krylov y el precondicionador
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')  # Método de Gradientes Conjugados
ksp.getPC().setType('asm')  # Método Aditivo de Schwarz como precondicionador

# Crear la matriz inversa `x` en la que almacenaremos la solución
x = PETSc.Mat().createAIJ([n, n], comm=comm)
x.setFromOptions()
x.setUp()

# Vector para la solución temporal
x_col = PETSc.Vec().create(comm=comm)
x_col.setSizes(n)
x_col.setFromOptions()

# Resolver A * x_col = I_col para cada columna de la matriz identidad
for j in range(n):
    # Crear el vector columna de la identidad
    I_col = PETSc.Vec().create(comm=comm)
    I_col.setSizes(n)
    I_col.setFromOptions()
    I_col.set(0)
    if start <= j < end:
        I_col[j - start] = 1.0
    I_col.assemble()

    # Resolver el sistema para la columna j
    ksp.solve(I_col, x_col)

    # Insertar los valores de la solución en la columna j de x
    for i in range(start, end):
        x.setValue(i, j, x_col[i - start])

x.assemble()

print("Matriz inversa calculada con éxito.")