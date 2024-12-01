import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

# Configurar el comunicador global para paralelización
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 1000  # Dimensión de la matriz

# Crear la matriz A de tamaño nxn con valores aleatorios
A = PETSc.Mat().createAIJ([n, n], comm=PETSc.COMM_WORLD)  # Matriz dispersa en paralelo
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

# Crear la matriz identidad I
I = PETSc.Mat().createAIJ([n, n], comm=PETSc.COMM_WORLD)
I.setFromOptions()
I.setUp()

for i in range(start, end):
    I.setValue(i, i, 1.0)

I.assemble()

# Configuración del solucionador Krylov y el precondicionador
ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
ksp.setOperators(A)
ksp.setType('cg')  # Método de Gradientes Conjugados
ksp.getPC().setType('asm')  # Método Aditivo de Schwarz como precondicionador

# Crear la matriz inversa `x` en la que almacenaremos la solución
x = PETSc.Mat().createAIJ([n, n], comm=PETSc.COMM_WORLD)
x.setFromOptions()
x.setUp()

# Vector para la solución temporal
x_col = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
x_col.setSizes(n)
x_col.setFromOptions()

# Resolver A * x_col = I_col para cada columna de la matriz identidad
for j in range(n):
    # Crear el vector columna de la identidad para el índice j
    I_col = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)
    I_col.set(0)
    
    # Asignar el valor de la columna j en el rango correspondiente
    if start <= j < end:
        I_col.setValue(j, 1.0)
    I_col.assemble()

    # Resolver el sistema para la columna j
    ksp.solve(I_col, x_col)

    # Insertar los valores locales de x_col en la columna j de la matriz x
    for i in range(start, end):
        x.setValue(i, j, x_col[i])

x.assemble()

if rank == 0:
    print("Matriz inversa calculada con éxito")