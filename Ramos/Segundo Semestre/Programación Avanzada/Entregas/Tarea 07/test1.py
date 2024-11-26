import numpy as np
from petsc4py import PETSc

# Configurar el comunicador global para paralelización
comm = PETSc.COMM_WORLD
size = comm.getSize()  # Número total de procesos
rank = comm.getRank()  # Identificador del proceso actual

# Dimensión del sistema (por ejemplo, 1000 x 1000)
n = 1000
A = PETSc.Mat().createAIJ([n, n], comm=comm)  # Matriz dispersa en paralelo
A.setFromOptions()
A.setUp()

start, end = A.getOwnershipRange() # Rango de filas que maneja cada proceso

print(start, end)

# Crear una matriz PETSc de tamaño nxn con valores aleatorios
A = PETSc.Mat().create(PETSc.COMM_WORLD)
A.setSizes((n, n))
A.setFromOptions()
A.setUp()

# Generar valores aleatorios y fijar semilla para reproducibilidad
np.random.seed(0)
random_values = np.random.rand(n, n) * 10

# Insertar los valores en la matriz PETSc
for i in range(n):
    for j in range(n):
        A.setValue(i, j, random_values[i, j])

A.assemble()

# Crear una matriz PETSc de tamaño nxn con valores aleatorios
I = PETSc.Mat().create(PETSc.COMM_WORLD)
I.setSizes((n, n))
I.setFromOptions()
I.setUp()

# Insertar los valores en la matriz PETSc
for i in range(n):
    for j in range(n):
        if(i == j):
            I.setValue(i, j, 1)
        else:
            I.setValue(i, j, 0)

I.assemble()

# Crear una matriz PETSc de tamaño nxn con valores aleatorios
x = PETSc.Mat().create(PETSc.COMM_WORLD)
x.setSizes((n, n))
x.setFromOptions()
x.setUp()

# Insertar los valores en la matriz PETSc
for i in range(n):
    for j in range(n):
        x.setValue(i, j, 0)

A.assemble()

# Configuración del solucionador Krylov y el precondicionador
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')  # Método de Gradientes Conjugados
ksp.getPC().setType('asm')  # Metodo Aditivo de Schwarz como precondicionador





#PAx= #PI

# Resolver el sistema en paralelo
ksp.solve(I, x)
