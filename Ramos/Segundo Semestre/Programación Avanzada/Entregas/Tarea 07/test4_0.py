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


# Specify the column index to extract
col_index = 2

# Create a vector to store the extracted column
I_col = PETSc.Vec().createSeq(n)
I_col.setFromOptions()

# Fill col_vector with the values from the specified column
for i in range(n):
    value = A.getValue(i, col_index)  # Get the value at (i, col_index)
    I_col.setValue(i, value)

I_col.assemble()



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
    # Fill col_vector with the values from the specified column
    for i in range(n):
        value = A.getValue(i, col_index)  # Get the value at (i, col_index)
        I_col.setValue(i, value)

    I_col.assemble()

    # Resolver el sistema para la columna j
    ksp.solve(I_col, x_col)

    # Insertar los valores de la solución en la columna j de x
    for i in range(start, end):
        x.setValue(i, j, x_col[i - start])

x.assemble()




print("Matriz inversa calculada con éxito.")