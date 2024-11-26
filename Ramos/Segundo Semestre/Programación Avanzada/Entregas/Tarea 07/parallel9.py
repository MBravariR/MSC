from petsc4py import PETSc
import numpy as np

# Configurar el comunicador global para paralelización
comm = PETSc.COMM_WORLD
size = comm.getSize()  # Número total de procesos
rank = comm.getRank()  # Identificador del proceso actual

def create_random_matrix(n):
    np.random.seed(0)  # Fijar semilla para reproducibilidad

    # Create a nxn NumPy matrix with random values
    np_matrix = np.random.rand(n, n)
    np.set_printoptions(precision=3, suppress=True)

    #print("Matriz original (c/3 digitos):")
    #print(np_matrix)

    # Create a 10x10 PETSc matrix
    petsc_matrix = PETSc.Mat().create(comm=comm)
    petsc_matrix.setSizes((n, n))
    petsc_matrix.setFromOptions()
    petsc_matrix.setUp()

    # Fill the PETSc matrix with values from the NumPy matrix
    for i in range(n):
        for j in range(n):
            petsc_matrix.PETSc.setValue(i, j, np_matrix[i, j])

    # Assemble the PETSc matrix to finalize it
    petsc_matrix.assemble()

    return petsc_matrix

def invert_matrix(A, n):
    # Crear matriz inversa
    inverse = PETSc.Mat().createDense([n, n], comm=comm)
    inverse.setFromOptions()
    inverse.setUp()

    # Resolver para cada columna de la matriz identidad y almacenar en la inversa
    for i in range(n):
        # Crear el vector b como una columna de la identidad
        b = PETSc.Vec().create(comm=comm)
        b.setSizes(n)
        b.setFromOptions()
        b.set(0.0)
        b.setValue(i, 1.0)  # Vector con un 1 en la posición i
        b.assemble()

        # Vector solución
        x = A.createVecRight()

        # Configurar y resolver el sistema lineal
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)

        # Almacenar resultado en la matriz inversa
        for j in range(n):
            inverse.setValue(j, i, x[j])

    inverse.assemble()
    return inverse

if __name__ == "__main__":
    n = 10  # Tamaño de la matriz
    # Crear y ensamblar matriz A con valores aleatorios
    print("Creando Matriz cuadrada A, con valores aleatorios de tamaño", n, "x", n, ".")
    A = create_random_matrix(n)

    # Invertir la matriz
    print("Invirtiendo la Matriz A.")
    inverse_matrix = invert_matrix(A, n)
    print("Terminado la operacion de invertir la matriz A.")