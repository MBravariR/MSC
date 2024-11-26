from petsc4py import PETSc
import numpy as np


def create_random_matrix(n):
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
    return A


def find_inverse_matrix(A, n):
    # Crear la matriz B que contendrá la solución tal que A * B = I
    B = PETSc.Mat().createDense([n, n], comm=PETSc.COMM_WORLD)
    B.setFromOptions()
    B.setUp()

    # Resolver para cada columna de la matriz identidad y almacenar en B
    for i in range(n):
        # Crear el vector b como una columna de la identidad
        b = PETSc.Vec().create(PETSc.COMM_WORLD)
        b.setSizes(n)
        b.setFromOptions()
        b.set(0.0)
        b.setValue(i, 1.0)  # Vector con un 1 en la posición i
        b.assemble()

        # Vector solución para la columna i de B
        x = A.createVecRight()

        # Configurar y resolver el sistema lineal
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)

        # Almacenar resultado en la columna i de la matriz B
        for j in range(n):
            B.setValue(j, i, x[j])

    B.assemble()
    return B


def verify_identity(A, B, n):
    # Multiplicar A * B y verificar si da como resultado la identidad
    C = PETSc.Mat().createDense([n, n], comm=PETSc.COMM_WORLD)
    C.setFromOptions()
    C.setUp()

    # Realizar la multiplicación A * B y almacenar el resultado en C
    A.matMult(B, C)

    # Imprimir el resultado si es el proceso principal
    if PETSc.COMM_WORLD.getRank() == 0:
        identity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                identity_matrix[i, j] = C.getValue(i, j)
        print("Resultado de A * B:")
        print(identity_matrix)
        print("Matriz identidad esperada:")
        print(np.eye(n))


if __name__ == "__main__":
    n = 10  # Tamaño de la matriz
    # Crear y ensamblar matriz A con valores aleatorios
    A = create_random_matrix(n)

    # Encontrar la matriz B tal que A * B = I
    B = find_inverse_matrix(A, n)

    # Verificar el resultado
    verify_identity(A, B, n)