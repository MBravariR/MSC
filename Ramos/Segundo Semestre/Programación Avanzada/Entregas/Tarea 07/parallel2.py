from petsc4py import PETSc
import numpy as np


def create_matrix(n):
    # Crear una matriz PETSc de tamaño 10x10
    A = PETSc.Mat().create(PETSc.COMM_WORLD)
    A.setSizes((n, n))
    A.setFromOptions()
    A.setUp()

    # Llenar la matriz con valores de ejemplo
    rank = PETSc.COMM_WORLD.getRank()
    for i in range(n):
        for j in range(n):
            value = (i + 1) * (j + 1) + rank  # Valor dependiente del rango
            A.setValue(i, j, value)

    A.assemble()
    return A


def invert_matrix(A):
    # Crear un vector identidad y matriz inversa
    identity = PETSc.Mat().createDense([10, 10], comm=PETSc.COMM_WORLD)
    identity.setFromOptions()
    identity.setUp()

    inverse = PETSc.Mat().createDense([10, 10], comm=PETSc.COMM_WORLD)
    inverse.setFromOptions()
    inverse.setUp()

    # Resolver para cada columna de la matriz identidad y almacenar en la inversa
    for i in range(10):
        b = PETSc.Vec().create(PETSc.COMM_WORLD)
        b.setSizes(10)
        b.setFromOptions()
        b.set(0.0)
        b.setValue(i, 1.0)  # Vector con un 1 en la posición i
        b.assemble()

        # Vector solución
        x = A.createVecRight()

        # Configurar y resolver el sistema lineal
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)

        # Almacenar resultado en la matriz inversa
        for j in range(10):
            inverse.setValue(j, i, x[j])

    inverse.assemble()
    return inverse


def print_matrix(mat):
    # Imprimir la matriz completa solo desde el proceso principal
    if PETSc.COMM_WORLD.getRank() == 0:
        local_array = mat.getDenseArray()
        print("Matriz:")
        print(np.array(local_array))


if __name__ == "__main__":
    # Crear y ensamblar matriz A
    A = create_matrix(10)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Matriz Original A:")
    print_matrix(A)

    # Invertir la matriz
    inverse_matrix = invert_matrix(A)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Matriz Inversa de A:")
    print_matrix(inverse_matrix)