from petsc4py import PETSc
import numpy as np


def create_matrix(n):
    # Crear una matriz PETSc de tamaño nxn
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


def invert_matrix(A, n):
    # Crear matriz inversa
    inverse = PETSc.Mat().createDense([n, n], comm=PETSc.COMM_WORLD)
    inverse.setFromOptions()
    inverse.setUp()

    # Resolver para cada columna de la matriz identidad y almacenar en la inversa
    for i in range(n):
        # Crear el vector b como una columna de la identidad
        b = PETSc.Vec().create(PETSc.COMM_WORLD)
        b.setSizes(n)
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
        for j in range(n):
            inverse.setValue(j, i, x[j])

    inverse.assemble()
    return inverse


def gather_matrix(mat, n):
    # Obtener los valores locales de cada proceso
    local_values = mat.getValues(range(n), range(n))

    # Recopilar todos los valores locales en el proceso principal
    all_values = PETSc.COMM_WORLD.gather(local_values, root=0)

    # Solo el proceso principal ensamblará la matriz completa
    if PETSc.COMM_WORLD.getRank() == 0:
        full_matrix = np.zeros((n, n))
        for proc_values in all_values:
            # Copiar los valores locales de cada proceso en la matriz completa
            full_matrix += proc_values  # Sumar las contribuciones de cada proceso
        print("Matriz:")
        print(full_matrix)


if __name__ == "__main__":
    n = 10  # Tamaño de la matriz
    # Crear y ensamblar matriz A
    A = create_matrix(n)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Matriz Original A:")
    gather_matrix(A, n)

    # Invertir la matriz
    inverse_matrix = invert_matrix(A, n)
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Matriz Inversa de A:")
    gather_matrix(inverse_matrix, n)