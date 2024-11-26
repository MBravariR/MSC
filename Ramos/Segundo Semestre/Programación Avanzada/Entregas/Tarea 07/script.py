from petsc4py import PETSc

def create_matrix():
    # Crear una matriz PETSc de tamaño 10x10
    A = PETSc.Mat().create(PETSc.COMM_WORLD)
    A.setSizes((10, 10))
    A.setFromOptions()
    A.setUp()

    # Llenar la matriz con valores (aquí se usa una matriz de ejemplo)
    for i in range(10):
        for j in range(10):
            A[i, j] = i + j + 1.0

    A.assemble()
    return A

def invert_matrix():
    # Crear la matriz original
    A = create_matrix()
    
    # Crear un vector identidad para resolver Ax = b
    identity = PETSc.Mat().createDense([10, 10], comm=PETSc.COMM_WORLD)
    identity.setFromOptions()
    identity.setUp()
    
    # Inicializar matriz inversa
    inverse = PETSc.Mat().createDense([10, 10], comm=PETSc.COMM_WORLD)
    inverse.setFromOptions()
    inverse.setUp()

    # Resolver para cada columna de la matriz identidad y almacenar en la inversa
    for i in range(10):
        b = PETSc.Vec().create(PETSc.COMM_WORLD)
        b.setSizes(10)
        b.setFromOptions()
        b.set(0.0)
        b[i] = 1.0  # Vector con un 1 en la posición i (columna de identidad)
        b.assemble()
        
        # Vector solución
        x = A.createVecRight()
        
        # Configurar y resolver el sistema lineal
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)
        
        # Colocar la solución en la matriz inversa
        for j in range(10):
            inverse[j, i] = x[j]
    
    inverse.assemble()
    return inverse

if __name__ == "__main__":
    # Llamar a la función para invertir la matriz
    inverse_matrix = invert_matrix()
    
    # Mostrar el resultado
    if PETSc.COMM_WORLD.getRank() == 0:
        print("Matriz inversa obtenida:")
        for i in range(10):
            print([inverse_matrix[i, j] for j in range(10)])