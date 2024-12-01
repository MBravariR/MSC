import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_geometry(A, B, C, D, E, color='yellow'):
    puntos = [A, C, E, B, D]  # Orden de los puntos para formar la estrella
    
    if color == "red":
      color1 = "Rojo"
    elif color == "yellow":
      color1 = "Amarillo"
    elif color == "green":
      color1 = "Verde"
    elif color == "blue":
      color1 = "Azul"
    else: 
      color1 = "bonito"


    fig, ax = plt.subplots()
    
    # Crear el polígono de la estrella
    estrella = Polygon([(p.x, p.y) for p in puntos], closed=True, fill=True, edgecolor='black', facecolor=color)
    
    ax.add_patch(estrella)
    
    plt.title(f"Estrella de 5 puntas de un color {color1}", fontsize=14)
    # Configuración de los ejes
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    
    plt.grid(True)
    plt.show()