import math

class Punto2D:
  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y

  def __str__(self):
    return f'({self.x}, {self.y})'

  def distancia(self, otro_punto):
    return math.sqrt((self.x - otro_punto.x) ** 2 + (self.y - otro_punto.y) ** 2)
    
  def __abs__(self):
    return math.sqrt(self.x ** 2 + self.y ** 2)
