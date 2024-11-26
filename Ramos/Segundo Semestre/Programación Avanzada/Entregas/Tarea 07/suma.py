import sys

def main():
  # Obtenga dos sumandos desde la terminal 
  # o consola, y despues los sume

  if len(sys.argv) !=3:
    print("Usar: python suma.py <arg1> <arg2>")
    sys.exit(1)

  a = int(sys.argv[1])
  b = int(sys.argv[2])

  # Sumar los dos argumentos
  resultado = a + b
  print(f'el resultado de sumar {a} y {b} es: {resultado}')

if __name__=="__main__":
  main()