import sys

try:
    anio = int(sys.argv[1])
except ValueError:
    print("Error, año inválido")
    sys.exit(1)

if anio < 1900 or anio > 2200:
    print("Error, año inválido")
    sys.exit(1)

if (anio % 4 == 0 and anio % 100 != 0) or (anio % 400 == 0):
    print("El año es bisiesto")
else:
    print("El año no es bisiesto")