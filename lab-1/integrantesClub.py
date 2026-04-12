ajedrez = set(input("Nombres de las personas en la ajedrez: ").split())
futbol = set(input("Nombres de las personas en el fútbol: ").split())
natacion = set(input("Nombres de las personas en la natación: ").split())
lectura = set(input("Nombres de las personas en la lectura: ").split())

en_todos = ajedrez & futbol & natacion & lectura

if not en_todos:
    print("No hay nombres en común")
else:
    print(",".join(sorted(en_todos)))
