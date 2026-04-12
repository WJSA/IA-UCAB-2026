from carro import Carro
from carroTipo import CarroTipo
from color import Color
from motor import Motor
from motorTipo import MotorTipo

def main() -> None:
    print("=== Enumeraciones ===")
    print("MotorTipo:", list(MotorTipo))
    print("CarroTipo:", list(CarroTipo))

    print("\n=== Color ===")
    c = Color(220, 20, 60)
    print("__str__:", c)

    print("\n=== Motor ===")
    m = Motor(MotorTipo.HIBRIDO, 180, 350, 220)
    print("__str__:", m)

    print("\n=== Carro (año válido) ===")
    carro = Carro(
        marca="Toyota",
        modelo="Corolla",
        año=2020,
        color=c,
        placa="ABC-123",
        tipo=CarroTipo.SEDAN,
        motor=m,
    )
    print("__str__:", carro)

    print("\n=== Métodos de Carro ===")
    carro.acelerar()
    carro.frenar()
    carro.parar()
    carro.parar()
    carro.luces()
    carro.luces()
    print("Estado final:", carro)

    print("\n=== Intento de Carro con año inválido ===")
    try:
        Carro(
            "X",
            "Y",
            1899,
            Color(10, 10, 10),
            "X",
            CarroTipo.SUV,
            Motor(MotorTipo.ELECTRICO, 100, 200, 150),
        )
    except ValueError:
        print("(Carro no creado: ValueError tras mensaje de error)")


if __name__ == "__main__":
    main()
