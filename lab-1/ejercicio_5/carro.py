from datetime import date
from carroTipo import CarroTipo
from color import Color
from motor import Motor

class Carro:

    def __init__(
        self,
        marca: str,
        modelo: str,
        año: int,
        color: Color,
        placa: str,
        tipo: CarroTipo,
        motor: Motor,
    ) -> None:
        año_actual = date.today().year
        if not (1900 < año <= año_actual):
            print("Error: el año debe ser mayor a 1900 y menor o igual al año actual")
            raise ValueError("año inválido")
        if not isinstance(color, Color):
            raise TypeError("color debe ser instancia de Color")
        if not isinstance(tipo, CarroTipo):
            raise TypeError("tipo debe ser instancia de CarroTipo")
        if not isinstance(motor, Motor):
            raise TypeError("motor debe ser instancia de Motor")
        self.marca = marca
        self.modelo = modelo
        self.año = año
        self.color = color
        self.placa = placa
        self.tipo = tipo
        self.motor = motor
        self._en_movimiento = False
        self._luces_encendidas = False

    def acelerar(self) -> None:
        print("Acelerando el carro")
        self._en_movimiento = True

    def frenar(self) -> None:
        if self._en_movimiento:
            print("Frenando el carro")
            self._en_movimiento = False
        else:
            print("El carro está detenido")

    def parar(self) -> None:
        if self._en_movimiento:
            print("Parando el carro")
            self._en_movimiento = False
        else:
            print("No hace falta parar el carro, el carro ya está detenido")

    def luces(self) -> None:
        if not self._luces_encendidas:
            print("Prendiendo las luces")
            self._luces_encendidas = True
        else:
            print("Apagando las luces")
            self._luces_encendidas = False

    def __str__(self) -> str:
        estado_mov = "en movimiento" if self._en_movimiento else "detenido"
        estado_luces = "encendidas" if self._luces_encendidas else "apagadas"
        return (
            f"Carro(marca={self.marca!r}, modelo={self.modelo!r}, año={self.año}, "
            f"color={self.color}, placa={self.placa!r}, tipo={self.tipo}, "
            f"motor={self.motor}, estado={estado_mov}, luces={estado_luces})"
        )
