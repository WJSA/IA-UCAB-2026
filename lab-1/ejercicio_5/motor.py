from motorTipo import MotorTipo

class Motor:

    def __init__(
        self,
        tipo: MotorTipo,
        potencia: int,
        torque: int,
        velocidad: int,
    ) -> None:
        if not isinstance(tipo, MotorTipo):
            raise TypeError("tipo debe ser una instancia de MotorTipo")
        for nombre, valor in (
            ("potencia", potencia),
            ("torque", torque),
            ("velocidad", velocidad),
        ):
            if not isinstance(valor, int):
                raise TypeError(f"{nombre} debe ser entero")
        self.tipo = tipo
        self.potencia = potencia
        self.torque = torque
        self.velocidad = velocidad

    def __str__(self) -> str:
        return (
            f"Motor(tipo={self.tipo}, potencia={self.potencia} HP, "
            f"torque={self.torque}, velocidad_max={self.velocidad} km/h)"
        )
