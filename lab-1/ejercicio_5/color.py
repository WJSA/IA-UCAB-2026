class Color:
    
    def __init__(self, red: int, green: int, blue: int) -> None:
        for nombre, valor in (("red", red), ("green", green), ("blue", blue)):
            if not isinstance(valor, int) or not (1 <= valor <= 255):
                raise ValueError(
                    f"{nombre} debe ser un entero entre 1 y 255, recibido: {valor!r}"
                )
        self.red = red
        self.green = green
        self.blue = blue

    def __str__(self) -> str:
        return f"Color(red={self.red}, green={self.green}, blue={self.blue})"
