import sys

def contar_nucleotidos(secuencia, tamano):
    conteos = {}
    
    # Conteo de fragmentos
    for i in range(len(secuencia) - tamano + 1):
        fragmento = secuencia[i:i+tamano]
        if fragmento in conteos:
            conteos[fragmento] += 1
        else:
            conteos[fragmento] = 1
            
    # Ordenamiento por frecuencia
    resultados = sorted(conteos.items(), key=lambda x: x[1])
    
    for nucleotido, frec in resultados:
        print(f"{nucleotido} {frec}")
        
    # Gráfica con MatPlotLib
    # encontré la documentación en: https://matplotlib.org/stable/tutorials/pyplot.html
    try:
        import matplotlib.pyplot as plt
        nombres = [n for n, f in resultados]
        frecuencias = [f for n, f in resultados]
        
        plt.figure(figsize=(10, 6))
        plt.bar(nombres, frecuencias, color='skyblue', edgecolor='black')
        plt.xlabel('Secuencias de Nucleótidos')
        plt.ylabel('Frecuencia')
        plt.title(f'Frecuencias de nucleótidos (Tamaño {tamano})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('frecuencias.png')
        # La grafica se guarda sin interrumpir la salida en consola
    except ImportError:
        pass

if __name__ == "__main__":
    try:
        secuencia = input("Secuencia: ").strip()
        tamano = int(input("Tamaño: ").strip())
        contar_nucleotidos(secuencia, tamano)
    except ValueError:
        print("Valor invalido")