def sumarNNumeros():
    N = int(input("Ingrese cantidad de Números a sumar: "))
    total = 0.0

    for i in range(N):
        numero = float(input("Ingrese un número: "))
        total += numero

    print(f"Suma total = {total}")
    
def sumarNNumerosParesPositivos():
    n = int(input("Ingrese cantidad de Números a sumar: "))
    total = 0.0

    for i in range(n):
        numero = float(input("Ingrese un número: "))
        if numero % 2 == 0 and numero > 0:
            total += numero

    print(f"Suma total de todos los números pares = {total}")

def serieFibonacci():
    n = int(input("Ingrese cantidad de Números a generar: "))
    a, b = 0, 1
    fibonacci_sequence = []

    for _ in range(n):
        fibonacci_sequence.append(a)
        a, b = b, a + b

    print(f"Serie Fibonacci: {fibonacci_sequence}")

def quintupleNumero():
    import math

    a = 3
    b = 2
    c = -1

    discriminante = b**2 - 4 * a * c

    x1 = (-b + math.sqrt(discriminante)) / (2 * a)
    x2 = (-b - math.sqrt(discriminante)) / (2 * a)

    print(f"Los valores de los números son: {x1} y {x2}")
    
def aguaDiaria():
    comida = 1.0
    metabolismo = 0.3
    inhalar = 0.05
    exhalar = 0.4
    sudor = 0.2
    orina = 1.4
    heces = 2.3
    piel = 0.35

    salida = sudor + orina + heces + piel + exhalar - inhalar
    bebida = salida - comida - metabolismo

    print(f"La cantidad de agua que debe tomar por día es: {bebida} litros")
    
def constanteE():
    n = int(input("Ingrese N: "))
    total = 0.0
    factorial = 1

    for i in range(n + 1):
        total += 1 / factorial
        factorial *= i + 1

    print(f"Valor de e = {total}")
    
def aproximarPi():
    n = int(input("Ingrese la cantidad de términos: "))
    suma = 0.0

    for i in range(n + 1):
        suma += ((-1) ** i) / (2 * i + 1)

    pi = 4 * suma
    print(f"Valor aproximado de PI = {pi}")
