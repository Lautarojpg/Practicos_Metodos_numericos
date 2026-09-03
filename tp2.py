import math
import time
def errorAbsoluto(a,b):
    
    Resultado = abs(a-b)

    print(f"El error absoluto es  {Resultado} ")

def errorRelativo(a,b):
       
       Resultado=abs((a-b)/a)
       print(f"El error relativo es  {Resultado} ")

def ejercicio1(p, p_aproximado):

    inicio = time.time()

    errorAbsoluto(p, p_aproximado)
    errorRelativo(p, p_aproximado)

    fin = time.time()

    print(f"Tiempo de ejecución: {fin - inicio} segundos")

def ejercicio2(p):
     
      inicio=time.time()
      aproximado=round(p,3)
      print(f"el valor aproximado es {aproximado}")
      errorAbsoluto(p,aproximado)
      errorRelativo(p,aproximado)
      fin=time.time()
      print(f"Tiempo de ejecución: {fin-inicio} segundos")
      

#ejercicio2((2/9) * (9/7))

def ejercicio3(p):

    inicio = time.time()

    aproximado = math.trunc(p * 1000) / 1000

    print(f"Valor exacto: {p}")
    print(f"Valor truncado: {aproximado}")

    errorAbsoluto(p, aproximado)
    errorRelativo(p, aproximado)

    fin = time.time()

    print(f"Tiempo de ejecución: {fin - inicio} segundos")


''' Punto 3
ejercicio3(133 + 0.921)
ejercicio3(133 - 0.499)
ejercicio3((121 - 119) - 0.327)
ejercicio3((121 - 0.327) - 119)
ejercicio3((2/9) * (9/7))
'''
def redondear4(numero):
    return float(f"{numero:.4g}")


def ejercicio4a():

    inicio = time.time()
    a = 1
    b = 62.10
    c = 1

    b2 = redondear4(b ** 2)
    cuatro_ac = redondear4(4 * a * c)

    discriminante = redondear4(b2 - cuatro_ac)
    # raíz del discriminante
    raiz = redondear4(math.sqrt(discriminante))

    print("VALORES")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")

    print(f"\nb^2 = {b2}")
    print(f"4ac = {cuatro_ac}")
    print(f"Discriminante = {discriminante}")
    print(f"raiz del Discriminante = {raiz}")

    numerador1 = redondear4(-b + raiz)
    denominador = redondear4(2 * a)
    x1 = redondear4(numerador1 / denominador)

    numerador2 = redondear4(-b - raiz)
    x2 = redondear4(numerador2 / denominador)

    print(f"\nNumerador x1 = {numerador1}")
    print(f"x1 aproximado = {x1}")

    print(f"\nNumerador x2 = {numerador2}")
    print(f"x2 aproximado = {x2}")

    # Valores 
    x1_exacto = -0.01610723
    x2_exacto = -62.08390

    print("\n--- ERROR DE x1 ---")
    errorAbsoluto(x1_exacto, x1)
    errorRelativo(x1_exacto, x1)

    print("\n--- ERROR DE x2 ---")
    errorAbsoluto(x2_exacto, x2)
    errorRelativo(x2_exacto, x2)

    fin = time.time()

    print(f"\nTiempo de ejecución: {fin - inicio} segundos")


#ejercicio4a()

def ejercicio4b():

    inicio = time.time()

    a = 1
    b = 62.10
    c = 1

    b2 = redondear4(b ** 2)
    cuatro_ac = redondear4(4 * a * c)
    discriminante = redondear4(b2 - cuatro_ac)

    raiz = redondear4(math.sqrt(discriminante))

    print("VALORES")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")

    print(f"\nb^2 = {b2}")
    print(f"4ac = {cuatro_ac}")
    print(f"Discriminante = {discriminante}")
    print(f"raiz del Discriminante = {raiz}")

    numerador_x1 = redondear4(-2 * c)
    denominador_x1 = redondear4(b + raiz)
    x1 = redondear4(numerador_x1 / denominador_x1)

    
    numerador_x2 = redondear4(-2 * c)
    denominador_x2 = redondear4(b - raiz)
    x2 = redondear4(numerador_x2 / denominador_x2)

    print("\n--- x1 RACIONALIZADO ---")
    print(f"Numerador: {numerador_x1}")
    print(f"Denominador: {denominador_x1}")
    print(f"x1 aproximado: {x1}")

    print("\n--- x2 RACIONALIZADO ---")
    print(f"Numerador: {numerador_x2}")
    print(f"Denominador: {denominador_x2}")
    print(f"x2 aproximado: {x2}")

    # Valores de referencia
    x1_exacto = -0.01610723
    x2_exacto = -62.08390

    print("\n--- ERROR DE x1 ---")
    errorAbsoluto(x1_exacto, x1)
    errorRelativo(x1_exacto, x1)

    print("\n--- ERROR DE x2 ---")
    errorAbsoluto(x2_exacto, x2)
    errorRelativo(x2_exacto, x2)

    fin = time.time()

    print(f"\nTiempo de ejecución: {fin - inicio} segundos")


def truncar3(numero):
    if numero == 0:
        return 0

    cantidad = 3 - int(math.floor(math.log10(abs(numero)))) - 1
    factor = 10 ** cantidad

    return math.trunc(numero * factor) / factor


def redondear3(numero):
    return float(f"{numero:.3g}")


def ejercicio5a():

    inicio = time.time()

    x = 4.71

    # EXACTO 
    x2_exacto = x ** 2
    x3_exacto = x ** 3
    seis_x2_exacto = 6 * x2_exacto
    tres_x_exacto = 3 * x

    resultado_exacto = (
        x3_exacto
        - seis_x2_exacto
        + tres_x_exacto
        - 0.149
    )

    # TRES DIGITOS TRUNCADOS 
    x2_truncado = truncar3(x ** 2)
    x3_truncado = truncar3(x2_truncado * x)
    seis_x2_truncado = truncar3(6 * x2_truncado)
    tres_x_truncado = truncar3(3 * x)

    resultado_truncado = (
        x3_truncado
        - seis_x2_truncado
        + tres_x_truncado
        - 0.149
    )

    # TRES DÍGITOS REDONDEADOS 
    x2_redondeado = redondear3(x ** 2)
    x3_redondeado = redondear3(x2_redondeado * x)
    seis_x2_redondeado = redondear3(6 * x2_redondeado)
    tres_x_redondeado = redondear3(3 * x)

    resultado_redondeado = (
        x3_redondeado
        - seis_x2_redondeado
        + tres_x_redondeado
        - 0.149
    )

    print("\n              x       x²       x³      6x²      3x")
    print("Exacto       ", x, x2_exacto, x3_exacto,
          seis_x2_exacto, tres_x_exacto)

    print("Truncado     ", x, x2_truncado, x3_truncado,
            seis_x2_truncado,   tres_x_truncado)

    print("Redondeado   ", x, x2_redondeado, x3_redondeado,
            seis_x2_redondeado,   tres_x_redondeado)

    print("\n--- RESULTADOS ---")
    print(f"Resultado exacto: {resultado_exacto}")
    print(f"Resultado truncado: {resultado_truncado}")
    print(f"Resultado redondeado: {resultado_redondeado}")

    print("\n--- ERROR DE TRUNCAMIENTO ---")
    errorAbsoluto(resultado_exacto, resultado_truncado)
    errorRelativo(resultado_exacto, resultado_truncado)

    print("\n--- ERROR DE REDONDEO ---")
    errorAbsoluto(resultado_exacto, resultado_redondeado)
    errorRelativo(resultado_exacto, resultado_redondeado)

    fin = time.time()

    print(f"\nTiempo de ejecución: {fin - inicio} segundos")



#ejercicio5a()
def ejercicio5b():

    inicio = time.time()

    x = 4.71

    # EXACTO
    resultado_exacto = ((x - 6) * x + 3) * x - 0.149

    # TRES DIGITOS TRUNCADOS 
    paso1_truncado = truncar3(x - 6)
    paso2_truncado = truncar3(paso1_truncado * x)
    paso3_truncado = truncar3(paso2_truncado + 3)
    paso4_truncado = truncar3(paso3_truncado * x)
    resultado_truncado = truncar3(paso4_truncado - 0.149)

    # TRES DÍGITOS REDONDEADO
    paso1_redondeado = redondear3(x - 6)
    paso2_redondeado = redondear3(paso1_redondeado * x)
    paso3_redondeado = redondear3(paso2_redondeado + 3)
    paso4_redondeado = redondear3(paso3_redondeado * x)
    resultado_redondeado = redondear3(paso4_redondeado - 0.149)

    print("\n--- FORMA ANIDADA ---")
    print("f(x) = ((x - 6) * x + 3) * x - 0.149")

    print("\n--- VALORES EXACTOS ---")
    print(f"x = {x}")
    print(f"x - 6 = {x - 6}")
    print(f"(x - 6) * x = {(x - 6) * x}")
    print(f"(x - 6) * x + 3 = {(x - 6) * x + 3}")
    print(f"((x - 6) * x + 3) * x = {((x - 6) * x + 3) * x}")

    print("\n TRES DÍGITOS TRUNCADOS ")
    print(f"x - 6 = {paso1_truncado}")
    print(f"(x - 6) * x = {paso2_truncado}")
    print(f"(x - 6) * x + 3 = {paso3_truncado}")
    print(f"((x - 6) * x + 3) * x = {paso4_truncado}")
    print(f"Resultado truncado = {resultado_truncado}")

    print("\n TRES DÍGITOS REDONDEADOS ")
    print(f"x - 6 = {paso1_redondeado}")
    print(f"(x - 6) * x = {paso2_redondeado}")
    print(f"(x - 6) * x + 3 = {paso3_redondeado}")
    print(f"((x - 6) * x + 3) * x = {paso4_redondeado}")
    print(f"Resultado redondeado = {resultado_redondeado}")

    print("\n ERRORES DEL TRUNCAMIENTO ")
    errorAbsoluto(resultado_exacto, resultado_truncado)
    errorRelativo(resultado_exacto, resultado_truncado)

    print("\n ERRORES DEL REDONDEO ")
    errorAbsoluto(resultado_exacto, resultado_redondeado)
    errorRelativo(resultado_exacto, resultado_redondeado)

    fin = time.time()

    print(f"\nTiempo de ejecución: {fin - inicio} segundos")


ejercicio5b()