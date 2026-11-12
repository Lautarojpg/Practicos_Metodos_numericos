"""
===============================================================================
UNIVERSIDAD NACIONAL DEL NORDESTE (UNNE)
FACULTAD DE CIENCIAS EXACTAS Y NATURALES Y AGRIMENSURA (FaCENA)
METODOS COMPUTACIONALES / CALCULO NUMERICO
Guia de Actividades Practicas N° 6: Metodos de Integracion Numerica
Resolucion de los Ejercicios 2 y 3 (Metodo de los Trapecios)
===============================================================================
Este script implementa de forma modular los metodos de Integracion Numerica
y resuelve los Ejercicios 2 y 3 de la Guia N° 6 aplicando el Metodo Compuesto
de los Trapecios:
  - metodo_trapecios(x, y)   : Regla Compuesta de los Trapecios
  - metodo_simpson_1_3(x, y) : Regla Compuesta de Simpson 1/3 (funcion modular)
  - metodo_simpson_3_8(x, y) : Regla de Simpson 3/8 (funcion modular)

Cumple con los estandares de la catedra (FaCENA - UNNE):
  * Variable independiente denominada uniformemente como 'x' en todo el codigo
    y en todas las leyendas de entrada y salida.
  * Resolucion tanto del Ejercicio 2 como del Ejercicio 3 utilizando
    exclusivamente el Metodo de los Trapecios.
  * Desglose explicito de terminos Extremos (E), Impares (I), Pares (P) y
    Resto de ordenadas interiores (R).
  * Reemplazo paso a paso en la formula teorica y verificacion por suma de
    areas de sub-trapecios individuales.
  * Ingreso de datos interactivo por teclado con opcion de valores por defecto.
  * Medicion precisa del tiempo de ejecucion con time.perf_counter().
===============================================================================
"""

import sys
import time

# Configuracion de codificacion UTF-8 para consola en Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# FUNCIONES DE UTILIDAD, VALIDACION Y FORMATO EN CONSOLA
# =============================================================================

def imprimir_banner(titulo, ancho=92, caracter="="):
    """Imprime un encabezado destacado en consola al estilo de la catedra."""
    print("\n" + caracter * ancho)
    print(f"{titulo:^{ancho}}")
    print(caracter * ancho)


def imprimir_subtitulo(subtitulo, ancho=92):
    """Imprime un subtitulo separador en consola."""
    print("\n" + "-" * ancho)
    print(f" >>> {subtitulo}")
    print("-" * ancho)


def mostrar_tabla_datos(x, y, nombre_x="x", nombre_y="y"):
    """Muestra la tabla de puntos (x_i, y_i) numerados con su indice i."""
    n = len(x) - 1
    print(f"\n  Tabla de valores ({len(x)} puntos -> n = {n} subintervalos):")
    print("  +" + "-" * 8 + "+" + "-" * 18 + "+" + "-" * 20 + "+" + "-" * 24 + "+")
    print(f"  | {'i':^6} | {nombre_x + '_i':^16} | {nombre_y + '_i':^18} | {'Clasificacion':^22} |")
    print("  +" + "-" * 8 + "+" + "-" * 18 + "+" + "-" * 20 + "+" + "-" * 24 + "+")
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i == 0 or i == n:
            clasif = "Extremo (E)"
        elif i % 2 != 0:
            clasif = "Impar (I)"
        else:
            clasif = "Par (P)"
        print(f"  | {i:^6} | {xi:^16.6g} | {yi:^18.6f} | {clasif:^22} |")
    print("  +" + "-" * 8 + "+" + "-" * 18 + "+" + "-" * 20 + "+" + "-" * 24 + "+")


def mostrar_desglose_subtrapecios(x, y, h):
    """Muestra una tabla verificadora con el area de cada trapecio elemental."""
    n = len(x) - 1
    suma_areas = 0.0
    print("  +" + "-" * 14 + "+" + "-" * 22 + "+" + "-" * 32 + "+" + "-" * 18 + "+")
    print(f"  | {'Subintervalo':^12} | {'Tramo [x_{i-1}, x_i]':^20} | {'Formula (h/2)*(y_{i-1} + y_i)':^30} | {'Area Parcial':^16} |")
    print("  +" + "-" * 14 + "+" + "-" * 22 + "+" + "-" * 32 + "+" + "-" * 18 + "+")
    for i in range(n):
        area_i = (h / 2.0) * (y[i] + y[i + 1])
        suma_areas += area_i
        tramo_str = f"[{x[i]:.4g}, {x[i+1]:.4g}]"
        form_str = f"({h:.4g}/2)*({y[i]:.5f} + {y[i+1]:.5f})"
        print(f"  | {i+1:^12} | {tramo_str:^20} | {form_str:^30} | {area_i:^16.6f} |")
    print("  +" + "-" * 14 + "+" + "-" * 22 + "+" + "-" * 32 + "+" + "-" * 18 + "+")
    print(f"  | {'SUMA TOTAL DE AREAS PARCIALES':^68} | {suma_areas:^16.6f} |")
    print("  +" + "-" * 70 + "+" + "-" * 18 + "+")


def verificar_equidistancia(x, tol=1e-7):
    """
    Verifica que los nodos x esten igualmente espaciados (paso h constante)
    y retorna el paso h y el numero de subintervalos n.
    """
    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos (1 subintervalo) para integrar.")
    if len(x) != len(set(x)):
        raise ValueError("Los valores de la variable independiente x no deben repetirse.")

    n = len(x) - 1
    h = (x[-1] - x[0]) / n

    for i in range(n):
        hi = x[i + 1] - x[i]
        if abs(hi - h) > tol:
            raise ValueError(
                f"Los puntos de x no son equidistantes: x[{i+1}] - x[{i}] = {hi:.6g} != h = {h:.6g}."
            )
    return h, n


def ingresar_datos_interactivo(x_default, y_default, nombre_x="x", nombre_y="y"):
    """
    Permite al usuario elegir entre utilizar los datos por defecto de la guia
    o ingresar una nueva tabla de valores (x, y) por teclado con validacion.
    """
    x_str = ", ".join(str(v) for v in x_default)
    y_str = ", ".join(str(v) for v in y_default)

    print(f"\n  Datos predeterminados de la Guia:")
    print(f"    {nombre_x} = [{x_str}]")
    print(f"    {nombre_y} = [{y_str}]")

    opcion = input("\n  ¿Desea utilizar los datos por defecto de la guia? [S/n]: ").strip().lower()
    if opcion in ("", "s", "si", "y", "yes", "1"):
        return [float(v) for v in x_default], [float(v) for v in y_default]

    print("\n  --- INGRESO MANUAL DE TABLA DE VALORES ---")
    while True:
        try:
            cant_str = input("  Ingrese la cantidad total de puntos N (minimo 2): ").strip()
            cant = int(cant_str)
            if cant < 2:
                print("  [!] Error: Debe ingresar al menos 2 puntos.")
                continue

            print(f"  Ingrese los {cant} valores de '{nombre_x}' separados por espacio o coma:")
            entrada_x = input(f"    {nombre_x}: ").replace(",", " ").split()
            if len(entrada_x) != cant:
                print(f"  [!] Error: Se esperaban {cant} valores para '{nombre_x}' y se ingresaron {len(entrada_x)}.")
                continue
            x_usr = [float(val) for val in entrada_x]

            # Validar paso constante h
            verificar_equidistancia(x_usr)

            print(f"  Ingrese los {cant} valores de '{nombre_y}' separados por espacio o coma:")
            entrada_y = input(f"    {nombre_y}: ").replace(",", " ").split()
            if len(entrada_y) != cant:
                print(f"  [!] Error: Se esperaban {cant} valores para '{nombre_y}' y se ingresaron {len(entrada_y)}.")
                continue
            y_usr = [float(val) for val in entrada_y]

            return x_usr, y_usr
        except ValueError as err:
            print(f"  [!] Entrada no valida: {err}. Intente nuevamente.\n")


# =============================================================================
# METODOS NUMERICOS DE INTEGRACION (MODULARIZADOS)
# =============================================================================

def metodo_trapecios(x, y, verbose=True):
    """
    Calcula la integral definida mediante el Metodo Compuesto de los Trapecios:
        I_T = (h / 2) * [ E + 2 * (I + P) ]
            = (h / 2) * [ (y_0 + y_n) + 2 * sum_{i=1}^{n-1} y_i ]

    Parametros:
        x       : lista o array de abscisas equidistantes
        y       : lista o array de ordenadas f(x)
        verbose : si es True, imprime el desarrollo paso a paso (E, I, P, formula)

    Retorna:
        dict con 'integral', 'h', 'n', 'E', 'I', 'P', 'R', 'tiempo_s'
    """
    if len(x) != len(y):
        raise ValueError("Los vectores 'x' e 'y' deben tener la misma longitud.")

    inicio = time.perf_counter()

    h, n = verificar_equidistancia(x)

    # Clasificacion de ordenadas segun nomenclatura FaCENA - UNNE
    E = y[0] + y[-1]
    impares = [y[i] for i in range(1, n) if i % 2 != 0]
    pares = [y[i] for i in range(1, n) if i % 2 == 0]
    I = sum(impares)
    P = sum(pares)
    R = I + P  # Suma de todas las ordenadas interiores

    integral = (h / 2.0) * (E + 2.0 * R)

    fin = time.perf_counter()
    tiempo_s = fin - inicio

    if verbose:
        print(f"\n  [Metodo de los Trapecios en x in [{x[0]:.4g}, {x[-1]:.4g}] con n = {n} subintervalo(s)]")
        print(f"    * Paso de integracion : h = (x_{n} - x_0) / n = ({x[-1]:.6g} - {x[0]:.6g}) / {n} = {h:.6f}")
        print(f"    * Suma de Extremos (E): y_0 + y_{n} = {y[0]:.6f} + {y[-1]:.6f} = {E:.6f}")
        if n > 1:
            imp_str = " + ".join(f"{v:.6f}" for v in impares) if impares else "0.000000"
            par_str = " + ".join(f"{v:.6f}" for v in pares) if pares else "0.000000"
            print(f"    * Suma de Impares  (I): {imp_str} = {I:.6f}")
            print(f"    * Suma de Pares    (P): {par_str} = {P:.6f}")
            print(f"    * Resto Interior   (R): I + P = {I:.6f} + {P:.6f} = {R:.6f}")
        else:
            print(f"    * Puntos interiores   : No posee (Trapecio Simple, R = 0)")
        print(f"    * Formula Teorica     : I_T = (h / 2) * [ E + 2 * (I + P) ]")
        print(f"    * Reemplazo en Formula: I_T = ({h:.6f} / 2) * [ {E:.6f} + 2 * ({R:.6f}) ]")
        print(f"    * Calculo Intermedio  : I_T = {h / 2.0:.6f} * [ {E + 2.0 * R:.6f} ]")
        print(f"    * RESULTADO           : I_T = {integral:.6f}")
        print(f"    * Tiempo de ejecucion : {tiempo_s:.8f} s ({tiempo_s * 1000:.4f} ms)")

    return {
        "metodo": "Metodo de los Trapecios",
        "integral": integral,
        "h": h,
        "n": n,
        "E": E,
        "I": I,
        "P": P,
        "R": R,
        "tiempo_s": tiempo_s,
    }


def metodo_simpson_1_3(x, y, verbose=True):
    """
    Funcion modular para el Metodo Compuesto de Simpson 1/3:
        I_{S1/3} = (h / 3) * [ E + 4 * I + 2 * P ]
    Requiere que la cantidad de subintervalos n = len(x) - 1 sea PAR (n >= 2).
    """
    if len(x) != len(y):
        raise ValueError("Los vectores 'x' e 'y' deben tener la misma longitud.")

    inicio = time.perf_counter()

    h, n = verificar_equidistancia(x)
    if n < 2 or n % 2 != 0:
        raise ValueError(
            f"El Metodo de Simpson 1/3 requiere un numero PAR de subintervalos (n >= 2). Se recibio n = {n}."
        )

    E = y[0] + y[-1]
    impares = [y[i] for i in range(1, n) if i % 2 != 0]
    pares = [y[i] for i in range(1, n) if i % 2 == 0]
    I = sum(impares)
    P = sum(pares)

    integral = (h / 3.0) * (E + 4.0 * I + 2.0 * P)

    fin = time.perf_counter()
    tiempo_s = fin - inicio

    if verbose:
        imp_str = " + ".join(f"{v:.6f}" for v in impares) if impares else "0.000000"
        par_str = " + ".join(f"{v:.6f}" for v in pares) if pares else "0.000000"
        print(f"\n  [Metodo de Simpson 1/3 en x in [{x[0]:.4g}, {x[-1]:.4g}] con n = {n} subintervalos]")
        print(f"    * Paso de integracion : h = ({x[-1]:.6g} - {x[0]:.6g}) / {n} = {h:.6f}")
        print(f"    * Suma de Extremos (E): y_0 + y_{n} = {y[0]:.6f} + {y[-1]:.6f} = {E:.6f}")
        print(f"    * Suma de Impares  (I): {imp_str} = {I:.6f}")
        print(f"    * Suma de Pares    (P): {par_str} = {P:.6f}")
        print(f"    * Formula Teorica     : I_(S1/3) = (h / 3) * [ E + 4 * I + 2 * P ]")
        print(f"    * Reemplazo en Formula: I_(S1/3) = ({h:.6f} / 3) * [ {E:.6f} + 4*({I:.6f}) + 2*({P:.6f}) ]")
        print(f"    * RESULTADO           : I_(S1/3) = {integral:.6f}")
        print(f"    * Tiempo de ejecucion : {tiempo_s:.8f} s ({tiempo_s * 1000:.4f} ms)")

    return {
        "metodo": "Metodo de Simpson 1/3",
        "integral": integral,
        "h": h,
        "n": n,
        "E": E,
        "I": I,
        "P": P,
        "tiempo_s": tiempo_s,
    }


def metodo_simpson_3_8(x, y, verbose=True):
    """
    Funcion modular para el Metodo de Simpson 3/8:
        I_{S3/8} = (3 * h / 8) * [ E + 3 * S_no_mult3 + 2 * S_mult3 ]
    Requiere que la cantidad de subintervalos n = len(x) - 1 sea MULTIPLO DE 3 (n >= 3).
    """
    if len(x) != len(y):
        raise ValueError("Los vectores 'x' e 'y' deben tener la misma longitud.")

    inicio = time.perf_counter()

    h, n = verificar_equidistancia(x)
    if n < 3 or n % 3 != 0:
        raise ValueError(
            f"El Metodo de Simpson 3/8 requiere que n sea multiplo de 3 (n >= 3). Se recibio n = {n}."
        )

    E = y[0] + y[-1]
    no_mult3 = [y[i] for i in range(1, n) if i % 3 != 0]
    mult3 = [y[i] for i in range(1, n) if i % 3 == 0]
    S_no_mult3 = sum(no_mult3)
    S_mult3 = sum(mult3)

    integral = (3.0 * h / 8.0) * (E + 3.0 * S_no_mult3 + 2.0 * S_mult3)

    fin = time.perf_counter()
    tiempo_s = fin - inicio

    if verbose:
        nm3_str = " + ".join(f"{v:.6f}" for v in no_mult3) if no_mult3 else "0.000000"
        print(f"\n  [Metodo de Simpson 3/8 en x in [{x[0]:.4g}, {x[-1]:.4g}] con n = {n} subintervalos]")
        print(f"    * Paso de integracion     : h = ({x[-1]:.6g} - {x[0]:.6g}) / {n} = {h:.6f}")
        print(f"    * Suma de Extremos (E)    : y_0 + y_{n} = {y[0]:.6f} + {y[-1]:.6f} = {E:.6f}")
        print(f"    * Internos no mult. de 3  : {nm3_str} = {S_no_mult3:.6f}")
        print(f"    * RESULTADO               : I_(S3/8) = {integral:.6f}")
        print(f"    * Tiempo de ejecucion     : {tiempo_s:.8f} s ({tiempo_s * 1000:.4f} ms)")

    return {
        "metodo": "Metodo de Simpson 3/8",
        "integral": integral,
        "h": h,
        "n": n,
        "E": E,
        "S_no_mult3": S_no_mult3,
        "S_mult3": S_mult3,
        "tiempo_s": tiempo_s,
    }


# =============================================================================
# RESOLUCION DEL EJERCICIO 2 (EXCLUSIVAMENTE METODO DE LOS TRAPECIOS)
# =============================================================================

def resolver_ejercicio_2():
    """
    Resuelve el Ejercicio 2 de la Guia de Trabajos Practicos N° 6:
    Tabla de valores:
        x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.0, 0.15, 0.3, 0.55, 0.73, 0.81]
    Calcula la integral definida utilizando UNICAMENTE el Metodo de los Trapecios
    (ya que n = 5 subintervalos es impar y no aplica Simpson 1/3).
    """
    imprimir_banner("EJERCICIO 2: INTEGRACION NUMERICA POR EL METODO DE LOS TRAPECIOS")

    print("  Consigna:")
    print("    Dada la tabla de valores (x, y), calcular la integral definida en el")
    print("    intervalo [x_0, x_n] aplicando el Metodo de los Trapecios.")

    x_def = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    y_def = [0.0, 0.15, 0.3, 0.55, 0.73, 0.81]

    x, y = ingresar_datos_interactivo(x_def, y_def, nombre_x="x", nombre_y="y")
    h, n = verificar_equidistancia(x)

    mostrar_tabla_datos(x, y, nombre_x="x", nombre_y="y")

    imprimir_subtitulo(f"1. JUSTIFICACION DEL METODO SELECCIONADO (n = {n}, h = {h:.4g})")
    print(f"  * Cantidad de puntos de la tabla : N = {len(x)}")
    print(f"  * Cantidad de subintervalos      : n = N - 1 = {n}")
    print(f"  * Amplitud de paso constante     : h = ({x[-1]} - {x[0]}) / {n} = {h:.6f}")
    print(f"  * Seleccion del metodo           : Al tener n = {n} subintervalos (impar), se aplica")
    print("    el Metodo Compuesto de los Trapecios sobre todo el intervalo [x_0, x_n].")

    imprimir_subtitulo(f"2. CALCULO PASO A PASO POR EL METODO DE LOS TRAPECIOS (n = {n}, h = {h:.4g})")
    res = metodo_trapecios(x, y, verbose=True)

    imprimir_subtitulo("3. VERIFICACION: SUMA DE AREAS DE CADA SUB-TRAPECIO INDIVIDUAL")
    mostrar_desglose_subtrapecios(x, y, h)

    print("\n  Conclusion del Ejercicio 2:")
    print(f"    * Integral definida aproximada en x in [{x[0]:.4g}, {x[-1]:.4g}] por Trapecios: I_T = {res['integral']:.6f}")
    print(f"    * Tiempo de procesamiento medido: {res['tiempo_s']:.8f} segundos ({res['tiempo_s']*1000:.4f} ms).")


# =============================================================================
# RESOLUCION DEL EJERCICIO 3 (EXCLUSIVAMENTE METODO DE LOS TRAPECIOS)
# =============================================================================

def resolver_ejercicio_3():
    """
    Resuelve el Ejercicio 3 de la Guia de Trabajos Practicos N° 6:
    Utilizando la tabla de valores del Ejercicio 1:
        x = [12, 13, 14, 15, 16, 17]
        y = [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151]
    Calcula la integral definida aplicando EXCLUSIVAMENTE el Metodo de los Trapecios.
    """
    imprimir_banner("EJERCICIO 3: INTEGRACION NUMERICA POR EL METODO DE LOS TRAPECIOS")

    print("  Consigna:")
    print("    Utilizando la tabla de valores (x, y) del Ejercicio 1, calcular la integral")
    print("    definida aplicando EXCLUSIVAMENTE el Metodo de los Trapecios.")

    x_def = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    y_def = [0.96262, 0.94718, 0.93108, 0.92221, 0.91181, 0.90151]

    x, y = ingresar_datos_interactivo(x_def, y_def, nombre_x="x", nombre_y="y")
    h, n = verificar_equidistancia(x)

    mostrar_tabla_datos(x, y, nombre_x="x", nombre_y="y")

    imprimir_subtitulo(f"1. CALCULO PASO A PASO POR EL METODO DE LOS TRAPECIOS (n = {n}, h = {h:.4g})")
    res = metodo_trapecios(x, y, verbose=True)

    imprimir_subtitulo("2. VERIFICACION: SUMA DE AREAS DE CADA SUB-TRAPECIO INDIVIDUAL")
    mostrar_desglose_subtrapecios(x, y, h)

    print("\n  Conclusion del Ejercicio 3:")
    print(f"    * Integral definida aproximada en x in [{x[0]:.4g}, {x[-1]:.4g}] por Trapecios: I_T = {res['integral']:.6f}")
    print(f"    * Tiempo de procesamiento medido: {res['tiempo_s']:.8f} segundos ({res['tiempo_s']*1000:.4f} ms).")


# =============================================================================
# MENU PRINCIPAL INTERACTIVO
# =============================================================================

def menu_principal():
    """Menu interactivo principal del Trabajo Practico N° 6."""
    while True:
        imprimir_banner(
            "FaCENA - UNNE | TRABAJO PRACTICO N° 6: METODOS DE INTEGRACION NUMERICA"
        )
        print("  1. Ejecutar Ejercicio 2 (Tabla x vs y en [0, 0.5] - Metodo de los Trapecios)")
        print("  2. Ejecutar Ejercicio 3 (Tabla Ej. 1 x vs y en [12, 17] - Metodo de los Trapecios)")
        print("  3. Ejecutar AMBOS Ejercicios (2 y 3) en forma secuencial")
        print("  4. Salir")
        print("=" * 92)

        try:
            opcion = input("  Seleccione una opcion (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Ejecucion finalizada por el usuario.")
            break

        if opcion == "1":
            resolver_ejercicio_2()
        elif opcion == "2":
            resolver_ejercicio_3()
        elif opcion == "3":
            resolver_ejercicio_2()
            resolver_ejercicio_3()
        elif opcion == "4":
            print("\n  ¡Gracias por utilizar el programa de Integracion Numerica (TP N° 6 - FaCENA UNNE)!\n")
            break
        else:
            print("\n  [!] Opcion no valida. Por favor, ingrese un numero entre 1 y 4.")


if __name__ == "__main__":
    menu_principal()
