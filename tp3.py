"""
===============================================================================
METODOS COMPUTACIONALES - LABORATORIO: Guia de Actividades N° 3
Implementacion del Metodo de Intervalo Medio (Biseccion)
===============================================================================
Este script resuelve los tres ejercicios propuestos en el TP N° 3 aplicando 
exclusivamente el Metodo de Intervalo Medio.

Requerimientos cumplidos:
  - Algoritmo completo del Metodo de Intervalo Medio (Biseccion).
  - Reporte de la cantidad de iteraciones necesarias.
  - Medicion del tiempo de ejecucion (cputime/tictoc).
  - Calculo e informe del error cometido y tabla de convergencia.
  - Ingreso de datos por teclado con opcion de valores por defecto del TP.
  - Graficos correspondientes con matplotlib.
  - Leyendas explicativas y conclusiones teorico-practicas para cada punto.
===============================================================================
"""

import math
import time

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# ALGORITMO PRINCIPAL: METODO DE INTERVALO MEDIO (BISECCION)
# =============================================================================

def metodo_intervalo_medio(f, a, b, tol=1e-3, max_iter=100, var_name="x"):
    """
    Implementacion del Metodo de Intervalo Medio (Biseccion).

    Parametros:
        f        : funcion matematica f(x) a evaluar
        a        : limite inferior del intervalo inicial
        b        : limite superior del intervalo inicial
        tol      : cota de error permitida (tolerancia)
        max_iter : cantidad maxima de iteraciones
        var_name : nombre de la variable para leyendas (ej. 'h' o 'x')

    Retorna:
        raiz           : valor aproximado de la raiz
        k              : cantidad de iteraciones realizadas
        historial      : lista de diccionarios con el detalle iteracion a iteracion
        tiempo_ejec    : tiempo de proceso en segundos
    """
    inicio = time.perf_counter()

    fa = f(a)
    fb = f(b)

    # Verificacion de condicion de Bolzano: f(a) * f(b) <= 0
    if fa * fb > 0:
        print(f"\n[ADVERTENCIA] La funcion no cambia de signo en el intervalo [{a}, {b}].")
        print(f"  f({a}) = {fa:.6f}, f({b}) = {fb:.6f}")
        print("  El Metodo de Intervalo Medio no garantiza la convergencia a una raiz en este intervalo.\n")

    historial = []
    k = 0
    c = (a + b) / 2.0

    print("\n" + "=" * 95)
    print(f"{'i':<4} | {'a':<12} | {'b':<12} | {f'c ({var_name}_k)':<14} | {'f(c)':<14} | {'Error (b-a)/2':<14}")
    print("=" * 95)

    while k < max_iter:
        k += 1
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0

        historial.append({
            'iter': k,
            'a': a,
            'b': b,
            'c': c,
            'fa': fa,
            'fb': fb,
            'fc': fc,
            'error': error
        })

        print(f"{k:<4} | {a:<12.6f} | {b:<12.6f} | {c:<14.6f} | {fc:<14.6e} | {error:<14.6e}")

        # Criterio de parada
        if error < tol or abs(fc) == 0.0:
            break

        # Cambio de intervalo segun teorema del valor medio
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    fin = time.perf_counter()
    tiempo_ejec = fin - inicio

    raiz = c
    error_final = (b - a) / 2.0

    print("=" * 95)
    print(f"\n---> RESULTADOS DEL METODO DE INTERVALO MEDIO <---")
    print(f"  * Aproximacion de la raiz ({var_name}): {raiz:.6f}")
    print(f"  * Valor de f({var_name}): {f(raiz):.6e}")
    print(f"  * Cantidad de iteraciones necesarias: {k}")
    print(f"  * Cota de error alcanzada: {error_final:.6e}")
    print(f"  * Tiempo de ejecucion del proceso: {tiempo_ejec:.6f} segundos\n")

    return raiz, k, historial, tiempo_ejec


# =============================================================================
# FUNCIONES AUXILIARES PARA INGRESO DE DATOS
# =============================================================================

def ingresar_float(prompt, default_val):
    """
    Permite ingresar un valor por teclado. Si se presiona Enter, se usa el valor por defecto.
    """
    user_input = input(f"{prompt} [Por defecto = {default_val}]: ").strip()
    if user_input == "":
        return float(default_val)
    try:
        return float(user_input)
    except ValueError:
        print(f"Entrada no valida. Se utilizara el valor por defecto: {default_val}")
        return float(default_val)


# =============================================================================
# EJERCICIO 1: TANQUE ESFERICO
# =============================================================================

def f_tanque(h, R=3.0, V=30.0):
    """
    f(h) = pi * h^2 * (3R - h) / 3 - V = 0
    Para R = 3 y V = 30:
    f(h) = pi * h^2 * (9 - h) / 3 - 30
    """
    return math.pi * (h ** 2) * ((3.0 * R - h) / 3.0) - V


def ejercicio_1():
    print("\n" + "#" * 95)
    print("EJERCICIO 1: CALCULO DE LA PROFUNDIDAD EN UN TANQUE ESFERICO")
    print("#" * 95)
    print("Un tanque esferico de radio R = 3 m debe contener V = 30 m^3 de agua.")
    print("Ecuacion del volumen del liquido: V = pi * h^2 * (3R - h) / 3")
    print("Buscamos la profundidad h en metros tal que f(h) = pi * h^2 * (3R - h) / 3 - 30 = 0\n")

    print("Ingreso de parametros por teclado para el Metodo de Intervalo Medio:")
    a = ingresar_float("  Ingrese el limite inferior a", 0.0)
    b = ingresar_float("  Ingrese el limite superior b", 3.0)
    tol = ingresar_float("  Ingrese la cota de error E", 0.001)

    raiz_h, iters, historial, t_ejec = metodo_intervalo_medio(
        f_tanque, a, b, tol=tol, var_name="h"
    )

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 1 ---")
    print(f"1. Para un tanque esferico de radio R = 3 m y un volumen deseado de V = 30 m^3,")
    print(f"   el agua debe llenarse a una profundidad aproximada de h ~ {raiz_h:.4f} metros.")
    print(f"2. Se requirieron {iters} iteraciones para alcanzar una tolerancia de E < {tol}.")
    print(f"3. El Metodo de Intervalo Medio garantiza la convergencia continua en el intervalo [{a}, {b}]")
    print(f"   dado que f({a}) = {f_tanque(a):.4f} y f({b}) = {f_tanque(b):.4f} tienen signos opuestos.")

    # Grafico
    if HAS_MATPLOTLIB:
        h_vals = np.linspace(0, 6, 400)
        f_vals = [f_tanque(h) for h in h_vals]

        plt.figure(figsize=(9, 5))
        plt.plot(h_vals, f_vals, label=r'$f(h) = \pi h^2 (9-h)/3 - 30$', color='blue', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(a, color='green', linestyle=':', label=f'Limite a = {a}')
        plt.axvline(b, color='orange', linestyle=':', label=f'Limite b = {b}')
        plt.plot(raiz_h, f_tanque(raiz_h), 'ro', markersize=8, label=f'Raiz h ~ {raiz_h:.4f} m')

        plt.title('Ejercicio 1: Profundidad en Tanque Esferico (Metodo de Intervalo Medio)')
        plt.xlabel('Profundidad del agua h [m]')
        plt.ylabel('f(h) [m^3]')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Nota: Instale matplotlib para visualizar los graficos interactivos.]")


# =============================================================================
# EJERCICIO 2: UTILIDAD TOTAL DE IMPRESORAS (PUNTOS DE EQUILIBRIO)
# =============================================================================

def f_utilidad(x):
    """
    P(x) = 8x + 0.3x^2 - 0.0013x^3 - 372
    Punto de equilibrio cuando P(x) = 0
    """
    return 8.0 * x + 0.3 * (x ** 2) - 0.0013 * (x ** 3) - 372.0


def ejercicio_2():
    print("\n" + "#" * 95)
    print("EJERCICIO 2: PUNTOS DE EQUILIBRIO EN VENTA DE IMPRESORAS")
    print("#" * 95)
    print("Funcion de utilidad total: P(x) = 8x + 0.3x^2 - 0.0013x^3 - 372")
    print("Se requiere determinar los dos puntos de equilibrio donde P(x) = 0.\n")

    print("--- PARTE A: PRIMER PUNTO DE EQUILIBRIO ---")
    print("Ingreso de parametros por teclado para la Raiz 1:")
    a1 = ingresar_float("  Ingrese el limite inferior a1", 24.0)
    b1 = ingresar_float("  Ingrese el limite superior b1", 26.0)
    tol1 = ingresar_float("  Ingrese la cota de error E", 0.001)

    raiz1, iters1, hist1, t1 = metodo_intervalo_medio(
        f_utilidad, a1, b1, tol=tol1, var_name="x1"
    )

    print("\n--- PARTE B: SEGUNDO PUNTO DE EQUILIBRIO ---")
    print("Ingreso de parametros por teclado para la Raiz 2:")
    a2 = ingresar_float("  Ingrese el limite inferior a2", 250.0)
    b2 = ingresar_float("  Ingrese el limite superior b2", 252.0)
    tol2 = ingresar_float("  Ingrese la cota de error E", 0.001)

    raiz2, iters2, hist2, t2 = metodo_intervalo_medio(
        f_utilidad, a2, b2, tol=tol2, var_name="x2"
    )

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 2 ---")
    print(f"1. La compania alcanza el punto de equilibrio (utilidad P(x) = $0) en dos niveles de produccion:")
    print(f"   - Primer punto de equilibrio: x1 ~ {raiz1:.4f} impresoras/mes (aprox 25 o 26 impresoras).")
    print(f"   - Segundo punto de equilibrio: x2 ~ {raiz2:.4f} impresoras/mes (aprox 251 impresoras).")
    print(f"2. Para un volumen entre {raiz1:.2f} y {raiz2:.2f} impresoras al mes, la utilidad P(x) es positiva (ganancias).")
    print(f"   Por debajo de {raiz1:.2f} o por encima de {raiz2:.2f} impresoras, la utilidad es negativa (perdidas).")
    print(f"3. Iteraciones realizadas: {iters1} para el primer punto y {iters2} para el segundo con E < {tol1}.")

    # Grafico
    if HAS_MATPLOTLIB:
        x_vals = np.linspace(0, 300, 500)
        p_vals = [f_utilidad(x) for x in x_vals]

        plt.figure(figsize=(10, 5.5))
        plt.plot(x_vals, p_vals, label=r'$P(x) = 8x + 0.3x^2 - 0.0013x^3 - 372$', color='purple', linewidth=2)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='P(x) = 0 (Equilibrio)')
        plt.plot(raiz1, f_utilidad(raiz1), 'ro', markersize=8, label=f'Raiz 1: x ~ {raiz1:.2f}')
        plt.plot(raiz2, f_utilidad(raiz2), 'go', markersize=8, label=f'Raiz 2: x ~ {raiz2:.2f}')

        plt.title('Ejercicio 2: Puntos de Equilibrio de Utilidad (Metodo de Intervalo Medio)')
        plt.xlabel('Unidades producidas y vendidas al mes (x)')
        plt.ylabel('Utilidad total P(x) [pesos]')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Nota: Instale matplotlib para visualizar los graficos interactivos.]")


# =============================================================================
# EJERCICIO 3: UTILIDAD NETA Y PUBLICIDAD
# =============================================================================

def f_publicidad(x):
    """
    Gasto en publicidad A(x) = 200 * ln(400 / (500 - x))
    Utilidad neta R(x) = 5x - A(x)
    Queremos R(x) = 1000 => f(x) = 5x - 200 * ln(400 / (500 - x)) - 1000 = 0
    """
    if 500.0 - x <= 0:
        return float('nan')
    a_cost = 200.0 * math.log(400.0 / (500.0 - x))
    r_net = 5.0 * x - a_cost
    return r_net - 1000.0


def ejercicio_3():
    print("\n" + "#" * 95)
    print("EJERCICIO 3: VENTA DE UNIDADES PARA ALCANZAR UTILIDAD NETA DE $1000")
    print("#" * 95)
    print("Gasto semanal en publicidad: A(x) = 200 * ln(400 / (500 - x))")
    print("Utilidad neta: R(x) = 5x - A(x)")
    print("Se busca el numero de unidades x para tener una utilidad neta de R(x) = $1000 en un entorno de x = 200.")
    print("Ecuacion a resolver: f(x) = 5x - 200 * ln(400 / (500 - x)) - 1000 = 0\n")

    print("Ingreso de parametros por teclado para el Metodo de Intervalo Medio:")
    a = ingresar_float("  Ingrese el limite inferior a", 200.0)
    b = ingresar_float("  Ingrese el limite superior b", 220.0)
    tol = ingresar_float("  Ingrese la cota de error E", 0.0001)

    raiz_x, iters, hist, t_ejec = metodo_intervalo_medio(
        f_publicidad, a, b, tol=tol, var_name="x"
    )

    # Calculo de metricas asociadas
    a_val = 200.0 * math.log(400.0 / (500.0 - raiz_x))
    r_val = 5.0 * raiz_x - a_val

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 3 ---")
    print(f"1. Para alcanzar una utilidad neta de $1000 semanales en un entorno de x = 200,")
    print(f"   la compania debe vender x ~ {raiz_x:.4f} unidades semanales (aprox 214 unidades).")
    print(f"2. En este nivel de ventas:")
    print(f"   - Ingreso bruto (5x): ${5.0 * raiz_x:.2f}")
    print(f"   - Gasto en publicidad A(x): ${a_val:.2f}")
    print(f"   - Utilidad neta resultante R(x): ${r_val:.2f}")
    print(f"3. Se alcanzaron {iters} iteraciones con una precision de E < {tol}.")

    # Grafico
    if HAS_MATPLOTLIB:
        x_vals = np.linspace(150, 250, 400)
        f_vals = [f_publicidad(x) for x in x_vals]
        r_vals = [5.0 * x - 200.0 * math.log(400.0 / (500.0 - x)) for x in x_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: f(x) = R(x) - 1000
        ax1.plot(x_vals, f_vals, color='teal', linewidth=2, label=r'$f(x) = R(x) - 1000$')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.plot(raiz_x, f_publicidad(raiz_x), 'ro', markersize=8, label=f'Raiz x ~ {raiz_x:.2f}')
        ax1.set_title('Raiz de f(x) = 0')
        ax1.set_xlabel('Unidades vendidas (x)')
        ax1.set_ylabel('f(x) [pesos]')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # Subplot 2: R(x) vs $1000
        ax2.plot(x_vals, r_vals, color='darkblue', linewidth=2, label='Utilidad Neta R(x)')
        ax2.axhline(1000, color='red', linestyle='--', label='Meta R(x) = $1000')
        ax2.plot(raiz_x, r_val, 'ro', markersize=8, label=f'x ~ {raiz_x:.2f}, R(x) = ${r_val:.2f}')
        ax2.set_title('Utilidad Neta R(x) vs Meta $1000')
        ax2.set_xlabel('Unidades vendidas (x)')
        ax2.set_ylabel('Pesos ($)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.suptitle('Ejercicio 3: Analisis de Ventas y Utilidad Neta (Intervalo Medio)')
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Nota: Instale matplotlib para visualizar los graficos interactivos.]")


# =============================================================================
# MENU PRINCIPAL INTERACTIVO
# =============================================================================

def menu_principal():
    while True:
        print("\n" + "=" * 95)
        print("     METODOS COMPUTACIONALES - TP N° 3: METODO DE INTERVALO MEDIO (BISECCION)")
        print("=" * 95)
        print("  1. Resolver Ejercicio 1 (Tanque Esferico)")
        print("  2. Resolver Ejercicio 2 (Puntos de Equilibrio - Utilidad de Impresoras)")
        print("  3. Resolver Ejercicio 3 (Gasto en Publicidad y Utilidad Neta)")
        print("  4. Ejecutar TODOS los ejercicios secuencialmente")
        print("  5. Salir")
        print("=" * 95)

        opcion = input("Seleccione una opcion (1-5): ").strip()

        if opcion == '1':
            ejercicio_1()
        elif opcion == '2':
            ejercicio_2()
        elif opcion == '3':
            ejercicio_3()
        elif opcion == '4':
            ejercicio_1()
            ejercicio_2()
            ejercicio_3()
        elif opcion == '5':
            print("\n¡Gracias por utilizar el programa de Metodo de Intervalo Medio!\n")
            break
        else:
            print("\nOpción no valida. Por favor, ingrese un numero del 1 al 5.")


if __name__ == "__main__":
    menu_principal()
