"""
===============================================================================
METODOS COMPUTACIONALES - EVALUACION Y COMPARACION DE METODOS NUMERICOS
EJERCICIO 1: CALCULO DE LA PROFUNDIDAD EN UN TANQUE ESFERICO
===============================================================================
Este script resuelve el Ejercicio 1 (Tanque Esférico) aplicando y comparando
dos métodos cerrados de resolución de ecuaciones no lineales:
  1. Método de Intervalo Medio (Bisección)
  2. Método de Interpolación Lineal (Regula Falsi / Falsa Posición)

Ecuación física del volumen del líquido:
  V = pi * h^2 * (3R - h) / 3
Para un radio R = 3.0 m y un volumen de agua V = 30.0 m^3:
  f(h) = pi * (h**2) * ((9.0 - h) / 3.0) - 30.0 = 0

Flujo de ejecución:
  1. Método de Tanteo (h entre 0 y 10 con paso de 0.5) para ubicar raíces.
  2. Selección de intervalo por parte del usuario mediante consola.
  3. Método de Intervalo Medio con cálculo previo de iteraciones estimadas.
  4. Método de Interpolación Lineal con la fórmula de la cátedra.
  5. Gráfica comparativa con Matplotlib mostrando la convergencia de ambos métodos.
===============================================================================
"""

import math
import sys
import time

# Configuración de codificación UTF-8 para consola en Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    # pyrefly: ignore [missing-import]
    import matplotlib.pyplot as plt
    # pyrefly: ignore [missing-import]
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# FUNCION MATEMATICA DEL EJERCICIO 1: TANQUE ESFERICO
# =============================================================================
def f_tanque(h):
    """
    Función matemática correspondiente al Ejercicio 1:
    f(h) = pi * (h**2) * ((9.0 - h) / 3.0) - 30.0
    """
    return math.pi * (h ** 2) * ((9.0 - h) / 3.0) - 30.0


# =============================================================================
# 1. METODO DE TANTEO
# =============================================================================
def metodo_tanteo(f, inicio=0.0, fin=10.0, incremento=0.5):
    """
    Evalua f(h) desde inicio hasta fin con un incremento especificado
    para hallar intervalos donde f(h) cambia de signo (Teorema de Bolzano).
    """
    intervalos = []
    x = inicio

    while x < fin:
        siguiente = round(min(x + incremento, fin), 6)
        fx = f(x)
        fsiguiente = f(siguiente)

        if fx == 0.0:
            intervalos.append((x, x))
        elif fx * fsiguiente < 0:
            intervalos.append((x, siguiente))

        x = siguiente

    return intervalos


# =============================================================================
# FORMATO VISUAL DE TABLAS
# =============================================================================
ANCHO_TABLA = 98

def imprimir_encabezado_tabla():
    """
    Imprime el encabezado estricto con el formato visual solicitado:
    | Iteración | x (Raíz) | f(x) | Intervalo actual [a, b] | Error |
    """
    print("=" * ANCHO_TABLA)
    encabezado = (
        f"| {'Iteración':^11} | {'x (Raíz)':^14} | {'f(x)':^16} "
        f"| {'Intervalo actual [a, b]':^31} | {'Error':^14} |"
    )
    print(encabezado)
    print("=" * ANCHO_TABLA)


def imprimir_fila_tabla(k, c, fc, a, b, error):
    """
    Imprime una fila de datos manteniendo la alineacion estricta de columnas.
    """
    intervalo_str = f"[{a:.6f}, {b:.6f}]"
    fila = (
        f"| {k:^11} | {c:^14.6f} | {fc:^16.6e} "
        f"| {intervalo_str:^31} | {error:^14.6e} |"
    )
    print(fila)


# =============================================================================
# 3. METODO DE INTERVALO MEDIO (BISECCION)
# =============================================================================
def metodo_intervalo_medio(f, a_inicial, b_inicial, tol=0.001, max_iter=100):
    """
    Aplica el Metodo de Intervalo Medio (Biseccion) en el intervalo [a, b].
    Calcula antes del ciclo la cantidad de iteraciones estimadas teoricas:
      n >= (math.log(b - a) - math.log(tol)) / math.log(2)
    """
    a = a_inicial
    b = b_inicial
    fa = f(a)
    fb = f(b)

    # Verificacion previa de Bolzano
    if fa * fb > 0:
        print(f"\n[ADVERTENCIA] No hay cambio de signo en [{a}, {b}].")
        print(f"  f({a}) = {fa:.6f}, f({b}) = {fb:.6f}")
        return None, 0, [], 0.0

    print("\n" + "=" * ANCHO_TABLA)
    print("                   EJECUCION: METODO DE INTERVALO MEDIO (BISECCION)")
    print("=" * ANCHO_TABLA)

    # Estimacion previa de iteraciones antes de ingresar al ciclo
    diff_intervalo = b - a
    n_formula = (math.log(diff_intervalo) - math.log(tol)) / math.log(2)
    n_estimadas = max(1, math.ceil(n_formula))

    print(f"Iteraciones estimadas teoricas con la formula:")
    print(f"  n >= (math.log(b - a) - math.log(E)) / math.log(2)")
    print(f"  n >= (math.log({diff_intervalo:.6f}) - math.log({tol:.6f})) / math.log(2)")
    print(f"  n >= ({math.log(diff_intervalo):.6f} - ({math.log(tol):.6f})) / {math.log(2):.6f} = {n_formula:.4f}")
    print(f"  --> Cantidad minima de iteraciones estimadas: n >= {n_estimadas}\n")

    historial = []
    k = 0
    c = (a + b) / 2.0
    error = (b - a) / 2.0

    imprimir_encabezado_tabla()

    inicio_tiempo = time.perf_counter()

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
            'fc': fc,
            'error': error
        })

        imprimir_fila_tabla(k, c, fc, a, b, error)

        # Condicion de parada: tolerancia alcanzada o raiz exacta
        if error < tol or abs(fc) == 0.0:
            break

        # Reasignacion de subintervalo segun el Teorema del Valor Medio / Bolzano
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    fin_tiempo = time.perf_counter()
    tiempo_ejec = fin_tiempo - inicio_tiempo
    raiz = c
    error_final = error

    print("=" * ANCHO_TABLA)
    print(f"---> REPORTE FINAL: METODO DE INTERVALO MEDIO <---")
    print(f"  * Raiz hallada: {raiz:.6f}")
    print(f"  * Valor de f(raiz): {f(raiz):.6e}")
    print(f"  * Cantidad de iteraciones realizadas: {k}")
    print(f"  * Cota de error final alcanzada: {error_final:.6e}")
    print(f"  * Tiempo de ejecucion: {tiempo_ejec:.6f} segundos\n")

    return raiz, k, historial, tiempo_ejec


# =============================================================================
# 4. METODO DE INTERPOLACION LINEAL (REGULA FALSI)
# =============================================================================
def metodo_interpolacion_lineal(f, a_inicial, b_inicial, tol=0.001, max_iter=100):
    """
    Aplica el Metodo de Interpolacion Lineal (Regula Falsi) en el intervalo [a, b].
    Formula exacta de la catedra:
      c = (a * f(b) - b * f(a)) / (f(b) - f(a))
    Reasignacion de limites:
      si f(a) * f(c) < 0 -> b = c
      sino             -> a = c
    """
    a = a_inicial
    b = b_inicial
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        print(f"\n[ADVERTENCIA] No hay cambio de signo en [{a}, {b}].")
        print(f"  f({a}) = {fa:.6f}, f({b}) = {fb:.6f}")
        return None, 0, [], 0.0

    print("\n" + "=" * ANCHO_TABLA)
    print("              EJECUCION: METODO DE INTERPOLACION LINEAL (REGULA FALSI)")
    print("=" * ANCHO_TABLA)
    print(f"Formula de la catedra: c = (a * f(b) - b * f(a)) / (f(b) - f(a))")
    print(f"Tolerancia establecida: E = {tol}\n")

    historial = []
    k = 0
    c_anterior = None

    imprimir_encabezado_tabla()

    inicio_tiempo = time.perf_counter()

    while k < max_iter:
        k += 1

        # Formula exacta de la catedra para Interpolacion Lineal
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        # Calculo del error
        if c_anterior is not None:
            error = abs(c - c_anterior)
        else:
            error = abs(b - a)

        historial.append({
            'iter': k,
            'a': a,
            'b': b,
            'c': c,
            'fc': fc,
            'error': error
        })

        imprimir_fila_tabla(k, c, fc, a, b, error)

        # Criterio de parada
        if (c_anterior is not None and error < tol) or abs(fc) == 0.0:
            break

        c_anterior = c

        # Reasignacion de limites segun f(a) * f(c) < 0
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    fin_tiempo = time.perf_counter()
    tiempo_ejec = fin_tiempo - inicio_tiempo
    raiz = c
    error_final = error

    print("=" * ANCHO_TABLA)
    print(f"---> REPORTE FINAL: METODO DE INTERPOLACION LINEAL <---")
    print(f"  * Raiz hallada: {raiz:.6f}")
    print(f"  * Valor de f(raiz): {f(raiz):.6e}")
    print(f"  * Cantidad de iteraciones realizadas: {k}")
    print(f"  * Cota de error final alcanzada: {error_final:.6e}")
    print(f"  * Tiempo de ejecucion: {tiempo_ejec:.6f} segundos\n")

    return raiz, k, historial, tiempo_ejec


# =============================================================================
# 5. GRAFICA COMPARATIVA (MATPLOTLIB)
# =============================================================================
def graficar_comparacion(f, a_orig, b_orig, raiz_im, raiz_il, iters_im, iters_il):
    """
    Genera la grafica comparativa con Matplotlib:
      - Curva de la funcion f(h).
      - Recta de referencia y = 0.
      - Limites del intervalo inicial [a, b].
      - Punto de la raiz hallada por Intervalo Medio.
      - Punto de la raiz hallada por Interpolacion Lineal.
    """
    if not HAS_MATPLOTLIB:
        print("\n[Nota: matplotlib o numpy no estan disponibles para graficar.]")
        return

    margen = max(abs(b_orig - a_orig) * 0.8, 0.4)
    x_min = max(0.0, a_orig - margen)
    x_max = min(10.0, b_orig + margen)

    h_vals = np.linspace(x_min, x_max, 500)
    f_vals = [f(h) for h in h_vals]

    plt.figure(figsize=(10, 6))

    # Curva f(h)
    plt.plot(
        h_vals,
        f_vals,
        label=r'$f(h) = \pi \cdot h^2 \cdot \frac{9-h}{3} - 30$',
        color='#1f77b4',
        linewidth=2.2
    )

    # Linea de referencia y = 0
    plt.axhline(0, color='black', linestyle='--', linewidth=1.0, label='y = 0 (Eje neutro)')

    # Limites del intervalo inicial
    plt.axvline(a_orig, color='#2ca02c', linestyle=':', linewidth=1.5, label=f'Limite inferior a = {a_orig:.2f}')
    plt.axvline(b_orig, color='#9467bd', linestyle=':', linewidth=1.5, label=f'Limite superior b = {b_orig:.2f}')

    # Puntos de las raices
    f_im = f(raiz_im)
    f_il = f(raiz_il)

    plt.plot(
        raiz_im,
        f_im,
        marker='o',
        markersize=9,
        color='#d62728',
        linestyle='None',
        label=f'Intervalo Medio: h = {raiz_im:.6f} ({iters_im} iters)'
    )

    plt.plot(
        raiz_il,
        f_il,
        marker='s',
        markersize=7,
        color='#ff7f0e',
        linestyle='None',
        label=f'Interpolacion Lineal: h = {raiz_il:.6f} ({iters_il} iters)'
    )

    # Configuracion estetica de la grafica
    plt.title(
        'Ejercicio 1: Tanque Esferico - Comparacion de Metodos Numericos\n'
        r'Resolucion de $f(h) = 0$ con Tolerancia $E = 0.001$',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    plt.xlabel('Profundidad del liquido en el tanque h [m]', fontsize=11)
    plt.ylabel('Volumen remanente f(h) [m^3]', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.tight_layout()

    print("\nVisualizando grafica comparativa...")
    plt.show()


# =============================================================================
# CUADRO COMPARATIVO CONSOLIDADO EN CONSOLA
# =============================================================================
def mostrar_cuadro_comparativo(raiz_im, iters_im, t_im, raiz_il, iters_il, t_il):
    """
    Muestra en consola un resumen visual comparativo entre ambos metodos.
    """
    f_im = f_tanque(raiz_im)
    f_il = f_tanque(raiz_il)

    print("\n" + "=" * ANCHO_TABLA)
    print("                    CUADRO COMPARATIVO RESUMIDO ENTRE METODOS")
    print("=" * ANCHO_TABLA)
    encabezado = (
        f"| {'Parametro de Comparacion':<30} "
        f"| {'Intervalo Medio (Biseccion)':<30} "
        f"| {'Interpolacion Lineal':<30} |"
    )
    print(encabezado)
    print("-" * ANCHO_TABLA)

    fila_raiz = f"| {'Raiz hallada h [m]':<30} | {f'{raiz_im:.6f}':<30} | {f'{raiz_il:.6f}':<30} |"
    fila_f = f"| {'Valor f(h) en la raiz':<30} | {f'{f_im:.6e}':<30} | {f'{f_il:.6e}':<30} |"
    fila_iter = f"| {'Iteraciones requeridas':<30} | {f'{iters_im}':<30} | {f'{iters_il}':<30} |"
    fila_tiempo = f"| {'Tiempo de ejecucion [s]':<30} | {f'{t_im:.6f} s':<30} | {f'{t_il:.6f} s':<30} |"

    print(fila_raiz)
    print(fila_f)
    print(fila_iter)
    print(fila_tiempo)
    print("=" * ANCHO_TABLA)

    print("\n---> CONCLUSIONES TECNICAS <---")
    print(f"1. Ambos metodos convergen hacia la raiz aproximada en el entorno de h ~ {raiz_il:.4f} m.")
    print(f"2. Eficiencia: El metodo de Interpolacion Lineal requirio {iters_il} iteraciones frente a")
    print(f"   las {iters_im} iteraciones del Metodo de Intervalo Medio, demostrando una mayor velocidad")
    print(f"   de aproximacion inicial al aprovechar las pendientes y valores de la funcion.")
    print("=" * ANCHO_TABLA + "\n")


# =============================================================================
# FUNCION PRINCIPAL (FLUJO DE EJECUCION ESTRICTO)
# =============================================================================
def main():
    print("\n" + "#" * ANCHO_TABLA)
    print("   METODOS COMPUTACIONALES - EJERCICIO 1: EVALUACION Y COMPARACION DE METODOS")
    print("             (Intervalo Medio vs. Interpolacion Lineal - Tanque Esferico)")
    print("#" * ANCHO_TABLA)
    print("Ecuacion a resolver:")
    print("  f(h) = pi * (h**2) * ((9.0 - h) / 3.0) - 30.0 = 0")
    print("Tolerancia fijada para los metodos: E = 0.001\n")

    # -------------------------------------------------------------------------
    # PASO 1: METODO DE TANTEO
    # -------------------------------------------------------------------------
    print("=" * ANCHO_TABLA)
    print("1. METODO DE TANTEO (BUSQUEDA DE INTERVALOS CON CAMBIO DE SIGNO)")
    print("=" * ANCHO_TABLA)
    print("Evaluando f(h) desde h = 0.0 hasta h = 10.0 con incremento de 0.5...")

    intervalos = metodo_tanteo(f_tanque, inicio=0.0, fin=10.0, incremento=0.5)

    if not intervalos:
        print("\n[ERROR] No se encontraron intervalos con cambio de signo en el rango evaluado.")
        return

    print(f"\nSe encontraron {len(intervalos)} intervalo(s) con cambio de signo:")
    for i, (a_int, b_int) in enumerate(intervalos, start=1):
        print(f"  {i}. [{a_int:.1f}, {b_int:.1f}]   ->   f({a_int:.1f}) = {f_tanque(a_int):.4f}, f({b_int:.1f}) = {f_tanque(b_int):.4f}")

    # -------------------------------------------------------------------------
    # PASO 2: SELECCION DE INTERVALO
    # -------------------------------------------------------------------------
    print("\n" + "=" * ANCHO_TABLA)
    print("2. SELECCION DE INTERVALO POR EL USUARIO")
    print("=" * ANCHO_TABLA)

    while True:
        try:
            opcion_usuario = input(f"Ingrese el numero del intervalo sobre el cual desea trabajar (1-{len(intervalos)}): ").strip()
            opcion = int(opcion_usuario)
            if 1 <= opcion <= len(intervalos):
                break
            print(f"Opcion fuera de rango. Por favor ingrese un numero entre 1 y {len(intervalos)}.")
        except ValueError:
            print("Entrada no valida. Debe ingresar un numero entero.")

    a_elegido, b_elegido = intervalos[opcion - 1]
    tol = 0.001

    print(f"\nIntervalo seleccionado para trabajar: [{a_elegido}, {b_elegido}]")
    print(f"Tolerancia asignada: E = {tol}")

    # -------------------------------------------------------------------------
    # PASO 3: METODO DE INTERVALO MEDIO
    # -------------------------------------------------------------------------
    raiz_im, iters_im, hist_im, t_im = metodo_intervalo_medio(
        f_tanque,
        a_inicial=a_elegido,
        b_inicial=b_elegido,
        tol=tol
    )

    # -------------------------------------------------------------------------
    # PASO 4: METODO DE INTERPOLACION LINEAL
    # -------------------------------------------------------------------------
    raiz_il, iters_il, hist_il, t_il = metodo_interpolacion_lineal(
        f_tanque,
        a_inicial=a_elegido,
        b_inicial=b_elegido,
        tol=tol
    )

    # Resumen comparativo en consola
    mostrar_cuadro_comparativo(raiz_im, iters_im, t_im, raiz_il, iters_il, t_il)

    # -------------------------------------------------------------------------
    # PASO 5: GRAFICA COMPARATIVA (MATPLOTLIB)
    # -------------------------------------------------------------------------
    graficar_comparacion(
        f_tanque,
        a_orig=a_elegido,
        b_orig=b_elegido,
        raiz_im=raiz_im,
        raiz_il=raiz_il,
        iters_im=iters_im,
        iters_il=iters_il
    )


if __name__ == "__main__":
    main()
