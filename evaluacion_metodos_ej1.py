import math
import sys
import time

# Configuración de codificación UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import sympy as sp
from sympy import Interval
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ANCHO = 85


# =============================================================================
# 1. PARSING DE FUNCION MATEMATICA
# =============================================================================
def parsear_funcion(expr_str):
    """Parsea la expresion matematica y devuelve lambdas numericos para f y f'."""
    trans = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {'e': sp.E, 'E': sp.E, 'pi': sp.pi, 'PI': sp.pi, 'ln': sp.log, 'sen': sp.sin}
    sym_expr = parse_expr(expr_str.strip(), local_dict=local_dict, transformations=trans)

    simbolos = [s for s in sym_expr.free_symbols if s not in (sp.pi, sp.E)]
    var = simbolos[0] if simbolos else sp.Symbol('x')
    df_expr = sp.diff(sym_expr, var)

    f_num = sp.lambdify(var, sym_expr, modules=['numpy', 'math'])
    df_num = sp.lambdify(var, df_expr, modules=['numpy', 'math'])

    f = lambda val: float(np.asarray(f_num(val)).item()) if isinstance(f_num(val), (np.ndarray, list)) else float(f_num(val))
    df = lambda val: float(np.asarray(df_num(val)).item()) if isinstance(df_num(val), (np.ndarray, list)) else float(df_num(val))

    return sym_expr, var, df_expr, f, df


# =============================================================================
# 2. METODO DE TANTEO (RANGO FIJO -15 A 15 CON PASO ELEGIDO)
# =============================================================================
def metodo_tanteo(f, a=-15.0, b=15.0, paso=0.5):
    """Busca intervalos con cambio de signo en [-15, 15] con el paso especificado."""
    intervalos = []
    x = a
    while x < b:
        sig = round(min(x + paso, b), 4)
        try:
            fx, fsig = f(x), f(sig)
            if fx * fsig <= 0:
                intervalos.append((x, sig))
        except Exception:
            pass
        x = sig
    return intervalos


# =============================================================================
# 3. ANALISIS DE CONVERGENCIA PREVIO (SEPARADOS POR METODO)
# =============================================================================
def analisis_convergencia_biseccion(f, a, b, sym_expr, var):
    """Analiza continuidad y Bolzano para Intervalo Medio (Biseccion)."""
    try:
        dominio = continuous_domain(sym_expr, var, Interval(a, b))
        es_continua = (dominio == Interval(a, b))
    except Exception:
        es_continua = True

    txt_cont = (
        f"Se verifica el dominio en [{a:.4f}, {b:.4f}]. Se tiene que la funcion no presenta saltos ni asintotas, por lo tanto, SE CUMPLE."
        if es_continua else
        f"Se verifica el dominio en [{a:.4f}, {b:.4f}]. Se tiene que presenta asintotas o discontinuidades, por lo tanto, NO SE CUMPLE."
    )

    fa, fb = f(a), f(b)
    prod = fa * fb
    bolzano = (prod <= 0) or (fa == 0.0) or (fb == 0.0)
    txt_bolz = (
        f"Se verifica el cambio de signo. Se tiene que f(a)*f(b) = {prod:.4e} <= 0, por lo tanto, SE CUMPLE."
        if bolzano else
        f"Se verifica el cambio de signo. Se tiene que f(a)*f(b) = {prod:.4e} > 0, por lo tanto, NO SE CUMPLE."
    )

    cumple_todo = es_continua and bolzano
    txt_conc = (
        "Al cumplirse las condiciones, se garantiza al menos una raiz. Procediendo con la ejecucion..."
        if cumple_todo else
        "NO SE CUMPLE. Por lo tanto, se saltea la ejecucion del Metodo de Intervalo Medio."
    )

    print(f"""
{"=" * ANCHO}
ANALISIS DE CONVERGENCIA: INTERVALO MEDIO (BISECCION)
{"=" * ANCHO}
- Condicion 1 (Continuidad): {txt_cont}
- Condicion 2 (Teorema de Bolzano): {txt_bolz}
--> Conclusion: {txt_conc}""")
    return cumple_todo


def analisis_convergencia_interpolacion(f, df, a, b, sym_expr, var):
    """Analiza continuidad, Bolzano, rango de f'(x) y monotonia para Interpolacion Lineal."""
    try:
        dominio = continuous_domain(sym_expr, var, Interval(a, b))
        es_continua = (dominio == Interval(a, b))
    except Exception:
        es_continua = True

    txt_cont = (
        "Se verifica el dominio. Se tiene que es continua, por lo tanto, SE CUMPLE."
        if es_continua else
        f"Se verifica el dominio en [{a:.4f}, {b:.4f}]. Se tiene que presenta asintotas o discontinuidades, por lo tanto, NO SE CUMPLE."
    )

    fa, fb = f(a), f(b)
    prod = fa * fb
    bolzano = (prod <= 0) or (fa == 0.0) or (fb == 0.0)
    txt_bolz = (
        f"Se verifica el cambio de signo. Se tiene f(a)*f(b) = {prod:.4e} <= 0, por lo tanto, SE CUMPLE."
        if bolzano else
        f"Se verifica el cambio de signo. Se tiene f(a)*f(b) = {prod:.4e} > 0, por lo tanto, NO SE CUMPLE."
    )

    xs = np.linspace(a, b, 200) if HAS_MATPLOTLIB else [a + i * (b - a) / 200 for i in range(201)]
    dfs = []
    for x in xs:
        try:
            dfs.append(df(x))
        except Exception:
            pass
    df_min, df_max = (min(dfs), max(dfs)) if dfs else (0.0, 0.0)
    no_cero = (df_min * df_max > 0)
    monotona = no_cero

    if no_cero and monotona:
        txt_deriv = f"Se verifica el comportamiento de f'(x). Se tiene que la derivada fluctua entre [{df_min:.4e}, {df_max:.4e}]. Como la derivada no se hace cero y mantiene un signo constante, la funcion es estrictamente monotona, por lo tanto, SE CUMPLE."
    else:
        txt_deriv = f"Se verifica el comportamiento de f'(x). Se tiene que la derivada fluctua entre [{df_min:.4e}, {df_max:.4e}]. Como la derivada alcanza el cero o cambia de signo, hay riesgo de estancamiento, por lo tanto, NO SE CUMPLE."

    cumple_todo = es_continua and bolzano and no_cero and monotona
    txt_conc = (
        "Procediendo con la ejecucion..."
        if cumple_todo else
        "NO SE CUMPLE. Por lo tanto, se saltea la ejecucion de Interpolacion Lineal."
    )

    print(f"""
{"=" * ANCHO}
ANALISIS DE CONVERGENCIA: INTERPOLACION LINEAL
{"=" * ANCHO}
- Condicion 1 (Continuidad): {txt_cont}
- Condicion 2 (Bolzano): {txt_bolz}
- Condicion 3 (Monotonia y Derivada): {txt_deriv}
--> Conclusion: {txt_conc}""")
    return cumple_todo


def verificar_convergencia(f, df, a, b, sym_expr, var):
    """Funcion unificada de compatibilidad que retorna (apto_biseccion, apto_interpolacion)."""
    return analisis_convergencia_biseccion(f, a, b, sym_expr, var), analisis_convergencia_interpolacion(f, df, a, b, sym_expr, var)


# =============================================================================
# FORMATO COMPACTO DE TABLA DE ITERACIONES
# =============================================================================
def imprimir_encabezado():
    print("=" * ANCHO)
    print(f"| {'Iteracion':^10} | {'x (Raiz)':^11} | {'f(x)':^13} | {'Intervalo actual [a, b]':^23} | {'Error':^12} |")
    print("=" * ANCHO)


def imprimir_fila(k, c, fc, a, b, err):
    inter = f"[{a:.5f}, {b:.5f}]"
    print(f"| {k:^10} | {c:^11.6f} | {fc:^13.4e} | {inter:^23} | {err:^12.4e} |")


# =============================================================================
# 4. METODO DE INTERVALO MEDIO (BISECCION)
# =============================================================================
def metodo_intervalo_medio(f, a, b, tol=0.001, max_iter=100):
    fa, fb = f(a), f(b)
    diff = b - a
    n_est = max(1, math.ceil((math.log(diff) - math.log(tol)) / math.log(2))) if diff > 0 and tol > 0 else 1

    print("=" * ANCHO)
    print(f"EJECUCION: METODO DE INTERVALO MEDIO (BISECCION) - Iteraciones estimadas: n >= {n_est}")
    imprimir_encabezado()

    k = 0
    t0 = time.perf_counter()
    while k < max_iter:
        k += 1
        c = (a + b) / 2.0
        fc = f(c)
        err = (b - a) / 2.0
        imprimir_fila(k, c, fc, a, b, err)

        if err < tol or fc == 0.0:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    t1 = time.perf_counter()
    print("=" * ANCHO)
    print(f"--> Biseccion: Raiz = {c:.6f} | f(Raiz) = {fc:.4e} | Iters = {k} | Error = {err:.4e} | Tiempo = {t1 - t0:.6f} s\n")
    return c, k, err, t1 - t0


# =============================================================================
# 5. METODO DE INTERPOLACION LINEAL (REGULA FALSI)
# =============================================================================
def metodo_interpolacion_lineal(f, a, b, tol=0.001, max_iter=100):
    fa, fb = f(a), f(b)

    print("=" * ANCHO)
    print("EJECUCION: METODO DE INTERPOLACION LINEAL")
    imprimir_encabezado()

    k = 0
    c_ant = None
    err = abs(b - a)
    t0 = time.perf_counter()

    while k < max_iter:
        k += 1
        den = fb - fa
        c = (a * fb - b * fa) / den if abs(den) > 1e-15 else (a + b) / 2.0
        fc = f(c)
        err = abs(c - c_ant) if c_ant is not None else abs(b - a)
        imprimir_fila(k, c, fc, a, b, err)

        if (c_ant is not None and err < tol) or fc == 0.0:
            break
        c_ant = c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    t1 = time.perf_counter()
    print("=" * ANCHO)
    print(f"--> Interpolacion: Raiz = {c:.6f} | f(Raiz) = {fc:.4e} | Iters = {k} | Error = {err:.4e} | Tiempo = {t1 - t0:.6f} s\n")
    return c, k, err, t1 - t0


# =============================================================================
# 6. CUADRO COMPARATIVO RESUMIDO (CON MANEJO DE METODOS SALTEADOS)
# =============================================================================
def mostrar_cuadro_comparativo(r_im, it_im, t_im, err_im, r_il, it_il, t_il, err_il, f):
    im_ok = r_im is not None
    il_ok = r_il is not None

    r_im_str = f"{r_im:.6f}" if im_ok else "No ejecutado"
    f_im_str = f"{f(r_im):.4e}" if im_ok else "N/A"
    it_im_str = str(it_im) if im_ok else "N/A"
    err_im_str = f"{err_im:.4e}" if im_ok else "N/A"
    t_im_str = f"{t_im:.6f} s" if im_ok else "N/A"

    r_il_str = f"{r_il:.6f}" if il_ok else "No ejecutado"
    f_il_str = f"{f(r_il):.4e}" if il_ok else "N/A"
    it_il_str = str(it_il) if il_ok else "N/A"
    err_il_str = f"{err_il:.4e}" if il_ok else "N/A"
    t_il_str = f"{t_il:.6f} s" if il_ok else "N/A"

    print("=" * ANCHO)
    print("                      CUADRO COMPARATIVO RESUMIDO ENTRE METODOS")
    print("=" * ANCHO)
    print(f"| {'Parametro':<27} | {'Intervalo Medio (Biseccion)':<27} | {'Interpolacion Lineal':<21} |")
    print("-" * ANCHO)
    print(f"| {'Raiz hallada x*':<27} | {r_im_str:<27} | {r_il_str:<21} |")
    print(f"| {'Valor f(x*)':<27} | {f_im_str:<27} | {f_il_str:<21} |")
    print(f"| {'Iteraciones realizadas':<27} | {it_im_str:<27} | {it_il_str:<21} |")
    print(f"| {'Cota de error final':<27} | {err_im_str:<27} | {err_il_str:<21} |")
    print(f"| {'Tiempo de ejecucion [s]':<27} | {t_im_str:<27} | {t_il_str:<21} |")
    print("=" * ANCHO)


# =============================================================================
# 7. GRAFICA COMPARATIVA (SUBPLOTS 1x2 CON ORIGEN VISIBLE Y MANEJO DE SALTEADOS)
# =============================================================================
def graficar_comparacion(f, var, a, b, r_im, it_im, r_il, it_il):
    if not HAS_MATPLOTLIB:
        return
    x_min = min(a, 0.0) - 1.0
    x_max = max(b, 0.0) + 1.0

    xs = np.linspace(x_min, x_max, 500)
    ys = []
    for x in xs:
        try:
            val = f(x)
            if abs(val) > 1e4:
                ys.append(float('nan'))
            else:
                ys.append(val)
        except Exception:
            ys.append(float('nan'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title, raiz, iters, col in [
        (ax1, "Intervalo Medio (Biseccion)", r_im, it_im, "red"),
        (ax2, "Interpolacion Lineal", r_il, it_il, "blue")
    ]:
        ax.plot(xs, ys, color="tab:blue", linewidth=2.0, label=f"f({var})")
        # Ejes cartesianos en negro formando cruz visible en (0, 0)
        ax.axhline(0, color='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_xlim(x_min, x_max)

        # Referencia visual del intervalo [a, b]
        ax.axvline(a, color='gray', linestyle=':', linewidth=1.3, label=f'Intervalo [{a:.2f}, {b:.2f}]')
        ax.axvline(b, color='gray', linestyle=':', linewidth=1.3)

        if raiz is not None:
            ax.plot(raiz, f(raiz), marker='o', color=col, markersize=7, label=f"Raiz: {raiz:.6f} ({iters} iters)")
        else:
            ax.text(0.5, 0.5, "Metodo no ejecutado\n(No cumplio condiciones)",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='darkred',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', alpha=0.8))

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(str(var))
        ax.set_ylabel(f"f({var})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    print("\nVisualizando grafica comparativa...")
    plt.show()


# =============================================================================
# PROGRAMA PRINCIPAL (FLUJO ESTRICTO: ANALISIS -> EJECUCION -> ANALISIS -> EJECUCION)
# =============================================================================
def main():
    print("=" * ANCHO)
    print("   METODOS COMPUTACIONALES - BISECCION VS INTERPOLACION LINEAL")
    print("=" * ANCHO)

    expr_in = input("Ingrese la funcion matematica f(x): ").strip()
    if not expr_in:
        expr_in = "pi * x^2 * ((9 - x) / 3) - 30"

    expr, var, df_expr, f, df = parsear_funcion(expr_in)
    print(f"--> Funcion interpretada: f({var}) = {expr}")
    print(f"--> Derivada analitica:   f'({var}) = {df_expr}\n")

    paso_in = input("Ingrese el incremento / paso para el tanteo en [-15, 15] [0.5]: ").strip()
    try:
        paso = float(paso_in) if paso_in else 0.5
        if paso <= 0:
            paso = 0.5
    except ValueError:
        paso = 0.5

    print(f"\nMetodo de Tanteo (evaluando de -15 a 15 con paso {paso})...")
    intervalos = metodo_tanteo(f, a=-15.0, b=15.0, paso=paso)

    if not intervalos:
        print(f"[ERROR] No se encontraron intervalos con cambio de signo en [-15, 15] con paso {paso}.")
        return

    print(f"\nSe encontraron {len(intervalos)} intervalo(s) con cambio de signo:")
    for i, (a_int, b_int) in enumerate(intervalos, start=1):
        print(f"  {i}. [{a_int:.4f}, {b_int:.4f}] -> f(a) = {f(a_int):.4e}, f(b) = {f(b_int):.4e}")

    opc = input(f"\nSeleccione el intervalo a evaluar (1-{len(intervalos)}) [1]: ").strip()
    idx = int(opc) if opc else 1
    a_elegido, b_elegido = intervalos[idx - 1]

    tol_in = input("Ingrese la tolerancia E [0.001]: ").strip()
    tol = float(tol_in) if tol_in else 0.001

    # =========================================================================
    # BLOQUE 1: ANALISIS DE BISECCION -> EJECUCION DE BISECCION (SI CUMPLE)
    # =========================================================================
    if analisis_convergencia_biseccion(f, a_elegido, b_elegido, expr, var):
        r_im, it_im, err_im, t_im = metodo_intervalo_medio(f, a_elegido, b_elegido, tol)
    else:
        r_im, it_im, err_im, t_im = None, None, None, None

    # =========================================================================
    # BLOQUE 2: ANALISIS DE INTERPOLACION -> EJECUCION DE INTERPOLACION (SI CUMPLE)
    # =========================================================================
    if analisis_convergencia_interpolacion(f, df, a_elegido, b_elegido, expr, var):
        r_il, it_il, err_il, t_il = metodo_interpolacion_lineal(f, a_elegido, b_elegido, tol)
    else:
        r_il, it_il, err_il, t_il = None, None, None, None

    # =========================================================================
    # BLOQUE 3: CUADRO COMPARATIVO RESUMIDO
    # =========================================================================
    mostrar_cuadro_comparativo(r_im, it_im, t_im, err_im, r_il, it_il, t_il, err_il, f)

    # =========================================================================
    # BLOQUE 4: GRAFICA COMPARATIVA
    # =========================================================================
    graficar_comparacion(f, var, a_elegido, b_elegido, r_im, it_im, r_il, it_il)


if __name__ == "__main__":
    main()
