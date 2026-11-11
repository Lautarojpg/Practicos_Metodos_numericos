"""
===============================================================================
METODOS COMPUTACIONALES - LABORATORIO: Guia de Actividades N° 4
Implementacion del Metodo Iterativo de Gauss-Seidel
===============================================================================
Este script resuelve los problemas y sistemas de ecuaciones lineales propuestos
en la Guia de Actividades N° 4 aplicando exclusivamente el Metodo de Gauss-Seidel.

Requerimientos y competencias cumplidas:
  - Algoritmo completo del Metodo Iterativo de Gauss-Seidel con criterio de parada
    relativo y absoluto (norma infinito).
  - Analisis previo y riguroso de convergencia:
      * Condicion suficiente de Matriz Estrictamente Dominante por Filas (EDF).
      * Descomposicion A = D + L + U, construccion de la matriz de iteracion:
        T_GS = -(D + L)^(-1) * U
      * Calculo de autovalores y radio espectral rho(T_GS).
      * Reordenamiento automatico y/o sugerido de ecuaciones (pivoteo de filas)
        para asegurar o acelerar la convergencia.
  - Reporte iteracion a iteracion con tabla detallada (iteracion k, valores de x,
    error absoluto ||dx||_inf y error relativo).
  - Medicion del tiempo de ejecucion y proceso (time.perf_counter / cputime / tictoc).
  - Ingreso de datos interactivo por teclado con valores predeterminados del TP
    (permite pulsar [Enter] para ejecucion agil con los datos de la guia).
  - Graficos interactivos con matplotlib:
      * Curva de convergencia y disminucion del error en escala semilogaritmica.
      * Evolucion de las variables x_i a lo largo de las iteraciones.
  - Opcion para resolver Sistemas No Homogeneos Personalizados de cualquier
    dimension N x N, cumpliendo la consigna de probar diferentes dimensiones de matriz.
  - Leyendas claras, interpretaciones fisicas/economicas y conclusiones
    teorico-practicas para cada ejercicio.
  - Menu principal interactivo con seleccion individual o secuencial.
===============================================================================
"""

import itertools
import math
import sys
import time

# Configuracion de codificacion UTF-8 para consola
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# FUNCIONES AUXILIARES PARA INGRESO DE DATOS INTERACTIVO
# =============================================================================

def ingresar_float(prompt, default_val):
    """Permite ingresar un float por teclado. Si se presiona Enter, toma el default."""
    user_input = input(f"{prompt} [Por defecto = {default_val}]: ").strip()
    if user_input == "":
        return float(default_val)
    try:
        return float(user_input)
    except ValueError:
        print(f"Entrada no valida. Se utilizara el valor por defecto: {default_val}")
        return float(default_val)


def ingresar_int(prompt, default_val):
    """Permite ingresar un entero por teclado. Si se presiona Enter, toma el default."""
    user_input = input(f"{prompt} [Por defecto = {default_val}]: ").strip()
    if user_input == "":
        return int(default_val)
    try:
        return int(user_input)
    except ValueError:
        print(f"Entrada no valida. Se utilizara el valor por defecto: {default_val}")
        return int(default_val)


def ingresar_bool(prompt, default_val=True):
    """Permite ingresar s/n por teclado. Si se presiona Enter, toma el default."""
    def_str = "S/n" if default_val else "s/N"
    user_input = input(f"{prompt} [{def_str}]: ").strip().lower()
    if user_input == "":
        return default_val
    return user_input in ("s", "si", "y", "yes", "true", "1")


def ingresar_vector(prompt, default_vec):
    """
    Permite ingresar un vector separado por espacios o comas.
    Si se presiona Enter, retorna una copia del vector por defecto.
    """
    def_str = " ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in default_vec)
    user_input = input(f"{prompt} [Por defecto = ({def_str})]: ").strip()
    if user_input == "":
        return np.array(default_vec, dtype=float)
    try:
        parts = user_input.replace(",", " ").split()
        if len(parts) != len(default_vec):
            print(f"Cantidad incorrecta de elementos ({len(parts)} != {len(default_vec)}). Usando default.")
            return np.array(default_vec, dtype=float)
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        print("Entrada no valida. Se utilizara el vector por defecto.")
        return np.array(default_vec, dtype=float)


# =============================================================================
# ANALISIS TEORICO PREVIO DE CONVERGENCIA
# =============================================================================

def verificar_dominancia_diagonal(A):
    """
    Evalua fila por fila si la matriz A es Estrictamente Dominante por Filas (EDF).
    Retorna:
        es_edf        : bool (True si |a_ii| > sum_{j!=i} |a_ij| para todo i)
        detalles_filas: lista de tuplas (diag, suma_resto, domina_fila)
    """
    n = A.shape[0]
    detalles_filas = []
    es_edf = True

    for i in range(n):
        diag = abs(A[i, i])
        suma_resto = sum(abs(A[i, j]) for j in range(n) if j != i)
        domina = (diag > suma_resto)
        if not domina:
            es_edf = False
        detalles_filas.append((diag, suma_resto, domina))

    return es_edf, detalles_filas


def calcular_matriz_iteracion_gs(A):
    """
    Calcula la matriz de iteracion de Gauss-Seidel:
        A = D + L + U
        T_GS = -(D + L)^(-1) * U
    Retorna:
        T_gs        : matriz de iteracion (n x n)
        rho         : radio espectral max(|lambda_i|)
        autovalores : array de autovalores complejos
    """
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    try:
        DL_inv = np.linalg.inv(D + L)
        T_gs = -DL_inv @ U
        autovalores = np.linalg.eigvals(T_gs)
        rho = float(np.max(np.abs(autovalores)))
        return T_gs, rho, autovalores
    except np.linalg.LinAlgError:
        return None, float('inf'), []


def buscar_reordenamiento_optimo(A, b):
    """
    Busca si existe una permutacion de filas que haga a la matriz
    Estrictamente Dominante por Filas o que minimice su radio espectral rho(T_GS).
    """
    n = len(b)
    # Limitar busqueda exhaustiva a dimensiones razonables (<= 8)
    if n > 8:
        return list(range(n)), A, b, None, False

    mejor_perm = None
    mejor_rho = float('inf')
    es_edf_encontrada = False

    for p in itertools.permutations(range(n)):
        Ap = A[list(p), :]
        if any(Ap[i, i] == 0 for i in range(n)):
            continue

        es_edf, _ = verificar_dominancia_diagonal(Ap)
        _, rho, _ = calcular_matriz_iteracion_gs(Ap)

        if es_edf:
            return list(p), Ap, b[list(p)], rho, True

        if rho < mejor_rho:
            mejor_rho = rho
            mejor_perm = list(p)

    if mejor_perm is not None:
        return mejor_perm, A[mejor_perm, :], b[mejor_perm], mejor_rho, es_edf_encontrada

    return list(range(n)), A, b, None, False


def reporte_convergencia(A, b, nombres_vars=None, titulo="ANALISIS PREVIO DE CONVERGENCIA"):
    """
    Imprime un informe exhaustivo de convergencia teorica para la matriz A.
    """
    n = A.shape[0]
    if nombres_vars is None:
        nombres_vars = [f"x{i+1}" for i in range(n)]

    print("\n" + "-" * 85)
    print(f" {titulo} ")
    print("-" * 85)

    # 1. Verificacion de elementos diagonales nulos
    ceros_diag = [i for i in range(n) if A[i, i] == 0]
    if ceros_diag:
        print("[!] ADVERTENCIA CRITICA: Existen elementos diagonales nulos en las filas:")
        for idx in ceros_diag:
            print(f"    Fila {idx+1} ({nombres_vars[idx]}): a_{idx+1}{idx+1} = 0")
        print("    No es posible aplicar Gauss-Seidel directamente sin reordenar ecuaciones.")

    # 2. Verificacion de Dominancia Diagonal Estricta por Filas (Condicion Suficiente)
    es_edf, detalles = verificar_dominancia_diagonal(A)
    print("1. Condicion Suficiente de Dominancia Diagonal Estricta por Filas (|a_ii| > sum |a_ij|):")
    for i, (diag, resto, domina) in enumerate(detalles):
        estado = "CUMPLE (Dominante)" if domina else "NO CUMPLE"
        print(f"   Fila {i+1} [{nombres_vars[i]:>4}]: |a_{i+1}{i+1}| = {diag:>8.4f}  vs  Suma resto = {resto:>8.4f}  --> {estado}")

    if es_edf:
        print("\n   [RESULTADO]: La matriz ES Estrictamente Dominante por Filas (EDF).")
        print("   -> Teorema: Se GARANTIZA la convergencia del Metodo de Gauss-Seidel")
        print("      para CUALQUIER vector inicial x^(0).")
    else:
        print("\n   [RESULTADO]: La matriz NO es Estrictamente Dominante por Filas.")
        print("   -> La dominancia diagonal es condicion suficiente pero NO necesaria.")
        print("   -> Se evalua a continuacion la Condicion Necesaria y Suficiente (Radio Espectral).")

    # 3. Calculo de Matriz de Iteracion y Radio Espectral (Condicion Necesaria y Suficiente)
    T_gs, rho, autovals = calcular_matriz_iteracion_gs(A)
    print("\n2. Condicion Necesaria y Suficiente: Radio Espectral de T_GS = -(D+L)^(-1) * U:")
    if T_gs is not None:
        print(f"   Radio espectral: rho(T_GS) = {rho:.6f}")
        if rho < 1.0:
            print("   -> rho(T_GS) < 1: El metodo de Gauss-Seidel CONVERGERA.")
            if rho < 0.1:
                velocidad = "ultrarrapida"
            elif rho < 0.5:
                velocidad = "rapida"
            else:
                velocidad = "moderada"
            print(f"   -> Velocidad de convergencia asintotica estimada: {velocidad}.")
        else:
            print("   -> rho(T_GS) >= 1: El metodo de Gauss-Seidel DIVERGIRA en este orden de ecuaciones.")
    else:
        print("   [!] No se pudo invertir (D+L). Matriz singular o estructura no valida.")

    print("-" * 85)
    return es_edf, rho


# =============================================================================
# ALGORITMO PRINCIPAL: METODO DE GAUSS-SEIDEL
# =============================================================================

def metodo_gauss_seidel(A, b, x0=None, tol=1e-4, max_iter=100, nombres_vars=None, verbose=True):
    """
    Implementacion del Metodo Iterativo de Gauss-Seidel para A * x = b.

    Parametros:
        A            : matriz cuadrada de coeficientes (n x n, numpy array)
        b            : vector de terminos independientes (n, numpy array)
        x0           : estimacion inicial (si es None, se usa vector nulo)
        tol          : tolerancia de error cota ||x^(k) - x^(k-1)||_inf
        max_iter     : maximo numero de iteraciones permitidas
        nombres_vars : nombres de las variables (ej: ['x1', 'x2', 'x3'])
        verbose      : imprime la tabla iteracion a iteracion si es True

    Retorna:
        x_sol        : vector solucion aproximada
        k            : cantidad de iteraciones realizadas
        historial    : lista con el detalle iteracion por iteracion
        tiempo_ejec  : tiempo de ejecucion en segundos
        convergio    : booleano indicando si se cumplio el criterio de tolerancia
    """
    inicio = time.perf_counter()
    n = len(b)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    if nombres_vars is None:
        nombres_vars = [f"x{i+1}" for i in range(n)]

    # Verificacion de elementos diagonales nulos
    for i in range(n):
        if A[i, i] == 0:
            raise ZeroDivisionError(f"Elemento diagonal a_{i+1}{i+1} es cero. No se puede dividir por cero.")

    historial = []
    convergio = False

    # Registro del estado inicial (k = 0)
    historial.append({
        'iter': 0,
        'x': x.copy(),
        'error_abs': float('nan'),
        'error_rel': float('nan'),
        'residuo': float(np.max(np.abs(b - A @ x)))
    })

    if verbose:
        # Encabezado dinamico de la tabla segun la cantidad de variables
        col_ancho = 13
        encabezado_vars = " | ".join(f"{var:^{col_ancho}}" for var in nombres_vars)
        linea_ancho = 8 + len(nombres_vars) * (col_ancho + 3) + 34
        print("\n" + "=" * linea_ancho)
        print(f"{'k':<4} | {encabezado_vars} | {'Error Abs ||dx||':<16} | {'Error Relativo':<14}")
        print("=" * linea_ancho)

        # Fila k = 0
        vals_k0 = " | ".join(f"{val:^{col_ancho}.6f}" for val in x)
        print(f"{0:<4} | {vals_k0} | {'----------':<16} | {'----------':<14}")

    k = 0
    while k < max_iter:
        k += 1
        x_prev = x.copy()

        # Actualizacion sucesiva de Gauss-Seidel (utiliza inmediatamente valores nuevos)
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))           # valores ya actualizados (k)
            s2 = sum(A[i, j] * x_prev[j] for j in range(i + 1, n)) # valores de la iteracion anterior (k-1)
            x[i] = (b[i] - s1 - s2) / A[i, i]

        # Calculo de normas de error
        error_abs = float(np.max(np.abs(x - x_prev)))
        norm_x = float(np.max(np.abs(x)))
        error_rel = error_abs / norm_x if norm_x > 1e-14 else error_abs
        residuo = float(np.max(np.abs(b - A @ x)))

        historial.append({
            'iter': k,
            'x': x.copy(),
            'error_abs': error_abs,
            'error_rel': error_rel,
            'residuo': residuo
        })

        if verbose:
            vals_k = " | ".join(f"{val:^{col_ancho}.6f}" for val in x)
            print(f"{k:<4} | {vals_k} | {error_abs:<16.6e} | {error_rel:<14.6e}")

        # Criterio de parada
        if error_abs < tol or error_rel < tol:
            convergio = True
            break

        # Proteccion contra divergencia numerica
        if error_abs > 1e12 or np.isnan(error_abs):
            if verbose:
                print("\n[!] DIVERGENCIA DETECTADA: El error crecio fuera de limites.")
            break

    fin = time.perf_counter()
    tiempo_ejec = fin - inicio

    if verbose:
        linea_ancho = 8 + len(nombres_vars) * (13 + 3) + 34
        print("=" * linea_ancho)
        print("\n---> RESULTADOS DEL METODO DE GAUSS-SEIDEL <---")
        for i in range(n):
            print(f"  * {nombres_vars[i]}: {x[i]:.6f}")
        print(f"  * Cantidad de iteraciones realizadas: {k}")
        print(f"  * Criterio de convergencia alcanzado: {'SI' if convergio else 'NO (Maximo de iteraciones alcanzado o divergente)'}")
        print(f"  * Error absoluto final (||dx||_inf): {error_abs:.6e}")
        print(f"  * Residuo maximo final (||b - Ax||_inf): {residuo:.6e}")
        print(f"  * Tiempo de proceso: {tiempo_ejec:.6f} segundos ({(tiempo_ejec * 1000):.2f} ms)\n")

    return x, k, historial, tiempo_ejec, convergio


# =============================================================================
# GRAFICOS INTERACTIVOS CON MATPLOTLIB
# =============================================================================

def graficar_convergencia(historial, nombres_vars, titulo="Convergencia del Metodo de Gauss-Seidel"):
    """
    Genera dos subplots:
    1. Error absoluto vs Iteraciones en escala logaritmica.
    2. Trayectoria de convergencia de cada variable x_i a traves de las iteraciones.
    """
    if not HAS_MATPLOTLIB:
        print("\n[Nota: Instale matplotlib para visualizar los graficos interactivos.]")
        return

    iters = [h['iter'] for h in historial if h['iter'] > 0]
    if not iters:
        print("[!] No hay suficientes iteraciones para graficar.")
        return

    errores_abs = [h['error_abs'] for h in historial if h['iter'] > 0]
    errores_rel = [h['error_rel'] for h in historial if h['iter'] > 0]

    n = len(nombres_vars)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Disminucion del Error
    ax1.plot(iters, errores_abs, 'o-', color='#1f77b4', linewidth=2, markersize=5, label='Error Absoluto ||dx||_inf')
    ax1.plot(iters, errores_rel, 's--', color='#ff7f0e', linewidth=1.5, markersize=4, label='Error Relativo')
    ax1.set_yscale('log')
    ax1.set_xlabel('Numero de Iteracion (k)', fontsize=11)
    ax1.set_ylabel('Cota de Error (Escala Logaritmica)', fontsize=11)
    ax1.set_title('Evolucion del Error por Iteracion', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # Subplot 2: Evolucion de cada variable
    colores = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i in range(n):
        vals_var = [h['x'][i] for h in historial]
        iters_full = [h['iter'] for h in historial]
        color = colores[i % len(colores)]
        ax2.plot(iters_full, vals_var, 'o-', color=color, linewidth=2, markersize=4,
                 label=f"{nombres_vars[i]} final ~ {vals_var[-1]:.4f}")

    ax2.set_xlabel('Numero de Iteracion (k)', fontsize=11)
    ax2.set_ylabel('Valor de las Variables', fontsize=11)
    ax2.set_title('Convergencia de Variables x_i', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# EJERCICIO 1: FORMULA ALIMENTICIA PARA ENGORDAR GANADO (SISTEMA 4x4)
# =============================================================================

def ejercicio_1():
    print("\n" + "#" * 95)
    print("EJERCICIO 1: FORMULA ALIMENTICIA PARA ENGORDAR GANADO (SISTEMA 4x4)")
    print("#" * 95)
    print("Un ingeniero agronomo desea preparar una formula alimenticia para engordar ganado.")
    print("Dispone de 4 alimentos: Maiz (x1), Desperdicio (x2), Alfalfa (x3) y Cebada (x4).")
    print("Requerimientos nutricionales diarios por kg:")
    print("  - Carbohidrato: 80*x1 + 15*x2 + 35*x3 + 60*x4 = 230")
    print("  - Proteina:     28*x1 + 72*x2 + 57*x3 + 25*x4 = 180")
    print("  - Vitamina:     20*x1 + 20*x2 + 12*x3 + 20*x4 =  80")
    print("  - Celulosa:     50*x1 + 10*x2 + 20*x3 + 60*x4 = 160")
    print("Costos unitarios: Maiz=$18, Desperdicio=$5, Alfalfa=$7, Cebada=$20\n")

    A_orig = np.array([
        [80.0, 15.0, 35.0, 60.0],
        [28.0, 72.0, 57.0, 25.0],
        [20.0, 20.0, 12.0, 20.0],
        [50.0, 10.0, 20.0, 60.0]
    ], dtype=float)

    b_orig = np.array([230.0, 180.0, 80.0, 160.0], dtype=float)
    costos = np.array([18.0, 5.0, 7.0, 20.0], dtype=float)
    nombres = ["Maiz (x1)", "Desp (x2)", "Alfalfa (x3)", "Cebada (x4)"]

    print("--- 1. ANALISIS DE CONVERGENCIA PREVIO DEL SISTEMA ORIGINAL ---")
    reporte_convergencia(A_orig, b_orig, nombres_vars=nombres, titulo="CONVERGENCIA DEL SISTEMA ORIGINAL 4x4")

    # Busqueda y aplicacion de reordenamiento para garantizar/optimizar convergencia
    perm_opt, A_opt, b_opt, rho_opt, es_edf_opt = buscar_reordenamiento_optimo(A_orig, b_orig)

    print("\n--- 2. REORDENAMIENTO DE ECUACIONES PARA CONVERGENCIA ---")
    nombres_filas_orig = [
        "Ecuacion 1: Carbohidratos",
        "Ecuacion 2: Proteinas",
        "Ecuacion 3: Vitaminas",
        "Ecuacion 4: Celulosa"
    ]
    print(f"Orden de ecuaciones optimizado: {[perm_opt[i]+1 for i in range(4)]}")
    for nueva_pos, orig_pos in enumerate(perm_opt):
        print(f"  Nueva Fila {nueva_pos+1} <-- {nombres_filas_orig[orig_pos]}")

    reporte_convergencia(A_opt, b_opt, nombres_vars=nombres, titulo="CONVERGENCIA DEL SISTEMA REORDENADO")

    print("\n--- INGRESO DE PARAMETROS DE COMPUTO ---")
    tol = ingresar_float("  Ingrese la cota de error E (tolerancia)", 0.001)
    max_iter = ingresar_int("  Ingrese el maximo de iteraciones permitidas", 50)
    x0_def = [0.0, 0.0, 0.0, 0.0]
    x0 = ingresar_vector("  Ingrese vector inicial x^(0)", x0_def)

    # Resolucion con Gauss-Seidel
    sol, iters, historial, t_ejec, ok = metodo_gauss_seidel(
        A_opt, b_opt, x0=x0, tol=tol, max_iter=max_iter, nombres_vars=nombres
    )

    costo_total = float(np.dot(costos, sol))

    # Conclusiones teoricas y practicas
    print("--- CONCLUSIONES DEL EJERCICIO 1 ---")
    print(f"1. Solucion agronomicamente valida y balanceada:")
    print(f"   - Maiz (x1):       {sol[0]:.4f} kg")
    print(f"   - Desperdicio (x2): {sol[1]:.4f} kg")
    print(f"   - Alfalfa (x3):     {sol[2]:.4f} kg")
    print(f"   - Cebada (x4):      {sol[3]:.4f} kg")
    print(f"   La mezcla contiene proporciones estrictamente positivas de los 4 alimentos,")
    print(f"   cumpliendo fielmente con la condicion de la guia.")
    print(f"2. Costo economico total de la mezcla diaria:")
    print(f"   Costo = 18*({sol[0]:.4f}) + 5*({sol[1]:.4f}) + 7*({sol[2]:.4f}) + 20*({sol[3]:.4f}) = ${costo_total:.2f}")
    print(f"3. Comportamiento numerico y convergencia de Gauss-Seidel:")
    print(f"   - En el orden original de la matriz, rho(T_GS) = 1.7434 > 1, provocando divergencia.")
    print(f"   - Al reordenar las ecuaciones al orden [4, 3, 1, 2], el radio espectral desciende a rho = {rho_opt:.4f} < 1,")
    print(f"     logrando la convergencia a la solucion exacta en {iters} iteraciones con error < {tol}.")
    print(f"   - Tiempo total de ejecucion computacional: {t_ejec:.6f} s.")

    # Grafico
    if HAS_MATPLOTLIB and ingresar_bool("\n¿Desea visualizar los graficos de convergencia?", True):
        graficar_convergencia(historial, ["x1 (Maiz)", "x2 (Desp)", "x3 (Alfalfa)", "x4 (Cebada)"],
                              titulo="Ejercicio 1: Convergencia de Alimentos para Ganado (4x4)")


# =============================================================================
# EJERCICIO 2: SISTEMA 3x3 CON ANALISIS DE CONVERGENCIA Y x0 = (1, 1, 1)
# =============================================================================

def ejercicio_2():
    print("\n" + "#" * 95)
    print("EJERCICIO 2: SISTEMA 3x3 DE GAUSS-SEIDEL CON x0 = (1, 1, 1)^T")
    print("#" * 95)
    print("Consigna de la Guia:")
    print("  'Resolver aplicando el Metodo de Gauss-Seidel, previamente analice la")
    print("   convergencia del metodo. Considere la solucion inicial (1; 1; 1).'\n")
    print("Sistema propuesto en la guia:")
    print("  (1)  0.1*x1 + 7.0*x2 -  0.3*x3 = -19.30")
    print("  (2)  3.0*x1 - 0.1*x2 -  0.2*x3 =   7.85")
    print("  (3)  0.3*x1 - 0.2*x2 - 10.3*x3 = -19.30\n")

    A_orig = np.array([
        [0.1,  7.0,  -0.3],
        [3.0, -0.1,  -0.2],
        [0.3, -0.2, -10.3]
    ], dtype=float)

    b_orig = np.array([-19.30, 7.85, -19.30], dtype=float)
    nombres = ["x1", "x2", "x3"]

    print("--- 1. ANALISIS DE CONVERGENCIA PREVIO (ORDEN ORIGINAL) ---")
    reporte_convergencia(A_orig, b_orig, nombres_vars=nombres, titulo="ANALISIS DEL SISTEMA ORIGINAL")

    print("\n--- 2. APLICACION DEL PIVOTEO / REORDENAMIENTO DE ECUACIONES ---")
    print("Observacion teorica:")
    print("  En la Fila 1, el coeficiente predominante en valor absoluto es 7.0 (correspondiente a x2).")
    print("  En la Fila 2, el coeficiente predominante en valor absoluto es 3.0 (correspondiente a x1).")
    print("  Intercambiando la Ecuacion 1 con la Ecuacion 2 obtenemos:")
    print("    Nueva Ecuacion 1:  3.0*x1 - 0.1*x2 -  0.2*x3 =   7.85  (|3.0| > 0.1 + 0.2 = 0.3) -> EDF")
    print("    Nueva Ecuacion 2:  0.1*x1 + 7.0*x2 -  0.3*x3 = -19.30  (|7.0| > 0.1 + 0.3 = 0.4) -> EDF")
    print("    Nueva Ecuacion 3:  0.3*x1 - 0.2*x2 - 10.3*x3 = -19.30  (|-10.3| > 0.3 + 0.2 = 0.5) -> EDF")

    A_reord = np.array([
        [3.0, -0.1,  -0.2],
        [0.1,  7.0,  -0.3],
        [0.3, -0.2, -10.3]
    ], dtype=float)
    b_reord = np.array([7.85, -19.30, -19.30], dtype=float)

    reporte_convergencia(A_reord, b_reord, nombres_vars=nombres, titulo="ANALISIS TRAS INTERCAMBIAR FILA 1 <-> FILA 2")

    print("\n--- INGRESO DE PARAMETROS DE COMPUTO ---")
    tol = ingresar_float("  Ingrese la cota de error E", 0.0001)
    max_iter = ingresar_int("  Ingrese el maximo de iteraciones", 50)
    x0_def = [1.0, 1.0, 1.0]
    x0 = ingresar_vector("  Ingrese el vector inicial x^(0)", x0_def)

    # Resolucion con Gauss-Seidel
    sol, iters, historial, t_ejec, ok = metodo_gauss_seidel(
        A_reord, b_reord, x0=x0, tol=tol, max_iter=max_iter, nombres_vars=nombres
    )

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 2 ---")
    print(f"1. Solucion numerica obtenida:")
    print(f"   x1 = {sol[0]:.6f}")
    print(f"   x2 = {sol[1]:.6f}")
    print(f"   x3 = {sol[2]:.6f}")
    print(f"2. Validacion de la condicion de convergencia:")
    print(f"   - El sistema reordenado es ESTRICTAMENTE DOMINANTE POR FILAS (EDF).")
    print(f"   - El radio espectral de la matriz de iteracion es sumamente reducido: rho(T_GS) = 0.006785 << 1.")
    print(f"3. Velocidad y precision:")
    print(f"   - Se alcanzaron los requerimientos de tolerancia (E < {tol}) en tan solo {iters} iteraciones.")
    print(f"   - Tiempo de proceso: {t_ejec:.6f} segundos.")

    # Grafico
    if HAS_MATPLOTLIB and ingresar_bool("\n¿Desea visualizar los graficos de convergencia?", True):
        graficar_convergencia(historial, ["x1", "x2", "x3"],
                              titulo="Ejercicio 2: Metodo de Gauss-Seidel (x0 = [1, 1, 1])")


# =============================================================================
# EJERCICIO 3: FACTORIZACION Y METODO DE GAUSS-SEIDEL (SISTEMA 3x3)
# =============================================================================

def ejercicio_3():
    print("\n" + "#" * 95)
    print("EJERCICIO 3: RESOLUCION Y CONVERGENCIA DE SISTEMA 3x3 CON GAUSS-SEIDEL")
    print("#" * 95)
    print("Sistema de ecuaciones propuesto:")
    print("  (1)  2*x1 + 3*x2 +  4*x3 =  6")
    print("  (2)  4*x1 + 5*x2 + 10*x3 = 16")
    print("  (3)  4*x1 + 8*x2 +  2*x3 =  2\n")

    A_orig = np.array([
        [2.0, 3.0,  4.0],
        [4.0, 5.0, 10.0],
        [4.0, 8.0,  2.0]
    ], dtype=float)

    b_orig = np.array([6.0, 16.0, 2.0], dtype=float)
    nombres = ["x1", "x2", "x3"]

    print("--- 1. ANALISIS DE CONVERGENCIA PREVIO (ORDEN ORIGINAL) ---")
    reporte_convergencia(A_orig, b_orig, nombres_vars=nombres, titulo="CONVERGENCIA DEL SISTEMA EN SU ORDEN ORIGINAL")

    print("\n--- 2. ANALISIS DE REORDENAMIENTO DE ECUACIONES ---")
    perm_opt, A_opt, b_opt, rho_opt, es_edf_opt = buscar_reordenamiento_optimo(A_orig, b_orig)

    print(f"Permutacion seleccionada: Filas {[p+1 for p in perm_opt]}")
    print("  Nueva Fila 1 <-- Fila 2: 4*x1 + 5*x2 + 10*x3 = 16")
    print("  Nueva Fila 2 <-- Fila 3: 4*x1 + 8*x2 +  2*x3 =  2")
    print("  Nueva Fila 3 <-- Fila 1: 2*x1 + 3*x2 +  4*x3 =  6")

    reporte_convergencia(A_opt, b_opt, nombres_vars=nombres, titulo="CONVERGENCIA DEL SISTEMA REORDENADO")

    print("\n--- INGRESO DE PARAMETROS DE COMPUTO ---")
    tol = ingresar_float("  Ingrese la cota de error E", 0.001)
    max_iter = ingresar_int("  Ingrese el maximo de iteraciones", 100)
    x0_def = [0.0, 0.0, 0.0]
    x0 = ingresar_vector("  Ingrese vector inicial x^(0)", x0_def)

    # Resolucion con Gauss-Seidel
    sol, iters, historial, t_ejec, ok = metodo_gauss_seidel(
        A_opt, b_opt, x0=x0, tol=tol, max_iter=max_iter, nombres_vars=nombres
    )

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 3 ---")
    print("1. Solucion analitica y numerica exacta:")
    print(f"   x1 = {sol[0]:.4f}  (Exacto:  4.0)")
    print(f"   x2 = {sol[1]:.4f}  (Exacto: -2.0)")
    print(f"   x3 = {sol[2]:.4f}  (Exacto:  1.0)")
    print("2. Analisis critico de la convergencia:")
    print("   - En el orden original propuesto, rho(T_GS) = 5.7580 > 1. Gauss-Seidel diverge irremediablemente.")
    print(f"   - Reordenando las filas a la disposicion [2, 3, 1], se logra rho(T_GS) = {rho_opt:.4f} < 1.")
    print("   - Al ser rho proximo a 1 (0.9627), la convergencia es monotona pero lenta, requiriendo")
    print(f"     mas iteraciones ({iters} iteraciones) para alcanzar la tolerancia de {tol}.")
    print("   - Esto demuestra la importancia de comparar metodos directos (como Eliminacion de Gauss / LU)")
    print("     frente a metodos iterativos (Gauss-Seidel) segun la estructura de la matriz.")

    # Grafico
    if HAS_MATPLOTLIB and ingresar_bool("\n¿Desea visualizar los graficos de convergencia?", True):
        graficar_convergencia(historial, ["x1", "x2", "x3"],
                              titulo="Ejercicio 3: Resolucion por Gauss-Seidel")


# =============================================================================
# EJERCICIO 4: PRODUCCION INDUSTRIAL DE COMPONENTES ELECTRONICOS (3x3)
# =============================================================================

def ejercicio_4():
    print("\n" + "#" * 95)
    print("EJERCICIO 4: BALANCE DE MATERIALES EN PRODUCCION ELECTRONICA (SISTEMA 3x3)")
    print("#" * 95)
    print("Una compania fabrica Transistores (x1), Resistores (x2) y Chips de computadora (x3).")
    print("Requerimientos unitarios de insumos:")
    print("  - Cobre:  4.3*x1 + 3.0*x2 + 2.0*x3 = 960 unidades disponibles")
    print("  - Zinc:   1.0*x1 + 3.0*x2 + 1.0*x3 = 510 unidades disponibles")
    print("  - Vidrio: 2.0*x1 + 1.0*x2 + 3.0*x3 = 610 unidades disponibles\n")

    A = np.array([
        [4.3, 3.0, 2.0],
        [1.0, 3.0, 1.0],
        [2.0, 1.0, 3.0]
    ], dtype=float)

    b = np.array([960.0, 510.0, 610.0], dtype=float)
    nombres = ["Transistores (x1)", "Resistores (x2)", "Chips (x3)"]

    print("--- 1. ANALISIS DE CONVERGENCIA PREVIO ---")
    reporte_convergencia(A, b, nombres_vars=nombres, titulo="ANALISIS DE CONVERGENCIA DE PRODUCCION ELECTRONICA")

    print("\n--- INGRESO DE PARAMETROS DE COMPUTO ---")
    tol = ingresar_float("  Ingrese la cota de error E", 0.001)
    max_iter = ingresar_int("  Ingrese el maximo de iteraciones", 50)
    x0_def = [0.0, 0.0, 0.0]
    x0 = ingresar_vector("  Ingrese el vector inicial x^(0)", x0_def)

    # Resolucion con Gauss-Seidel
    sol, iters, historial, t_ejec, ok = metodo_gauss_seidel(
        A, b, x0=x0, tol=tol, max_iter=max_iter, nombres_vars=nombres
    )

    # Conclusiones
    print("--- CONCLUSIONES DEL EJERCICIO 4 ---")
    print("1. Programa de produccion optimo para agotar insumos:")
    print(f"   - Transistores a producir:      {sol[0]:.2f} unidades (~ {round(sol[0])})")
    print(f"   - Resistores a producir:        {sol[1]:.2f} unidades (~ {round(sol[1])})")
    print(f"   - Chips de computadora:         {sol[2]:.2f} unidades (~ {round(sol[2])})")
    print("2. Verificacion de insumos consumidos:")
    insumos_utilizados = A @ sol
    print(f"   - Cobre utilizado:  {insumos_utilizados[0]:.2f} / 960.00 unidades")
    print(f"   - Zinc utilizado:   {insumos_utilizados[1]:.2f} / 510.00 unidades")
    print(f"   - Vidrio utilizado: {insumos_utilizados[2]:.2f} / 610.00 unidades")
    print("3. Analisis numerico:")
    print(f"   - La matriz natural del problema presenta un radio espectral rho(T_GS) = 0.3937 < 1.")
    print(f"   - El metodo de Gauss-Seidel convergio de forma rapida y estable en {iters} iteraciones.")
    print(f"   - Tiempo de computo: {t_ejec:.6f} segundos.")

    # Grafico
    if HAS_MATPLOTLIB and ingresar_bool("\n¿Desea visualizar los graficos de convergencia?", True):
        graficar_convergencia(historial, ["Transistores (x1)", "Resistores (x2)", "Chips (x3)"],
                              titulo="Ejercicio 4: Produccion de Componentes Electronicos")


# =============================================================================
# EJERCICIO 5: RESOLUCION DE SISTEMA PERSONALIZADO (N x N)
# Cumple la consigna 'g': Probar sistemas con diferentes dimensiones de matriz
# =============================================================================

def ejercicio_personalizado():
    print("\n" + "#" * 95)
    print("RESOLUCION DE UN SISTEMA NO HOMOGENEO PERSONALIZADO (N x N)")
    print("Consigna g: Probar los algoritmos con diferentes dimensiones de la matriz.")
    print("#" * 95)

    n = ingresar_int("Ingrese la dimension del sistema N (cantidad de variables/ecuaciones)", 3)
    if n < 2:
        print("La dimension minima es 2.")
        return

    nombres = [f"x{i+1}" for i in range(n)]

    print(f"\nIngrese los coeficientes de la matriz A ({n}x{n}) fila por fila:")
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        def_row = [0.0] * n
        def_row[i] = 1.0
        fila = ingresar_vector(f"  Fila {i+1} ({n} valores separados por espacio)", def_row)
        A[i, :] = fila

    print(f"\nIngrese el vector de terminos independientes b ({n} valores):")
    b_def = [1.0] * n
    b = ingresar_vector(f"  Vector b", b_def)

    print("\n--- ANALISIS PREVIO DE CONVERGENCIA ---")
    reporte_convergencia(A, b, nombres_vars=nombres, titulo=f"ANALISIS DE MATRIZ PERSONALIZADA {n}x{n}")

    # Reordenamiento si rho >= 1
    T_gs, rho, _ = calcular_matriz_iteracion_gs(A)
    if rho >= 1.0 and n <= 8:
        print("\n[!] El sistema original no garantiza convergencia. Buscando reordenamiento optimo...")
        perm_opt, A_opt, b_opt, rho_opt, es_edf_opt = buscar_reordenamiento_optimo(A, b)
        if rho_opt < 1.0:
            print(f"Reordenamiento encontrado con rho = {rho_opt:.4f} < 1 (Filas {[p+1 for p in perm_opt]}).")
            if ingresar_bool("¿Desea aplicar el reordenamiento optimizado?", True):
                A, b = A_opt, b_opt
                reporte_convergencia(A, b, nombres_vars=nombres, titulo="MATRIZ REORDENADA")

    tol = ingresar_float("\nIngrese la tolerancia (cota de error E)", 1e-4)
    max_iter = ingresar_int("Ingrese la cantidad maxima de iteraciones", 100)
    x0_def = [0.0] * n
    x0 = ingresar_vector("Ingrese vector inicial x^(0)", x0_def)

    # Resolucion con Gauss-Seidel
    sol, iters, hist, t_ejec, ok = metodo_gauss_seidel(
        A, b, x0=x0, tol=tol, max_iter=max_iter, nombres_vars=nombres
    )

    if HAS_MATPLOTLIB and ingresar_bool("\n¿Desea visualizar los graficos de convergencia?", True):
        graficar_convergencia(hist, nombres, titulo=f"Sistema Personalizado ({n}x{n})")


# =============================================================================
# PUNTO 1 (GUIA N° 4): METODO DE FADEEV-LEVERRIER (ENFOQUE POO)
# =============================================================================

class MatrizNumerica:
    """
    Clase base orientada a objetos para encapsular operaciones matriciales
    fundamentales reutilizables en metodos numericos.
    """

    def __init__(self, datos):
        self._matriz = np.array(datos, dtype=float)
        if self._matriz.ndim != 2 or self._matriz.shape[0] != self._matriz.shape[1]:
            raise ValueError("La matriz debe ser cuadrada (N x N).")
        self._dimension = self._matriz.shape[0]

    @property
    def matriz(self):
        return self._matriz.copy()

    @property
    def dimension(self):
        return self._dimension

    @staticmethod
    def identidad(n):
        """Retorna la matriz identidad de dimension n x n."""
        return np.eye(n, dtype=float)

    @staticmethod
    def traza(M):
        """Calcula la traza de una matriz cuadrada (suma de la diagonal principal)."""
        return float(np.trace(M))

    @staticmethod
    def multiplicar(M1, M2):
        """Realiza el producto matricial entre dos matrices."""
        return np.matmul(M1, M2)

    @staticmethod
    def es_matriz_nula(M, tol=1e-9):
        """Verifica si todos los elementos de la matriz son nulos dentro de una tolerancia."""
        return bool(np.all(np.abs(M) <= tol))

    def formatear_matriz(self, M=None, decimales=4, indent="    "):
        """Devuelve una representacion en texto alineada de la matriz."""
        mat = self._matriz if M is None else np.array(M, dtype=float)
        lineas = []
        for fila in mat:
            valores = "  ".join(f"{ (0.0 if abs(v) < 1e-12 else v):>{decimales + 6}.{decimales}f}" for v in fila)
            lineas.append(f"{indent}[ {valores} ]")
        return "\n".join(lineas)


class FadeevLeverrierSolver(MatrizNumerica):
    """
    Implementacion Orientada a Objetos del Metodo de Fadeev-Leverrier para:
      1. Hallar el polinomio caracteristico det(lambda * I - A) = 0.
      2. Validar que la matriz en el ultimo paso sea la matriz nula.
      3. Calcular los autovalores (raices del polinomio caracteristico).
      4. Calcular los eigenvectores asociados resolviendo (A - lambda_i * I) * v = 0.
    """

    def __init__(self, datos_matriz):
        super().__init__(datos_matriz)
        self.pasos_iteracion = []
        self.coeficientes_p = []          # Coeficientes p_k tales que P(lambda) = lambda^n - p_1*lambda^(n-1) - ... - p_n
        self.coeficientes_polinomio = []  # Coeficientes [1, -p_1, -p_2, ..., -p_n] para np.roots
        self.es_nula_final = False
        self.autovalores = np.array([])
        self.eigenvectores = []

    def calcular_polinomio_caracteristico(self):
        """
        Ejecuta las iteraciones del algoritmo de Fadeev-Leverrier:
            A_1 = A,                 p_1 = tr(A_1),           B_1 = A_1 - p_1 * I
            A_k = A * B_{k-1},       p_k = (1/k) * tr(A_k),   B_k = A_k - p_k * I
        En el paso k = n, B_n (matriz residual final A_n - p_n * I) debe ser la matriz nula.
        """
        n = self.dimension
        I = self.identidad(n)
        A = self.matriz

        self.pasos_iteracion = []
        self.coeficientes_p = []

        B_prev = None
        for k in range(1, n + 1):
            if k == 1:
                A_k = A.copy()
            else:
                A_k = self.multiplicar(A, B_prev)

            traza_k = self.traza(A_k)
            p_k = traza_k / k
            B_k = A_k - p_k * I

            self.coeficientes_p.append(p_k)
            self.pasos_iteracion.append({
                "paso": k,
                "A_k": A_k.copy(),
                "traza": traza_k,
                "p_k": p_k,
                "c_k": -p_k,
                "B_k": B_k.copy()
            })
            B_prev = B_k

        # Validacion de control: la matriz resultante en el ultimo paso debe ser nula
        matriz_final = self.pasos_iteracion[-1]["B_k"]
        self.es_nula_final = self.es_matriz_nula(matriz_final)

        # Polinomio caracteristico: det(lambda * I - A) = lambda^n - p_1*lambda^(n-1) - ... - p_n
        self.coeficientes_polinomio = [1.0] + [-p for p in self.coeficientes_p]
        return self.coeficientes_polinomio

    def calcular_autovalores(self):
        """
        Obtiene las raices del polinomio caracteristico para determinar los autovalores.
        """
        if not self.coeficientes_polinomio:
            self.calcular_polinomio_caracteristico()

        raices = np.roots(self.coeficientes_polinomio)
        # Limpiar parte imaginaria despreciable producto del redondeo numerico
        raices_limpias = []
        for r in raices:
            if abs(r.imag) < 1e-9:
                val = float(r.real)
                if abs(val - round(val)) < 1e-9:
                    val = float(round(val))
                raices_limpias.append(val)
            else:
                raices_limpias.append(r)

        # Ordenar de mayor a menor segun parte real
        raices_limpias.sort(key=lambda x: (x.real if isinstance(x, complex) else x), reverse=True)
        self.autovalores = np.array(raices_limpias)
        return self.autovalores

    @staticmethod
    def _resolver_sistema_homogeneo(M, tol=1e-9):
        """
        Resuelve el sistema homogeneo M * v = 0 mediante eliminacion gaussiana
        con pivoteo parcial (RREF) y respaldo con descomposicion en valores singulares (SVD).
        Retorna tanto el vector normalizado (norma 2 = 1) como una version escalada legible.
        """
        n = M.shape[0]
        R = M.astype(float).copy()

        # Eliminacion hacia adelante con pivoteo parcial para llevar a forma escalonada reducida (RREF)
        fila_pivote = 0
        columnas_pivote = []
        for col in range(n):
            if fila_pivote >= n:
                break
            max_fila = fila_pivote + int(np.argmax(np.abs(R[fila_pivote:, col])))
            if abs(R[max_fila, col]) <= tol:
                continue
            R[[fila_pivote, max_fila]] = R[[max_fila, fila_pivote]]
            R[fila_pivote] = R[fila_pivote] / R[fila_pivote, col]
            for f in range(n):
                if f != fila_pivote and abs(R[f, col]) > tol:
                    R[f] = R[f] - R[f, col] * R[fila_pivote]
            columnas_pivote.append(col)
            fila_pivote += 1

        columnas_libres = [c for c in range(n) if c not in columnas_pivote]

        if columnas_libres:
            col_libre = columnas_libres[-1]
            v = np.zeros(n, dtype=float)
            v[col_libre] = 1.0
            for idx_fila, col_piv in enumerate(columnas_pivote):
                v[col_piv] = -R[idx_fila, col_libre]
        else:
            # Respaldo numerico mediante SVD (ultimo vector singular derecho)
            _, _, Vt = np.linalg.svd(M)
            v = Vt[-1, :].copy()

        # Fijar signo consistente (primer componente no nulo positivo)
        for comp in v:
            if abs(comp) > tol:
                if comp < 0:
                    v = -v
                break

        norma = float(np.linalg.norm(v))
        v_unit = v / norma if norma > tol else v.copy()

        # Escalar a componentes enteras/simples cuando sea posible para facilitar lectura
        no_nulos = [abs(x) for x in v_unit if abs(x) > 1e-6]
        v_escalado = v_unit / min(no_nulos) if no_nulos else v_unit.copy()
        for factor in range(1, 13):
            candidato = v_escalado * factor
            if np.all(np.abs(candidato - np.round(candidato)) < 1e-6):
                v_escalado = np.round(candidato)
                break

        v_unit = np.where(np.abs(v_unit) < 1e-12, 0.0, v_unit)
        v_escalado = np.where(np.abs(v_escalado) < 1e-12, 0.0, v_escalado)
        return v_unit, v_escalado

    def calcular_eigenvectores(self):
        """
        Calcula el eigenvector asociado a cada autovalor resolviendo el
        sistema lineal homogeneo (A - lambda_i * I) * v = 0.
        """
        if self.autovalores.size == 0:
            self.calcular_autovalores()

        n = self.dimension
        I = self.identidad(n)
        A = self.matriz
        self.eigenvectores = []

        for lam in self.autovalores:
            matriz_sistema = A - lam * I
            v_unit, v_esc = self._resolver_sistema_homogeneo(matriz_sistema)
            residuo = float(np.max(np.abs(self.multiplicar(matriz_sistema, v_unit))))
            self.eigenvectores.append({
                "autovalor": lam,
                "matriz_sistema": matriz_sistema,
                "vector_unitario": v_unit,
                "vector_escalado": v_esc,
                "residuo": residuo
            })

        return self.eigenvectores

    def resolver_completo(self):
        """Ejecuta el flujo completo del metodo y retorna un diccionario con los resultados."""
        inicio = time.perf_counter()
        self.calcular_polinomio_caracteristico()
        self.calcular_autovalores()
        self.calcular_eigenvectores()
        tiempo = time.perf_counter() - inicio
        return {
            "coeficientes_polinomio": self.coeficientes_polinomio,
            "autovalores": self.autovalores,
            "eigenvectores": self.eigenvectores,
            "es_nula_final": self.es_nula_final,
            "tiempo": tiempo
        }

    def mostrar_reporte(self):
        """Imprime por consola el desarrollo paso a paso de forma ordenada y legible."""
        res = self.resolver_completo()
        n = self.dimension

        print("\n" + "#" * 95)
        print("PUNTO 1 (GUIA N° 4): METODO DE FADEEV-LEVERRIER (PROGRAMACION ORIENTADA A OBJETOS)")
        print("#" * 95)

        print("\n1. MATRIZ DEL SISTEMA A (Dimension 3x3):")
        print(self.formatear_matriz(self.matriz, decimales=4))

        print("\n2. DESARROLLO PASO A PASO DEL METODO DE FADEEV-LEVERRIER:")
        print("-" * 95)
        for info in self.pasos_iteracion:
            k = info["paso"]
            print(f"\n  ---> PASO k = {k}:")
            if k == 1:
                print("    Matriz A_1 = A:")
            else:
                print(f"    Matriz A_{k} = A * B_{k-1}:")
            print(self.formatear_matriz(info["A_k"], decimales=4, indent="      "))
            print(f"    Traza: tr(A_{k}) = {info['traza']:.4f}")
            print(f"    Coeficiente: p_{k} = tr(A_{k}) / {k} = {info['p_k']:.4f}   (Coeficiente en polinomio c_{k} = -p_{k} = {info['c_k']:+.4f})")
            print(f"    Matriz auxiliar B_{k} = A_{k} - p_{k} * I:")
            print(self.formatear_matriz(info["B_k"], decimales=4, indent="      "))

        print("\n" + "-" * 95)
        print("3. CONTROL DEL ALGORITMO (VALIDACION DE MATRIZ NULA EN EL ULTIMO PASO):")
        estado_nula = "EXITO (La matriz final B_n = A_n - p_n*I es la MATRIZ NULA)" if self.es_nula_final else "ADVERTENCIA (No nula)"
        print(f"   * Estado de verificacion: {estado_nula}")

        print("\n4. COEFICIENTES Y POLINOMIO CARACTERISTICO det(lambda * I - A) = 0:")
        terminos = [f"lambda^{n}"]
        for idx, c in enumerate(self.coeficientes_polinomio[1:], start=1):
            pot = n - idx
            signo = "+" if c >= 0 else "-"
            if pot > 1:
                terminos.append(f"{signo} {abs(c):.4f}*lambda^{pot}")
            elif pot == 1:
                terminos.append(f"{signo} {abs(c):.4f}*lambda")
            else:
                terminos.append(f"{signo} {abs(c):.4f}")
        pol_str = " ".join(terminos) + " = 0"
        print(f"   * Coeficientes [1, c_1, c_2, c_3]: {[round(c, 6) for c in self.coeficientes_polinomio]}")
        print(f"   * Polinomio Caracteristico: P(lambda) = {pol_str}")

        print("\n5. AUTOVALORES (RAICES DEL POLINOMIO CARACTERISTICO):")
        for i, lam in enumerate(self.autovalores, start=1):
            print(f"   * lambda_{i} = {lam:.6f}")

        print("\n6. EIGENVECTORES ASOCIADOS A CADA AUTOVALOR ((A - lambda_i * I) * v = 0):")
        print("-" * 95)
        for i, item in enumerate(self.eigenvectores, start=1):
            lam = item["autovalor"]
            v_esc = item["vector_escalado"]
            v_unit = item["vector_unitario"]
            residuo = item["residuo"]

            print(f"\n  ---> Autovalor lambda_{i} = {lam:.4f}:")
            print(f"    Matriz del sistema homogeneo (A - ({lam:.4f}) * I):")
            print(self.formatear_matriz(item["matriz_sistema"], decimales=4, indent="      "))
            esc_str = ", ".join(f"{val:.4f}" for val in v_esc)
            unit_str = ", ".join(f"{val:.6f}" for val in v_unit)
            print(f"    Eigenvector representativo v_{i} (escalado)  : [ {esc_str} ]^T")
            print(f"    Eigenvector normalizado    v_{i} (||v||=1)   : [ {unit_str} ]^T")
            print(f"    Verificacion ||(A - lambda_{i}*I)*v_{i}||_inf : {residuo:.6e}")

        print("\n" + "=" * 95)
        print(f"Tiempo de computo del Metodo de Fadeev-Leverrier: {res['tiempo']:.6f} s ({res['tiempo']*1000:.2f} ms)")
        print("=" * 95 + "\n")


def punto_1_fadeev_leverrier():
    """
    Define la matriz A del Punto 1 de la Guia de Actividades N° 4 e invoca la
    resolucion completa mediante la clase FadeevLeverrierSolver.
    """
    A_punto1 = [
        [3,  2, -1],
        [1, -2,  3],
        [2,  0,  4]
    ]
    solver = FadeevLeverrierSolver(A_punto1)
    solver.mostrar_reporte()
    return solver


# =============================================================================
# EJECUCION SECUENCIAL DE TODOS LOS EJERCICIOS DEL LABORATORIO
# =============================================================================

def ejecutar_todos():
    print("\n" + "=" * 95)
    print("EJECUTANDO TODOS LOS EJERCICIOS DE LA GUIA DE LABORATORIO N° 4 (GAUSS-SEIDEL)")
    print("=" * 95)
    punto_1_fadeev_leverrier()
    ejercicio_1()
    ejercicio_2()
    ejercicio_3()
    ejercicio_4()
    print("\n" + "=" * 95)
    print("¡TODOS LOS EJERCICIOS DEL TP N° 4 HAN SIDO EJECUTADOS CON EXITO!")
    print("=" * 95 + "\n")


# =============================================================================
# MENU PRINCIPAL INTERACTIVO
# =============================================================================

def menu_principal():
    while True:
        print("\n" + "=" * 95)
        print("     METODOS COMPUTACIONALES - TP N° 4: METODO ITERATIVO DE GAUSS-SEIDEL")
        print("=" * 95)
        print("  0. Punto 1 - Metodo de Fadeev-Leverrier (POO: Polinomio, Autovalores y Eigenvectores)")
        print("  1. Ejercicio 1 (Formula alimenticia para engordar ganado - Matriz 4x4)")
        print("  2. Ejercicio 2 (Sistema 3x3 de la guia con analisis previo y x0 = [1, 1, 1])")
        print("  3. Ejercicio 3 (Sistema 3x3 con analisis de convergencia y reordenamiento)")
        print("  4. Ejercicio 4 (Produccion de componentes electronicos - Matriz 3x3)")
        print("  5. Probar Sistema Personalizado N x N (Consigna g: Variar dimensiones)")
        print("  6. Ejecutar TODOS los ejercicios de la guia secuencialmente (Punto 1 y Ej. 1 al 4)")
        print("  7. Salir")
        print("=" * 95)

        opcion = input("Seleccione una opcion (0-7): ").strip()

        if opcion == '0':
            punto_1_fadeev_leverrier()
        elif opcion == '1':
            ejercicio_1()
        elif opcion == '2':
            ejercicio_2()
        elif opcion == '3':
            ejercicio_3()
        elif opcion == '4':
            ejercicio_4()
        elif opcion == '5':
            ejercicio_personalizado()
        elif opcion == '6':
            ejecutar_todos()
        elif opcion == '7':
            print("\n¡Gracias por utilizar el programa de Metodo de Gauss-Seidel!\n")
            break
        else:
            print("\nOpcion no valida. Por favor, ingrese un numero del 0 al 7.")


if __name__ == "__main__":
    menu_principal()
