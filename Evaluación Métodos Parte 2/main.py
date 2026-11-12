"""
===============================================================================
UNIVERSIDAD NACIONAL DEL NORDESTE (UNNE)
FACULTAD DE CIENCIAS EXACTAS Y NATURALES Y AGRIMENSURA (FaCENA)
METODOS COMPUTACIONALES / CALCULO NUMERICO
Evaluacion Metodos Parte 2 - Programa Principal (main.py)
===============================================================================
Arquitectura modular que integra:
  1. Metodo Directo de Eliminacion de Gauss (eliminacion_gauss.py)
  2. Metodo de Interpolacion Polinomica de Lagrange (interpolacion_lagrange.py)
===============================================================================
"""

import sys
from eliminacion_gauss import EliminacionGaussSolver, SistemaLinealError
from interpolacion_lagrange import ejecutar_modulo_lagrange

# Configuracion de codificacion UTF-8 para consola en Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# ENTRADA DINAMICA DE DATOS POR TECLADO CON GUIA VISUAL DE REFERENCIA (GAUSS)
# =============================================================================

def mostrar_guia_referencia_agronomo():
    """
    Imprime en pantalla un banner visual con los datos del problema de
    referencia del ingeniero agronomo (4 alimentos y 4 nutrientes) para que
    el usuario los tenga a la vista al momento de tipear por teclado.
    """
    print("\n" + "=" * 90)
    print(f"{'GUIA VISUAL DE REFERENCIA: PROBLEMA DEL INGENIERO AGRONOMO (4x4)':^90}")
    print("=" * 90)
    print("  Use estos valores como guia en pantalla si desea probar el ejercicio de la catedra,")
    print("  o ingrese los datos de cualquier otro sistema lineal n x n:")
    print("-" * 90)
    print("  * Tamano de referencia (n) : 4")
    print("  * Matriz A de referencia   :")
    print("      Fila 1 (Carbohidratos) ->   80   15   35   60")
    print("      Fila 2 (Proteinas)     ->   28   72   57   25")
    print("      Fila 3 (Vitaminas)     ->   20   20   12   20")
    print("      Fila 4 (Celulosa)      ->   50   10   20   60")
    print("  * Vector b de referencia   :   230  180   80  160")
    print("=" * 90)


def ingresar_sistema_manual():
    """
    Muestra la guia de referencia visual y solicita interactivamente por teclado:
      1. El tamano del sistema (n).
      2. Los coeficientes de la matriz A (fila por fila o elemento a elemento).
      3. Los terminos independientes del vector b.
    Retorna la matriz A (n x n) y el vector b (n) ingresados por el usuario.
    """
    mostrar_guia_referencia_agronomo()

    print("\n" + "-" * 90)
    print(" >>> INGRESO MANUAL DEL SISTEMA DE ECUACIONES LINEALES (A * x = b)")
    print("-" * 90)

    # 1. Ingreso de la dimension n
    while True:
        try:
            n = int(input("  1. Ingrese el tamano del sistema n (cantidad de ecuaciones/incognitas, n >= 2): ").strip())
            if n < 2:
                print("     [!] Error: El tamano del sistema debe ser al menos 2.")
                continue
            break
        except ValueError:
            print("     [!] Entrada no valida. Ingrese un numero entero (por ejemplo: 4).")

    # 2. Ingreso de los coeficientes de la matriz A (n x n)
    A = []
    print(f"\n  2. Ingrese los coeficientes de la matriz A ({n}x{n}) fila por fila:")
    print(f"     (Puede escribir los {n} numeros de cada fila separados por espacio, o ingresarlos uno por uno)")

    for i in range(n):
        while True:
            entrada = input(f"     Fila {i+1} ({n} coeficientes): ").replace(",", " ").split()
            if len(entrada) == n:
                try:
                    fila = [float(val) for val in entrada]
                    A.append(fila)
                    break
                except ValueError:
                    print("     [!] Error: Todos los coeficientes deben ser valores numericos. Reintente la fila.")
            elif len(entrada) == 1:
                try:
                    fila = [float(entrada[0])]
                    for j in range(1, n):
                        val = float(input(f"       Coeficiente a_{i+1}{j+1}: ").strip())
                        fila.append(val)
                    A.append(fila)
                    break
                except ValueError:
                    print("     [!] Error: Se ingreso un valor no numerico. Reintente la fila completa.")
            else:
                print(f"     [!] Error: Se esperaban {n} coeficientes y se ingresaron {len(entrada)}. Reintente la fila.")

    # 3. Ingreso del vector de terminos independientes b (n)
    b = []
    print(f"\n  3. Ingrese los {n} terminos independientes del vector b:")
    print(f"     (Puede escribir los {n} valores separados por espacio, o ingresarlos uno por uno)")

    while True:
        entrada_b = input(f"     Vector b ({n} valores): ").replace(",", " ").split()
        if len(entrada_b) == n:
            try:
                b = [float(val) for val in entrada_b]
                break
            except ValueError:
                print("     [!] Error: Todos los terminos independientes deben ser numericos. Reintente.")
        elif len(entrada_b) == 1:
            try:
                b = [float(entrada_b[0])]
                for i in range(1, n):
                    val = float(input(f"       Termino independiente b_{i+1}: ").strip())
                    b.append(val)
                break
            except ValueError:
                print("     [!] Error: Se ingreso un valor no numerico. Reintente el vector b.")
        else:
            print(f"     [!] Error: Se esperaban {n} valores y se ingresaron {len(entrada_b)}. Reintente.")

    return A, b


# =============================================================================
# CONTROLADOR DEL MODULO DE ELIMINACION DE GAUSS
# =============================================================================

def ejecutar_modulo_gauss():
    """
    Muestra la guia visual de referencia, solicita los datos del sistema lineal
    A * x = b por teclado y ejecuta su resolucion paso a paso mediante
    el solver generico `EliminacionGaussSolver`.
    """
    try:
        A, b = ingresar_sistema_manual()
        solver = EliminacionGaussSolver(A, b)
        solver.resolver()
        solver.mostrar_procedimiento_completo()
    except (SistemaLinealError, ZeroDivisionError, ValueError) as err:
        print("\n" + "!" * 90)
        print(f"  [ERROR NUMERICO / DE VALIDACION]: {err}")
        print("!" * 90 + "\n")


# =============================================================================
# MENU PRINCIPAL INTERACTIVO
# =============================================================================

def menu_principal():
    """
    Menu interactivo principal que integra Eliminacion de Gauss y Polinomio de Lagrange.
    """
    while True:
        print("\n" + "=" * 90)
        print(f"{'FaCENA - UNNE | EVALUACION METODOS COMPUTACIONALES - PARTE 2':^90}")
        print("=" * 90)
        print("  1. Resolver Sistema de Ecuaciones Lineales - Eliminacion de Gauss")
        print("  2. Resolver Interpolacion - Polinomio de Lagrange")
        print("  3. Salir")
        print("=" * 90)

        try:
            opcion = input("  Seleccione una opcion (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Ejecucion finalizada por el usuario.")
            break

        if opcion == "1":
            ejecutar_modulo_gauss()
        elif opcion == "2":
            ejecutar_modulo_lagrange()
        elif opcion == "3":
            print("\n  ¡Gracias por utilizar el programa! Hasta luego.\n")
            break
        else:
            print("\n  [!] Opcion no valida. Por favor, seleccione 1, 2 o 3.")


if __name__ == "__main__":
    menu_principal()
