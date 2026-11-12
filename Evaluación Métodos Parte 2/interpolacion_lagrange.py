"""
===============================================================================
UNIVERSIDAD NACIONAL DEL NORDESTE (UNNE)
FACULTAD DE CIENCIAS EXACTAS Y NATURALES Y AGRIMENSURA (FaCENA)
METODOS COMPUTACIONALES / CALCULO NUMERICO
Modulo: Interpolacion Polinomica de Lagrange (interpolacion_lagrange.py)
===============================================================================
Implementa el calculo del Polinomio Interpolador de Lagrange para estimar
el valor de una funcion en un punto x a partir de una tabla de nodos (x_i, y_i),
midiendo el tiempo de ejecucion con time.perf_counter().
===============================================================================
"""

import time


def interpolacion_lagrange(x_vals, y_vals, x_interp):
    """
    Calcula el valor interpolado utilizando el polinomio de Lagrange:
        P(x) = sum_{i=0}^{n-1} y_i * L_i(x)
    donde:
        L_i(x) = prod_{j=0, j!=i}^{n-1} (x - x_j) / (x_i - x_j)
    """
    n = len(x_vals)
    y_interp = 0.0

    for i in range(n):
        termino = y_vals[i]
        for j in range(n):
            if i != j:
                if x_vals[i] == x_vals[j]:
                    raise ZeroDivisionError(
                        f"Los nodos de interpolacion x[{i}] y x[{j}] son iguales ({x_vals[i]}). "
                        "Los valores de X deben ser distintos entre si."
                    )
                termino *= (x_interp - x_vals[j]) / (x_vals[i] - x_vals[j])
        y_interp += termino

    return y_interp


def ejecutar_modulo_lagrange():
    """
    Despliega la interfaz en consola para el Metodo de Interpolacion de Lagrange,
    permitiendo utilizar los datos de referencia (Censo de poblacion) o ingresar
    una tabla de puntos manualmente por teclado.
    """
    print("\n" + "=" * 90)
    print(f"{'METODO DE INTERPOLACION POLINOMICA DE LAGRANGE':^90}")
    print("=" * 90)
    print("  1. Usar datos de referencia (Censo de poblacion)")
    print("  2. Ingresar datos manualmente por teclado")
    print("-" * 90)

    opcion = input("  Seleccione una opcion (1 o 2): ").strip()

    try:
        if opcion == "1":
            # Datos de referencia: Censo de poblacion
            x_vals = [1940.0, 1950.0, 1960.0, 1970.0, 1980.0, 1990.0]
            y_vals = [132.165, 151.326, 179.323, 203.302, 226.542, 249.633]
            x_interp = 1965.0

            print("\n  [INFO] Usando datos de referencia (Censo de poblacion):")
            print(f"    X (Anos)              : {x_vals}")
            print(f"    Y (Poblacion en miles): {y_vals}")
            print(f"    Valor a interpolar (x): {x_interp}")
        else:
            # Ingreso manual de datos por teclado
            print("\n  --- Ingreso Manual de Datos ---")
            n = int(input("  Ingrese la cantidad de puntos de la tabla (n >= 2): ").strip())
            if n < 2:
                raise ValueError("Se requieren al menos 2 puntos para interpolar.")

            x_vals = []
            y_vals = []

            print("\n  Ingrese los valores de X:")
            for i in range(n):
                val = float(input(f"    x[{i}]: ").strip())
                x_vals.append(val)

            print("\n  Ingrese los valores de Y:")
            for i in range(n):
                val = float(input(f"    y[{i}]: ").strip())
                y_vals.append(val)

            x_interp = float(input("\n  Ingrese el valor de X que desea interpolar: ").strip())

        # Medicion del tiempo de ejecucion utilizando time.perf_counter()
        inicio_tiempo = time.perf_counter()

        resultado = interpolacion_lagrange(x_vals, y_vals, x_interp)

        fin_tiempo = time.perf_counter()
        tiempo_ejecucion = fin_tiempo - inicio_tiempo

        # Salida de resultados con leyendas claras
        print("\n" + "=" * 90)
        print(f"{'RESULTADOS DE LA INTERPOLACION DE LAGRANGE':^90}")
        print("=" * 90)
        print(f"  * Valor de X interpolado : {x_interp}")
        print(f"  * Resultado Y estimado   : {resultado:.6f}")
        print(
            f"  * Tiempo de ejecucion    : {tiempo_ejecucion:.8f} segundos "
            f"({tiempo_ejecucion * 1000:.4f} ms)"
        )
        print("=" * 90 + "\n")

    except (ValueError, ZeroDivisionError) as err:
        print("\n" + "!" * 90)
        print(f"  [ERROR EN INTERPOLACION DE LAGRANGE]: {err}")
        print("!" * 90 + "\n")


if __name__ == "__main__":
    ejecutar_modulo_lagrange()

