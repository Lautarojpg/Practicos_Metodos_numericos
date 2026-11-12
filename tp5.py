import time

def interpolacion_lagrange(x_vals, y_vals, x_interp):
    """
    Calcula el valor interpolado utilizando el polinomio de Lagrange.
    """
    n = len(x_vals)
    y_interp = 0.0
    
    for i in range(n):
        termino = y_vals[i]
        for j in range(n):
            if i != j:
                termino *= (x_interp - x_vals[j]) / (x_vals[i] - x_vals[j])
        y_interp += termino
        2
    return y_interp

def main():
    print("==================================================")
    print("    MÉTODO DE INTERPOLACIÓN DE LAGRANGE - TP      ")
    print("==================================================")
    print("1. Usar datos de ejemplo del Ejercicio 2 (Censo)")
    print("2. Ingresar datos manualmente por teclado")
    
    opcion = input("Seleccione una opción (1 o 2): ").strip()
    
    if opcion == '1':
        # Datos del Ejercicio 2 de la Guía de Laboratorio 5
        x_vals = [1940, 1950, 1960, 1970, 1980, 1990]
        y_vals = [132.165, 151.326, 179.323, 203.302, 226.542, 249.633]
        x_interp = 1965.0
        
        print("\n[INFO] Usando datos del Ejercicio 2 (Censo de población):")
        print(f"X (Años): {x_vals}")
        print(f"Y (Población en miles): {y_vals}")
        print(f"Valor a interpolar (x): {x_interp}")
    else:
        # Ingreso manual de datos (Punto b de la guía)
        print("\n--- Ingreso de Datos ---")
        n = int(input("Ingrese la cantidad de puntos de la tabla: "))
        
        x_vals = []
        y_vals = []
        
        print("Ingrese los valores de X:")
        for i in range(n):
            val = float(input(f"  x[{i}]: "))
            x_vals.append(val)
            
        print("Ingrese los valores de Y:")
        for i in range(n):
            val = float(input(f"  y[{i}]: "))
            y_vals.append(val)
            
        x_interp = float(input("\nIngrese el valor de X que desea interpolar: "))

    # Medición del tiempo de ejecución (utilizando time.perf_counter)
    inicio_tiempo = time.perf_counter()
    
    resultado = interpolacion_lagrange(x_vals, y_vals, x_interp)
    
    fin_tiempo = time.perf_counter()
    tiempo_ejecucion = fin_tiempo - inicio_tiempo

    # Salida de resultados con leyendas apropiadas (Punto b y comandos de tiempo)
    print("\n=================== RESULTADOS ===================")
    print(f"  Valor de X interpolado : {x_interp}")
    print(f"  Resultado Y estimado   : {resultado:.6f}")
    print(f"  Tiempo de ejecución    : {tiempo_ejecucion:.8f} segundos")
    print("==================================================")

if __name__ == "__main__":
    main()