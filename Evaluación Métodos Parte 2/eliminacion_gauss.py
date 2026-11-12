"""
===============================================================================
UNIVERSIDAD NACIONAL DEL NORDESTE (UNNE)
FACULTAD DE CIENCIAS EXACTAS Y NATURALES Y AGRIMENSURA (FaCENA)
METODOS COMPUTACIONALES / CALCULO NUMERICO
Modulo: Metodo Directo de Eliminacion de Gauss (Enfoque POO)
===============================================================================
Implementa la clase generica `EliminacionGaussSolver` para resolver cualquier
sistema de ecuaciones lineales cuadrado A * x = b mediante:
  1. Eliminacion hacia adelante (Triangularizacion superior de [A | b])
     con registro paso a paso de multiplicadores y matrices intermedias.
  2. Sustitucion regresiva (hacia atras) paso a paso.
  3. Deteccion de pivotes nulos, sistemas incompatibles o indeterminados.
  4. Medicion del tiempo de ejecucion con time.perf_counter().
===============================================================================
"""

import time


class SistemaLinealError(Exception):
    """Excepcion personalizada para sistemas singulares, incompatibles o mal definidos."""
    pass


class EliminacionGaussSolver:
    """
    Solver generico Orientado a Objetos para resolver sistemas lineales A * x = b
    aplicando el Metodo de Eliminacion de Gauss con sustitucion hacia atras.
    """

    def __init__(self, A, b, tol=1e-12):
        """
        Inicializa el solver validando las dimensiones de la matriz A y el vector b.

        Parametros:
            A   : lista de listas con los coeficientes de la matriz (n x n)
            b   : lista con los terminos independientes (n)
            tol : tolerancia numerica para considerar un pivote como cero
        """
        self._validar_dimensiones(A, b)
        self.n = len(b)
        self.tol = tol
        self.A_original = [[float(val) for val in fila] for fila in A]
        self.b_original = [float(val) for val in b]

        # Matriz ampliada de trabajo [A | b] de dimension n x (n + 1)
        self.matriz_ampliada = [
            fila[:] + [self.b_original[i]] for i, fila in enumerate(self.A_original)
        ]

        # Atributos de resultados
        self.pasos_triangularizacion = []
        self.pasos_sustitucion = []
        self.solucion = []
        self.tiempo_ejecucion = 0.0

    @staticmethod
    def _validar_dimensiones(A, b):
        """Verifica que A sea cuadrada (n x n) y compatible con el vector b (n)."""
        if not A or not b:
            raise ValueError("La matriz A y el vector b no pueden estar vacios.")
        n = len(A)
        if len(b) != n:
            raise ValueError(
                f"Dimensiones incompatibles: A tiene {n} filas pero b tiene {len(b)} elementos."
            )
        for i, fila in enumerate(A):
            if len(fila) != n:
                raise ValueError(
                    f"La matriz A debe ser cuadrada ({n}x{n}). La fila {i+1} tiene {len(fila)} columnas."
                )

    @staticmethod
    def formatear_matriz_ampliada(M, decimales=4, indent="    "):
        """
        Genera una representacion en texto clara y alineada de la matriz ampliada [A | b].
        """
        n = len(M)
        ancho = decimales + 7
        lineas = []
        for i in range(n):
            coefs = " ".join(
                f"{(0.0 if abs(M[i][j]) < 1e-12 else M[i][j]):>{ancho}.{decimales}f}"
                for j in range(n)
            )
            indep = f"{(0.0 if abs(M[i][n]) < 1e-12 else M[i][n]):>{ancho}.{decimales}f}"
            lineas.append(f"{indent}| {coefs}  | {indep} |")
        return "\n".join(lineas)

    def _intercambiar_fila_si_pivote_nulo(self, M, k):
        """
        Si el elemento pivote M[k][k] es nulo (menor o igual a tol), busca una fila
        inferior con coeficiente no nulo en la columna k para evitar division por cero.
        """
        if abs(M[k][k]) > self.tol:
            return None

        for fila_candidata in range(k + 1, self.n):
            if abs(M[fila_candidata][k]) > self.tol:
                M[k], M[fila_candidata] = M[fila_candidata], M[k]
                return (k + 1, fila_candidata + 1)

        # Si todos los elementos debajo del pivote son cero, verificar compatibilidad
        raise SistemaLinealError(
            f"Division por cero / Pivote nulo en el paso k = {k+1} (a_{k+1}{k+1} = 0). "
            "No existe ninguna fila inferior con elemento no nulo para intercambiar. "
            "La matriz es singular (sistema incompatible o con infinitas soluciones)."
        )

    def triangularizar(self):
        """
        Ejecuta la etapa de Eliminacion hacia adelante para llevar la matriz
        ampliada [A | b] a su forma triangular superior, guardando cada paso.
        """
        M = [fila[:] for fila in self.matriz_ampliada]
        self.pasos_triangularizacion = []

        for k in range(self.n - 1):
            intercambio = self._intercambiar_fila_si_pivote_nulo(M, k)
            pivote = M[k][k]
            operaciones_etapa = []

            for i in range(k + 1, self.n):
                factor = M[i][k] / pivote
                operaciones_etapa.append({
                    "fila_modificada": i + 1,
                    "fila_pivote": k + 1,
                    "multiplicador": factor,
                    "expresion": f"F_{i+1} <- F_{i+1} - ({factor:.6f}) * F_{k+1}"
                })

                # Se anula explicitamente la posicion debajo del pivote y se actualiza el resto de la fila
                M[i][k] = 0.0
                for j in range(k + 1, self.n + 1):
                    M[i][j] = M[i][j] - factor * M[k][j]

            self.pasos_triangularizacion.append({
                "etapa": k + 1,
                "pivote_pos": (k + 1, k + 1),
                "pivote_valor": pivote,
                "intercambio": intercambio,
                "operaciones": operaciones_etapa,
                "matriz_estado": [fila[:] for fila in M]
            })

        # Control del ultimo elemento diagonal U[n-1][n-1]
        ultimo_pivote = M[self.n - 1][self.n - 1]
        termino_indep_final = M[self.n - 1][self.n]
        if abs(ultimo_pivote) <= self.tol:
            if abs(termino_indep_final) > self.tol:
                raise SistemaLinealError(
                    "Sistema Incompatible (SI): Se obtuvo una fila de coeficientes nulos "
                    f"igualada a un termino distinto de cero ({termino_indep_final:.6f} != 0)."
                )
            raise SistemaLinealError(
                "Sistema Compatible Indeterminado (SCI): Se obtuvo una fila completamente nula (0 = 0). "
                "El sistema posee infinitas soluciones."
            )

        self.matriz_triangular = M
        return M

    def sustitucion_hacia_atras(self):
        """
        Ejecuta la etapa de Sustitucion Regresiva a partir de la matriz triangular superior.
        """
        M = self.matriz_triangular
        x = [0.0] * self.n
        self.pasos_sustitucion = []

        for i in range(self.n - 1, -1, -1):
            suma_conocidos = sum(M[i][j] * x[j] for j in range(i + 1, self.n))
            pivote = M[i][i]
            if abs(pivote) <= self.tol:
                raise ZeroDivisionError(
                    f"Error de division por cero en sustitucion regresiva: a_{i+1}{i+1} = 0."
                )

            x[i] = (M[i][self.n] - suma_conocidos) / pivote
            self.pasos_sustitucion.append({
                "variable": f"x_{i+1}",
                "indice": i + 1,
                "termino_indep": M[i][self.n],
                "suma_conocidos": suma_conocidos,
                "pivote": pivote,
                "valor": x[i],
                "formula": f"x_{i+1} = ({M[i][self.n]:.6f} - ({suma_conocidos:.6f})) / ({pivote:.6f}) = {x[i]:.6f}"
            })

        self.solucion = x
        return x

    def resolver(self):
        """
        Ejecuta el algoritmo completo de Eliminacion de Gauss midiendo el tiempo
        de procesamiento con time.perf_counter().
        """
        inicio = time.perf_counter()
        self.triangularizar()
        self.sustitucion_hacia_atras()
        fin = time.perf_counter()

        self.tiempo_ejecucion = fin - inicio
        return self.solucion

    def calcular_residuos(self):
        """
        Calcula el vector verificacion A * x y el error residual |A * x - b|.
        """
        ax = []
        residuos = []
        for i in range(self.n):
            prod = sum(self.A_original[i][j] * self.solucion[j] for j in range(self.n))
            ax.append(prod)
            residuos.append(abs(prod - self.b_original[i]))
        return ax, residuos

    def mostrar_procedimiento_completo(self):
        """
        Imprime por consola todo el desarrollo paso a paso de la Eliminacion de Gauss:
          1. Matriz ampliada inicial [A | b].
          2. Etapas de triangularizacion con multiplicadores y matrices intermedias.
          3. Sustitucion hacia atras paso a paso.
          4. Vector solucion final, verificacion y tiempo de ejecucion.
        """
        if not self.solucion:
            self.resolver()

        print("\n" + "=" * 90)
        print(f"{'DESARROLLO PASO A PASO - METODO DE ELIMINACION DE GAUSS':^90}")
        print("=" * 90)

        print(f"\n1. MATRIZ AMPLIADA INICIAL [A | b] (Dimension {self.n} x {self.n + 1}):")
        print("-" * 90)
        print(self.formatear_matriz_ampliada(self.matriz_ampliada))

        print("\n2. ETAPA DE ELIMINACION HACIA ADELANTE (TRIANGULARIZACION):")
        print("-" * 90)
        for paso in self.pasos_triangularizacion:
            etapa = paso["etapa"]
            f_piv, c_piv = paso["pivote_pos"]
            val_piv = paso["pivote_valor"]

            print(f"\n  ---> PASO {etapa} (Columna {etapa} - Pivote a_{f_piv}{c_piv} = {val_piv:.6f}):")
            if paso["intercambio"] is not None:
                f1, f2 = paso["intercambio"]
                print(f"    [!] Intercambio de filas por pivote nulo: Fila {f1} <-> Fila {f2}")

            print("    Operaciones elementales de fila y multiplicadores:")
            for op in paso["operaciones"]:
                print(
                    f"      * m_{op['fila_modificada']}{op['fila_pivote']} = {op['multiplicador']:>10.6f}  |  {op['expresion']}"
                )

            print(f"\n    Matriz ampliada resultante al finalizar el Paso {etapa}:")
            print(self.formatear_matriz_ampliada(paso["matriz_estado"]))

        print("\n" + "-" * 90)
        print("3. MATRIZ TRIANGULAR SUPERIOR FINAL [U | b']:")
        print("-" * 90)
        print(self.formatear_matriz_ampliada(self.matriz_triangular))

        print("\n" + "-" * 90)
        print("4. ETAPA DE SUSTITUCION HACIA ATRAS (REGRESIVA):")
        print("-" * 90)
        for s in self.pasos_sustitucion:
            print(f"    * {s['formula']}")

        ax, residuos = self.calcular_residuos()
        max_residuo = max(residuos) if residuos else 0.0

        print("\n" + "=" * 90)
        print(f"{'RESULTADOS FINALES DEL SISTEMA LINEAL':^90}")
        print("=" * 90)
        print("  Vector Solucion x:")
        for i, val in enumerate(self.solucion):
            print(
                f"    * x_{i+1:<2} = {val:>12.6f}   "
                f"(Verificacion Fila {i+1}: A*x = {ax[i]:>10.6f} vs b = {self.b_original[i]:>10.6f} | Error = {residuos[i]:.2e})"
            )

        print("-" * 90)
        print(f"  * Residuo maximo ||A*x - b||_inf : {max_residuo:.6e}")
        print(
            f"  * Tiempo de ejecucion (perf_counter): {self.tiempo_ejecucion:.8f} segundos "
            f"({self.tiempo_ejecucion * 1000:.4f} ms)"
        )
        print("=" * 90 + "\n")

