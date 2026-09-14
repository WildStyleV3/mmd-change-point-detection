import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEMILLA = 42
N_DATOS = 500
N_SIMULACIONES = 1000    # tamaño de la curva verdadera
N_BOOTSTRAP = 1000       # tamaño de la curva bootstrap
DIMENSION = 1
MOSTRAR_GRAFICO = False

# =============================================================================
# 2. MMD: MISMA BASE DEL PROTOTIPO V0
# =============================================================================

def como_matriz(Z):
    """Deja los datos con forma (n, d), tanto si vienen 1D como 2D."""
    Z = np.asarray(Z, dtype=float)
    return Z.reshape(-1, 1) if Z.ndim == 1 else Z


def bandwidth_mediana(X, Y):
    """Heuristica de la mediana: ell = mediana de las distancias entre puntos."""
    Z = np.vstack([como_matriz(X), como_matriz(Y)])
    d2 = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    return np.sqrt(np.median(d2[d2 > 0]) / 2.0)

def rbf_kernel(X, Y, bandwidth=1.0):
    """Kernel gaussiano RBF."""
    distancia_cuadrada = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return np.exp(-distancia_cuadrada / (2.0 * bandwidth**2))


def calculo_mmd2(X, Y, bandwidth=1.0):
    """Estimador sesgado de MMD^2 usado en las diapositivas."""
    X = como_matriz(X)
    Y = como_matriz(Y)

    n = X.shape[0]
    m = Y.shape[0]

    K_XX = rbf_kernel(X, X, bandwidth)
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_XY = rbf_kernel(X, Y, bandwidth)

    termino_XX = np.sum(K_XX) / (n * n)
    termino_YY = np.sum(K_YY) / (m * m)
    termino_XY = 2.0 * np.sum(K_XY) / (n * m)

    return termino_XX + termino_YY - termino_XY


# =============================================================================
# 3. WILD BOOTSTRAP Y TEST DE HIPOTESIS
# =============================================================================

def valores_wild_bootstrap(X, Y, bandwidth, B, rng):
    """Genera B valores de MMD^2 bajo H0 usando pesos Rademacher.

    Cada fila de W_X y W_Y es una repeticion bootstrap. Generarlas juntas
    hace exactamente el mismo calculo que un ciclo, pero tarda mucho menos.
    """
    X = como_matriz(X)
    Y = como_matriz(Y)

    n = X.shape[0]
    m = Y.shape[0]

    K_XX = rbf_kernel(X, X, bandwidth)
    K_YY = rbf_kernel(Y, Y, bandwidth)
    K_XY = rbf_kernel(X, Y, bandwidth)

    # Pesos independientes que toman los valores -1 y 1 con igual probabilidad.
    W_X = rng.choice([-1.0, 1.0], size=(B, n))
    W_Y = rng.choice([-1.0, 1.0], size=(B, m))

    # Centramos los pesos tal como aparece en el algoritmo de la clase.
    W_X = W_X - W_X.mean(axis=1, keepdims=True)
    W_Y = W_Y - W_Y.mean(axis=1, keepdims=True)

    termino_XX = np.einsum(
        "bi,ij,bj->b", W_X, K_XX, W_X, optimize=True
    ) / (n * n)
    termino_YY = np.einsum(
        "bi,ij,bj->b", W_Y, K_YY, W_Y, optimize=True
    ) / (m * m)
    termino_XY = 2.0 * np.einsum(
        "bi,ij,bj->b", W_X, K_XY, W_Y, optimize=True
    ) / (n * m)

    return termino_XX + termino_YY - termino_XY



def comparar_distribuciones():
    rng = np.random.default_rng(SEMILLA)

    X0 = rng.normal(0,1,size = N_DATOS)
    Y0 = rng.normal(0,1,size = N_DATOS)
    ell = bandwidth_mediana(X0,Y0)

    mmd_verdaderos =[]

    for _ in range(N_SIMULACIONES):
        X = rng.normal(0,1,size= N_DATOS)
        Y = rng.normal(0,1,size = N_DATOS)
        mmd_verdaderos.append(calculo_mmd2(X,Y,ell))
    mmd_verdaderos = np.asarray(mmd_verdaderos)
    
    mmd_bootstrap = valores_wild_bootstrap(X0,Y0,ell, B = 1000,rng=rng)
    return mmd_verdaderos,mmd_bootstrap

if __name__ == "__main__":
    verdaderos, bootstrap = comparar_distribuciones()
    print("Datos: ", N_DATOS)
    print(f"verdadera : media {verdaderos.mean():.5f}  p95 {np.quantile(verdaderos, 0.95):.5f}")
    print(f"bootstrap : media {bootstrap.mean():.5f}  p95 {np.quantile(bootstrap, 0.95):.5f}")

def graficar_comparacion(verdaderos, bootstrap):
    carpeta = Path(__file__).resolve().parent
    archivo = carpeta / f"comparacion_wb_d{DIMENSION}_n{N_DATOS}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel izquierdo: densidades superpuestas
    bins = np.histogram_bin_edges(
        np.concatenate([verdaderos, bootstrap]), bins=50
    )
    ax1.hist(verdaderos, bins=bins, density=True, alpha=0.45,
             color="black", label="Distribucion verdadera (simulacion)")
    ax1.hist(bootstrap, bins=bins, density=True, alpha=0.45,
             color="royalblue", label="Aproximacion wild bootstrap")
    ax1.set_xlabel("MMD^2")
    ax1.set_ylabel("Densidad")
    ax1.set_title("Densidades")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel derecho: CDF empiricas
    def cdf(v):
        v = np.sort(v)
        return v, np.arange(1, len(v) + 1) / len(v)

    xv, yv = cdf(verdaderos)
    xb, yb = cdf(bootstrap)
    ax2.plot(xv, yv, color="black", linewidth=2,
             label="Distribucion verdadera (simulacion)")
    ax2.plot(xb, yb, color="royalblue", linewidth=2,
             label="Aproximacion wild bootstrap")
    ax2.axhline(0.95, color="red", linestyle="--", linewidth=1,
                label="Percentil 95 (gamma)")
    ax2.set_xlabel("MMD^2")
    ax2.set_ylabel("F(x)")
    ax2.set_title("Funciones de distribucion acumulada")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Validacion del wild bootstrap bajo H0 — R^{DIMENSION}, n = {N_DATOS}"
    )
    fig.tight_layout()
    fig.savefig(archivo, dpi=180)

    if MOSTRAR_GRAFICO:
        plt.show()
    else:
        plt.close()

    print(f"Grafico guardado en: {archivo}")
plt

graficar_comparacion(verdaderos,bootstrap)