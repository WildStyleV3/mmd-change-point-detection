"""MMD/WB original de v3; seis escenarios normales i.i.d. en R^d."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT.parent / "time-series-scripts/prototipo_v3/prototipo_v3.py"
NAMES = {"como_matriz", "bandwidth_mediana", "rbf_kernel", "calculo_mmd2", "valores_wild_bootstrap"}
source_bytes = SOURCE.read_bytes()
parsed = ast.parse(source_bytes.decode("utf-8-sig"))
selected = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in NAMES], type_ignores=[])
assert {node.name for node in selected.body} == NAMES
scope = {"np": np}
exec(compile(selected, str(SOURCE), "exec"), scope)
R = B = 1000
rows, samples = [], {}
figures_only = "--figures-only" in sys.argv
if figures_only:
    previous_results = json.loads((HERE / "resultados_multivariados.json").read_text(encoding="utf-8"))
    assert previous_results["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    rows = previous_results["rows"]
    for row in rows:
        saved = np.load(HERE / f"valores_n{row['n']}_d{row['d']}.npz")
        samples[row["n"], row["d"]] = (saved["simulacion"], saved["bootstrap"])
for n in (() if figures_only else (50, 500)):
    for d in (1, 2, 5):
        start = time.perf_counter()
        rng = np.random.default_rng(42)
        X0 = rng.normal(size=(n, d))
        Y0 = rng.normal(size=(n, d))
        ell = scope["bandwidth_mediana"](X0, Y0)
        simulated = np.empty(R)
        for r in range(R):
            X = rng.normal(size=(n, d))
            Y = rng.normal(size=(n, d))
            simulated[r] = scope["calculo_mmd2"](X, Y, ell)
        bootstrap = scope["valores_wild_bootstrap"](X0, Y0, ell, B, rng)
        assert np.isfinite(simulated).all() and np.isfinite(bootstrap).all()
        # Comprobar contra la corrida d=1 verificada, antes de actualizarla.
        if d == 1:
            previous = np.load(HERE / f"valores_n{n}_d1.npz")
            np.testing.assert_allclose(simulated, previous["simulacion"], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(bootstrap, previous["bootstrap"], rtol=1e-12, atol=1e-14)
        q_mc, q_wb = np.quantile(simulated, .95), np.quantile(bootstrap, .95)
        row = {"n": n, "d": d, "ell": float(ell), "media_mc": float(simulated.mean()), "media_wb": float(bootstrap.mean()), "p95_mc": float(q_mc), "p95_wb": float(q_wb), "error_relativo_porcentaje": float(100*(q_wb-q_mc)/q_mc), "segundos": time.perf_counter()-start}
        rows.append(row)
        samples[n, d] = (simulated, bootstrap)
        np.savez_compressed(HERE / f"valores_n{n}_d{d}.npz", simulacion=simulated, bootstrap=bootstrap)
        print(json.dumps(row), flush=True)

result = {"source": str(SOURCE), "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "semilla_por_escenario": 42, "simulaciones": R, "replicas_wb": B, "numpy": np.__version__, "distribucion": "N_d(0,I_d)", "rows": rows}
(HERE / "resultados_multivariados.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 10})
for kind in ("cdf", "histogramas"):
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.6), layout="constrained")
    for i, d in enumerate((1, 2, 5)):
        for j, n in enumerate((50, 500)):
            ax = axes[i, j]
            simulated, bootstrap = samples[n, d]
            bins = np.histogram_bin_edges(np.concatenate([simulated, bootstrap]), bins=40)
            for values, color, label in [(simulated, "#222222", "Referencia Monte Carlo"), (bootstrap, "#3266bb", "Wild bootstrap")]:
                if kind == "cdf":
                    ax.plot(np.sort(values), np.arange(1, len(values)+1)/len(values), color=color, linewidth=1.7, label=label)
                else:
                    ax.hist(values, bins=bins, density=True, color=color, alpha=.48, label=label)
            if kind == "cdf":
                ax.axhline(.95, color="#aa4444", linewidth=.9, linestyle="--", label="Probabilidad 0,95")
            ax.set(title=rf"$d={d}$, $n={n}$ por grupo", xlabel=r"$\widehat{\mathrm{MMD}}^2$", ylabel="Probabilidad acumulada" if kind == "cdf" else "Densidad")
            ax.grid(alpha=.17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=10)
    fig.savefig(ROOT / f"figuras/{kind}_multivariado.png", dpi=220)
    plt.close(fig)

def fmt(x, digits=5):
    return f"{x:.{digits}f}".replace(".", ",")

lines, auxiliary = [], []
for row in rows:
    lines.append(f"{row['n']} & {row['d']} & {fmt(row['p95_mc'])} & {fmt(row['p95_wb'])} & {fmt(row['error_relativo_porcentaje'],2)} " + r"\\")
    auxiliary.append(f"{row['n']} & {row['d']} & {fmt(row['ell'])} & {fmt(row['media_mc'])} & {fmt(row['media_wb'])} " + r"\\")
headers = [r"$n$ & $d$ & $\widehat q_{\mathrm{MC}}$ & $\widehat q_{\mathrm{WB}}$ & $e_{95}$ (\%)\\", r"$n$ & $d$ & $\ell$ & Media MC & Media WB\\"]
for filename, header, body in zip(["tabla_percentiles.tex", "tabla_complementaria.tex"], headers, [lines, auxiliary]):
    table = "\n".join([r"\begin{tabular}{rrrrr}", r"\toprule", header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""])
    (ROOT / filename).write_text(table, encoding="utf-8")
print("Seis escenarios, figuras y tablas terminados.", flush=True)
