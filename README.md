# Validación de MMD² mediante Wild Bootstrap

Comparación de una referencia Monte Carlo bajo H0 con Wild Bootstrap desde un único par de muestras. Se incluyen seis escenarios normales i.i.d. N_d(0,I_d): n=50 y 500 por grupo, d=1, 2 y 5, con semilla 42, 1.000 simulaciones y 1.000 réplicas bootstrap por escenario. El ancho de banda se fija usando el par de referencia de cada escenario.

## Código y procedencia

- `time-series-scripts/prototipo_v3/prototipo_v3.py`: prototipo original de Mario, conservado sin modificaciones.
- `overleaf_validacion_mmd/verificacion/reproducir_multivariado.py`: ejecutor añadido con asistencia de Codex. Lee y ejecuta las cinco funciones originales de conversión, ancho de banda, kernel, MMD² y WB mediante AST; organiza los seis escenarios y genera resultados, figuras y tablas. No implementa un estadístico alternativo.

La carga mediante AST evita ejecutar el gráfico que el prototipo original llama fuera de su bloque principal. Conserva esta estructura de carpetas. El punto de entrada recomendado es el ejecutor, no cambiar DIMENSION en el prototipo: su generación original sigue siendo univariada.

## Reproducción

Entorno de referencia: Python 3.14, NumPy 2.5.1 y Matplotlib 3.11.0. Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
python overleaf_validacion_mmd/verificacion/reproducir_multivariado.py
```

La ejecución vuelve a simular y sobrescribe los seis NPZ, el JSON, las dos figuras y las dos tablas. Conserva los NPZ incluidos: los dos univariados se utilizan también como referencia de comprobación numérica. El JSON registra la ruta del código en el equipo donde se ejecutó; la ruta histórica no se utiliza para localizar el prototipo.

Para regenerar solo figuras y tablas desde los resultados guardados:

```powershell
python overleaf_validacion_mmd/verificacion/reproducir_multivariado.py --figures-only
```

Este modo también actualiza los metadatos del JSON. Ninguno de los comandos recompila el PDF ni actualiza automáticamente la interpretación escrita del informe.

## Informe

El PDF está en `overleaf_validacion_mmd/Informe_validacion_MMD_multivariado.pdf`. Sus fuentes son `main.tex`, `referencias.bib`, las dos tablas `.tex` y los dos PNG de `figuras/`. Para Overleaf, importa el ZIP entregado aparte y compila `main.tex` con pdfLaTeX.

## Alcance

Esta entrega estudia la aproximación de la distribución nula en escenarios i.i.d. univariados y multivariados. No demuestra convergencia asintótica general, validez con dependencia temporal, detección de puntos de cambio ni superioridad respecto de Cn de Xiao. Las dos curvas son aproximaciones obtenidas con simulación finita. La variabilidad entre distintos pares de referencia no se evalúa en esta entrega.
