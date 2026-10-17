# Librería de Python y paper para JSS: diseño v01 (2026-10-17)

**Decisión del autor.** Tres productos:
1. Paper A en JASA T&M.
2. Paper B hacia JMLR: teoría de R-tasas, método seguro y benchmark.
3. Una librería de Python con su paper en *Journal of Statistical Software*. Se envía cuando A esté aceptado o en arXiv, porque JSS espera que la metodología esté publicada.

**Nombre provisional:** `imbcal`, libre en PyPI. Otros candidatos libres: calibimb, probimb, calimb, prevaug e imbprob. El nombre final lo decide el autor.

## 1. Propósito y público

**Problema.** Estimar probabilidades bajo desbalance de clases. Las herramientas habituales, como imbalanced-learn, están orientadas a balancear para clasificar. No traen recalibración anidada, criterios con pérdida recalibrada ni diagnósticos de calibración.

**Público.** Modelación de riesgo clínico, finanzas y cualquier aplicación donde se usen probabilidades y no solo etiquetas.

**Mensaje de la librería.**
- El default seguro es ajustar a la prevalencia natural y recalibrar.
- Si se quieren usar datos sintéticos, hay que hacerlo preservando la prevalencia y eligiendo la intensidad con la pérdida recalibrada.
- Hay que diagnosticar cuándo eso puede ayudar.

## 2. API (v0.1, implementada)

| Componente | Contenido |
|---|---|
| `ImbalanceCalibratedClassifier(estimator, strategy, generators, chi_grid, calibration, criterion, selection, n_synthetic, cv, random_state)` | Envoltorio compatible con scikit-learn: `fit`, `predict_proba`, `predict`, `clone` y `get_params`. Atributos: `selected_`, `cv_results_`, `candidate_distance_`, `calibrator_` |
| Estrategias | `"none"` (default), `"weights"` (comparación) y `"augment"` (aumento que preserva la prevalencia) |
| Selección | Pérdida recalibrada con Platt ajustado de forma cruzada sobre las predicciones fuera de fold; `"best"` o `"1se"` (aumenta solo si mejora a χ = 1 en más de 1 EE) |
| Generadores | `SMOTEGenerator(k)`, con k entero o `"sqrt"`; `GaussianGenerator(h)` (bootstrap suavizado); `CallableGenerator(f)`; cualquier objeto con `sample(X_min, m, rng)` |
| Recalibración | `PlattCalibrator`, `IsotonicCalibrator` y `crossfit_calibrated` |
| Métricas | `probability_report` (log-loss, Brier, AUC, pendiente e intercepto de calibración, ECE, media predicha), `calibration_slope_intercept`, `expected_calibration_error` y `paired_comparison` (EE de Nadeau–Bengio) |
| Diagnósticos | `majority_exposure` (heurística Δη del Lema 1 del memo de teoría del Paper B) |

**Pruebas.** 8 pruebas unitarias (pytest), todas pasan. El ejemplo `examples/ejemplo_ist.py` corre de punta a punta con el IST y da resultados coherentes con el benchmark: las tres estrategias quedan dentro de 0,3 % en log-loss.

## 3. Hoja de ruta

- **v0.2**
  - `rebalancing_cost` del Paper A, portando los solvers de punto fijo como módulo interno.
  - Truncación de candidatos a p ≤ Cπ̂ para eventos muy raros (Corolario 7 de las pruebas de R-tasas).
  - Paso de las verificaciones de `sklearn.utils.estimator_checks`.
  - Documentación con Sphinx.
- **v0.3**
  - Módulo de benchmark: particiones repetidas, tablas con z de Nadeau–Bengio y gráficos de calibración.
  - Envoltorios opcionales para generadores profundos (SDV/CTGAN, TVAE).
  - Diagnóstico del conflicto de ancho (R-tasas).
- **v1.0:** API estable, PyPI, conda-forge, integración continua en GitHub, Zenodo y documentación en ReadTheDocs.

## 4. Esquema del paper JSS

Los papers de JSS describen el software con ejemplos reproducibles.

1. **Introducción.** El problema de la calibración bajo desbalance, por qué rebalancear daña (cita a A y B) y la brecha en el software disponible.
2. **Metodología.** Prevalencia natural y recalibración; aumento que preserva la prevalencia; selección por pérdida recalibrada; la regla de 1 EE y su justificación (Proposición 4); diagnósticos.
3. **Diseño de la librería.** API compatible con scikit-learn e interfaz de generadores.
4. **Ilustraciones.**
   - IST (riesgo clínico, desarrollo de 2000);
   - un conjunto de imbalanced-learn con minoritaria compacta;
   - una comparación sistemática en la colección de imbalanced-learn con `paired_comparison`.
5. **Comparación con imbalanced-learn y con `CalibratedClassifierCV` de scikit-learn.**
6. **Discusión y extensiones.**

**Requisitos de JSS a verificar antes del envío:** licencia libre (MIT es compatible con GPL), código de replicación completo y estilo del manuscrito (plantilla jss.cls).

## 5. Relación con los otros productos

- El benchmark del Paper B se reescribirá sobre la librería, y la librería será el código de replicación de B.
- La librería no debe adelantar resultados de B que no estén en arXiv. La regla 1EE y los diagnósticos se documentan como "ver Paper B" cuando B esté en arXiv.
