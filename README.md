# Detector de Fraude con Tarjeta de Crédito — MLOps Demo

## Resumen

Este proyecto implementa un detector de fraude con tarjeta de crédito usando un pipeline MLOps completo. Se combinan **DVC** para el linaje y versionado de datos con **MLflow** para el linaje de experimentos y el registro de modelos. El almacenamiento remoto (datos y artefactos) se delega a **Cloudflare R2**, que expone una API compatible con S3. El dataset proviene de OpenML (id `1597`), equivalente al dataset canónico de Kaggle `creditcard.csv` (~284 000 transacciones, ~0.17 % de fraude).

---

## Arquitectura

```
┌───────────────────────────────────────────────────────────┐
│                      Plano de datos (DVC)                 │
│                                                           │
│  Cloudflare R2 ──► dvc pull  ──► data/raw/*.parquet       │
│  (remoto S3-compatible)          data/processed/*.parquet │
└───────────────────────────────────────────────────────────┘
         │ dvc repro
         ▼
┌───────────────────────────────────────────────────────────┐
│                 Plano de experimentos (MLflow)             │
│                                                           │
│  Servidor MLflow local (file://./mlruns)                  │
│  Artifact store → Cloudflare R2 (s3://luci-mlops-fraud)  │
│                                                           │
│  7 runs  ──► métricas + artefactos ──► Registro de modelos│
└───────────────────────────────────────────────────────────┘
```

**Dos planos, un remoto:**

- **DVC** versiona los datos (parquets) y el manifiesto de runs (`models/runs.json`). El remoto DVC es el bucket R2 configurado como endpoint S3-compatible. `dvc repro` ejecuta el pipeline completo de forma reproducible: si ningún dep cambió, no vuelve a ejecutar.
- **MLflow** versiona los experimentos: parámetros, métricas, artefactos de modelos y plots. El servidor corre localmente (sqlite o `file://`), pero los artefactos pesados (modelos `.pkl`, CSVs de curvas) se almacenan en R2 cuando `MLFLOW_S3_ENDPOINT_URL` está configurado.

---

## Setup local

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd mlops

# 2. Crear y activar virtualenv (Python 3.11 recomendado)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Inicializar DVC
dvc init

# 5. (Opcional) Configurar remoto R2 — ver sección Cloudflare R2
# dvc remote add myremote s3://luci-mlops-fraud
# dvc remote modify myremote endpointurl https://48f381bf59212dbd98d2b424ba4b9a04.r2.cloudflarestorage.com

# 6. Ejecutar el pipeline completo
dvc repro
```

---

## Pipeline DVC

El pipeline tiene cuatro etapas definidas en `dvc.yaml`:

| Etapa | Comando | Se invalida cuando… |
|---|---|---|
| `pull_raw` | `python src/data.py pull` | Cambia `data.openml_dataset_id` en `params.yaml` o se modifica `src/data.py` |
| `preprocess` | `python src/data.py preprocess` | Cambia el output de `pull_raw`, los parámetros de `preprocessing` o `src/data.py` |
| `train` | `python src/train_all.py` | Cambia cualquier split procesado, parámetros de `models` o `mlflow`, o los scripts de entrenamiento |
| `evaluate` | `python src/evaluate.py` | Cambia `models/runs.json` (output de `train`), parámetros de `mlflow` o `src/evaluate.py` |

DVC rastrea los hashes SHA-256 de cada output y sólo re-ejecuta la etapa si algún dep o param cambió.

---

## MLflow

Se ejecutan **7 configuraciones** de entrenamiento bajo el experimento `credit-fraud`. Cada run registra:

- **Parámetros**: tipo de modelo, hiperparámetros, `scale_pos_weight`.
- **Métricas primarias**: `pr_auc` (PR-AUC, métrica principal), `roc_auc`, `best_threshold`, `f1_at_threshold`, `precision_at_threshold`, `recall_at_threshold`.
- **Artefactos**: modelo serializado, matriz de confusión CSV, CSV `y_true/y_prob` para curvas ROC y PR.

### Por qué PR-AUC como métrica primaria

Con ~0.17 % de fraudes, accuracy y ROC-AUC son engañosos (predecir siempre 0 da 99.83 % de accuracy). PR-AUC evalúa precision y recall sobre la clase positiva (fraude), que es la que importa operativamente.

### Registro de modelos (sin etapas deprecadas)

Tras los 7 runs, `register_best.py` busca el run con mayor `pr_auc`, registra el modelo bajo el nombre `fraud-detector` y asigna el alias `staging` usando la API moderna de MLflow:

```python
client.set_registered_model_alias("fraud-detector", "staging", version)
# Carga:
mlflow.pyfunc.load_model("models:/fraud-detector@staging")
```

> Las etapas `Staging`/`Production` de la API antigua están **deprecadas** desde MLflow 2.x y eliminadas en 3.x. Este proyecto usa alias exclusivamente.

---

## Modelos

| # | Run name | Tipo | Hiperparámetros clave |
|---|---|---|---|
| 1 | `logreg-baseline` | LogisticRegression | `C=0.1`, `class_weight=balanced`, `solver=lbfgs` |
| 2 | `xgb-d4-lr10-n300` | XGBoost | `max_depth=4`, `lr=0.1`, `n_estimators=300` |
| 3 | `xgb-d6-lr05-n500` | XGBoost | `max_depth=6`, `lr=0.05`, `n_estimators=500` |
| 4 | `xgb-d8-lr01-n1000` | XGBoost | `max_depth=8`, `lr=0.01`, `n_estimators=1000` |
| 5 | `lgbm-l31-lr10-n300-isunb` | LightGBM | `num_leaves=31`, `lr=0.1`, `n_estimators=300`, `is_unbalance=True` |
| 6 | `lgbm-l63-lr05-n500-spw` | LightGBM | `num_leaves=63`, `lr=0.05`, `n_estimators=500`, `scale_pos_weight` |
| 7 | `lgbm-l127-lr01-n1000-isunb` | LightGBM | `num_leaves=127`, `lr=0.01`, `n_estimators=1000`, `is_unbalance=True` |

Los árboles usan las 29 features directamente (V1–V28 + Amount). LogReg aplica `log1p` + `StandardScaler` sobre Amount y `StandardScaler` sobre V1–V28.

---

## Cloudflare R2 (configuración)

Cloudflare R2 implementa la API S3 estándar. No requiere cambios en el código — sólo configurar el endpoint URL.

**Cuenta:** `48f381bf59212dbd98d2b424ba4b9a04`  
**Bucket destino:** `luci-mlops-fraud`  
**Estado:** pendiente — creación del bucket y token API S3-compatible (otro agente lo aprovisiona).

### Configurar DVC

```bash
dvc remote add myremote s3://luci-mlops-fraud
dvc remote modify myremote endpointurl https://48f381bf59212dbd98d2b424ba4b9a04.r2.cloudflarestorage.com
dvc remote modify myremote access_key_id <R2_ACCESS_KEY_ID>
dvc remote modify myremote secret_access_key <R2_SECRET_ACCESS_KEY>
```

### Configurar MLflow

```bash
export AWS_ACCESS_KEY_ID=<R2_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<R2_SECRET_ACCESS_KEY>
export MLFLOW_S3_ENDPOINT_URL=https://48f381bf59212dbd98d2b424ba4b9a04.r2.cloudflarestorage.com
# Luego arrancar MLflow con artifact-root en R2:
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --default-artifact-root s3://luci-mlops-fraud/mlflow-artifacts
```

---

## Resultados

_Ver `metrics/metrics.json` después de ejecutar `dvc repro`._

El ranking completo por PR-AUC se genera automáticamente al finalizar la etapa `evaluate` y se muestra en consola. El resumen de la ejecución está en `evidence/run_summary.md`.

---

## Autores

- **Eduardo García**
- **Fernando Ramos**

Curso: Integración de Servicios de Aprendizaje Automático, Primavera 2026.
