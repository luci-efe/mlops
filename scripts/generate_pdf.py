#!/usr/bin/env python3
"""Proyecto Final PDF report — MLOps con DVC + MLflow + Cloudflare R2
Authors: Eduardo Garcia, Fernando Ramos
"""
import json
import os
from pathlib import Path

from fpdf import FPDF

ROOT  = Path(__file__).resolve().parent
MLOPS = ROOT / "mlops"
SHOTS = MLOPS / "evidence" / "screenshots"

FONT_R = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
FONT_I = "/usr/share/fonts/Adwaita/AdwaitaSans-Italic.ttf"
FONT_M = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Main", "", FONT_R)
        self.add_font("Main", "B", FONT_R)
        self.add_font("Main", "I", FONT_I)
        self.add_font("Mono", "", FONT_M)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Main", "", 8)
            self.set_text_color(100, 100, 100)
            self.cell(
                0,
                10,
                "Proyecto Final - MLOps DVC + MLflow + R2 | ISAA | Primavera 2026",
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Main", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter(self, title, subtitle=None):
        self.add_page()
        self.set_font("Main", "B", 16)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_font("Main", "I", 10)
            self.set_text_color(110, 110, 110)
            self.cell(0, 6, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section(self, title):
        self.set_font("Main", "B", 12)
        self.set_text_color(0, 102, 204)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Main", "", 10)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def code(self, text, max_lines=None):
        if max_lines:
            lines = text.splitlines()
            if len(lines) > max_lines:
                text = "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines truncated]"
        self.set_font("Mono", "", 7.5)
        self.set_fill_color(245, 245, 248)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 4, text, fill=True)
        self.ln(2)

    def img(self, path, width=170, caption=None):
        if not Path(path).exists():
            self.set_font("Main", "I", 9)
            self.set_text_color(180, 0, 0)
            self.cell(0, 6, f"[missing: {path}]", new_x="LMARGIN", new_y="NEXT")
            return
        self.image(str(path), x=(210 - width) / 2, w=width)
        if caption:
            self.ln(1)
            self.set_font("Main", "I", 9)
            self.set_text_color(110, 110, 110)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def read(p):
    return Path(p).read_text() if Path(p).exists() else f"[missing {p}]"


# ── Load source data ─────────────────────────────────────────────────────────
metrics_raw = json.loads(read(MLOPS / "metrics" / "metrics.json"))
# Sort by pr_auc descending
leaderboard = sorted(metrics_raw.items(), key=lambda kv: kv[1]["pr_auc"] or 0, reverse=True)

params_yaml_text  = read(MLOPS / "params.yaml")
dvc_yaml_text     = read(MLOPS / "dvc.yaml")
dvc_repro_text    = read(MLOPS / "evidence" / "03_dvc_repro.txt")
register_out_text = read(MLOPS / "evidence" / "04_register_best.txt")
smoke_test_text   = read(MLOPS / "evidence" / "05_smoke_test.txt")

# Key snippets from train.py
train_snippet = """\
def train_one(config: dict) -> str:
    \"\"\"Train one config, log to MLflow, return run_id.\"\"\"
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_test  = pd.read_parquet(X_TEST_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH).squeeze()
    y_test  = pd.read_parquet(Y_TEST_PATH).squeeze()

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("learning_rate", config["lr"])
        mlflow.log_param("n_estimators", config["n_estimators"])

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_prob)

        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(k, v)

        signature = infer_signature(input_example, model.predict_proba(input_example))
        mlflow.xgboost.log_model(model, "model", signature=signature,
                                 input_example=input_example)
    return run.info.run_id
"""

evaluate_snippet = """\
# evaluate.py — search_runs pattern (MlflowClient)
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.pr_auc DESC"],
    max_results=1,
)
best = runs[0]
# Then download plot_data artifact for curve reconstruction:
artifact_files = client.list_artifacts(run_id, path="plot_data")
local_dir = client.download_artifacts(run_id, artifact_files[0].path, tmpdir)
plot_df = pd.read_csv(local_dir)
"""

register_snippet = """\
# register_best.py — alias API (MLflow 3.x, stages deprecated)
mv = mlflow.register_model(model_uri, model_name)

# set_registered_model_alias replaces transition_model_version_stage (deprecated)
client.set_registered_model_alias(model_name, "staging", mv.version)

# Load by alias — no magic string "Staging":
model = mlflow.pyfunc.load_model("models:/fraud-detector@staging")
predictions = model.predict(X_test.iloc[:5])
"""


# ── Build PDF ────────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ============================================================
# 1. PORTADA
# ============================================================
pdf.add_page()
pdf.ln(40)
pdf.set_font("Main", "B", 22)
pdf.set_text_color(0, 102, 204)
pdf.multi_cell(0, 13, "Proyecto Final - MLOps con DVC + MLflow + Cloudflare R2", align="C")
pdf.ln(4)
pdf.set_font("Main", "B", 14)
pdf.set_text_color(33, 37, 41)
pdf.multi_cell(0, 9, "Detector de fraude en transacciones de tarjeta de credito", align="C")
pdf.ln(10)
pdf.set_font("Main", "", 12)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "Integracion de Servicios de Aprendizaje Automatico", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Primavera 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(16)
pdf.set_font("Main", "B", 11)
pdf.set_text_color(33, 37, 41)
pdf.cell(0, 8, "Autores:", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Main", "", 11)
pdf.cell(0, 8, "Eduardo Garcia", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Fernando Ramos", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(16)
pdf.set_font("Main", "I", 10)
pdf.set_text_color(128, 128, 128)
pdf.cell(0, 6, "Fecha: 5 de mayo de 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# ============================================================
# 2. TABLA DE CONTENIDOS
# ============================================================
pdf.chapter("Tabla de Contenidos")
toc = [
    ("1.", "Resumen ejecutivo"),
    ("2.", "Contexto y motivacion"),
    ("3.", "Dataset y exploracion"),
    ("4.", "Arquitectura MLOps"),
    ("5.", "Versionado de datos con DVC"),
    ("6.", "Configuracion de Cloudflare R2"),
    ("7.", "Entrenamiento y MLflow"),
    ("8.", "Comparacion de modelos (leaderboard)"),
    ("9.", "Registro de modelo y despliegue"),
    ("10.", "Evidencias"),
    ("11.", "Conclusiones"),
    ("12.", "Referencias"),
]
pdf.set_font("Main", "", 11)
for num, title in toc:
    pdf.set_font("Main", "B", 11)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(14, 8, num, align="L")
    pdf.set_font("Main", "", 11)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 8, title, align="L", new_x="LMARGIN", new_y="NEXT")

# ============================================================
# 3. RESUMEN EJECUTIVO
# ============================================================
pdf.chapter("1. Resumen ejecutivo")
pdf.body(
    "Este proyecto implementa un pipeline MLOps de extremo a extremo para la deteccion "
    "de fraude en transacciones de tarjeta de credito. El problema se aborda sobre el dataset "
    "OpenML 1597 (creditcard), compuesto por 284,807 transacciones reales con una tasa de "
    "fraude del 0.173% (492 casos positivos).\n\n"
    "La solucion combina tres herramientas:\n"
    "- DVC (Data Version Control): versionado reproducible de datos y pipeline de etapas.\n"
    "- MLflow: tracking de experimentos, registro de metricas y model registry con aliases.\n"
    "- Cloudflare R2: almacenamiento S3-compatible como remote backend para artefactos.\n\n"
    "Se entrenaron 7 configuraciones de modelos (1 LogReg, 3 XGBoost, 3 LightGBM). El mejor "
    "modelo fue XGBoost (xgb-d6-lr05-n500) con PR-AUC = 0.8819. Este modelo fue registrado "
    "en el MLflow Model Registry con el alias 'staging', siguiendo la API moderna de MLflow 3 "
    "(set_registered_model_alias), que reemplaza al mecanismo de stages que fue deprecado. "
    "El smoke test confirmo que el modelo cargado por alias produce predicciones validas."
)

# ============================================================
# 4. CONTEXTO Y MOTIVACION
# ============================================================
pdf.chapter("2. Contexto y motivacion")
pdf.body(
    "La deteccion de fraude es un caso paradigmatico de ML en produccion: el modelo debe "
    "actualizarse con nuevos datos periodicamente, compararse contra versiones anteriores y "
    "auditarse ante reclamos de falsos positivos. Sin un sistema MLOps robusto es imposible "
    "responder preguntas basicas: 'con que version del dataset fue entrenado este modelo?' o "
    "'que metricas tenia el modelo en produccion hace tres semanas?'. DVC responde la primera; "
    "MLflow responde la segunda."
)
pdf.section("Por que reproducibilidad importa en fraude")
pdf.body(
    "Un falso negativo (fraude no detectado) tiene costo directo en dinero. Un falso positivo "
    "(transaccion legitima bloqueada) genera friccion con el cliente. Cualquier auditoria "
    "regulatoria o de negocio exige trazar cada decision del modelo a su version exacta de "
    "entrenamiento. Sin reproducibilidad no hay auditoria posible.\n\n"
    "Adicionalmente, en un dataset con 0.17% de positivos es trivial subir las metricas "
    "equivocadas: accuracy del 99.83% la logra cualquier modelo que siempre prediga 'no fraude'. "
    "Necesitamos PR-AUC como metrica primaria y trackear cada run individualmente para no "
    "perdernos de fallas del tipo que presentan los modelos lgbm-l31 y lgbm-l63 en este proyecto."
)
pdf.section("Pivote de AWS Academy a Cloudflare R2")
pdf.body(
    "El plan original contemplaba usar AWS S3 como remote backend para DVC y MLflow. Sin "
    "embargo, las credenciales de AWS Academy tienen duracion de ~4 horas por sesion y no "
    "permiten crear IAM users permanentes, lo que hace imposible automatizar el push/pull "
    "en CI o reproducir el pipeline en otro momento sin re-autenticarse manualmente.\n\n"
    "Cloudflare R2 es un servicio de almacenamiento de objetos completamente compatible con "
    "la API de Amazon S3. Las diferencias relevantes para este proyecto son:\n"
    "- Credenciales permanentes: Access Key ID y Secret Access Key generados desde el "
    "dashboard de R2 sin fecha de expiracion.\n"
    "- Endpoint URL personalizado: https://<account_id>.r2.cloudflarestorage.com (en lugar "
    "del endpoint regional de AWS). Tanto DVC como mlflow.set_tracking_uri() aceptan el "
    "endpoint via variable de entorno MLFLOW_S3_ENDPOINT_URL.\n"
    "- Sin costo de egreso: R2 no cobra por transferencia de datos salientes, ventaja "
    "economica para iteraciones frecuentes de entrenamiento."
)

# ============================================================
# 5. DATASET Y EXPLORACION
# ============================================================
pdf.chapter("3. Dataset y exploracion")
pdf.body(
    "El dataset OpenML 1597 (creditcard) contiene transacciones de tarjeta de credito de "
    "dos dias de septiembre 2013, registradas por bancos europeos. Cada transaccion esta "
    "descrita por 30 atributos:\n\n"
    "- Time: segundos transcurridos desde la primera transaccion del dataset (descartada).\n"
    "- V1 a V28: componentes PCA de las features originales (datos anonimizados por el banco).\n"
    "- Amount: monto de la transaccion en euros.\n"
    "- Class: etiqueta binaria (0 = legitima, 1 = fraude).\n\n"
    "Total de registros: 284,807. Fraudes: 492 (0.173%). El desbalance es de aproximadamente "
    "1:577, lo que exige estrategias especificas:"
)
pdf.section("Decisiones de preprocesamiento")
pdf.body(
    "1. Descarte de Time: la columna Time representa el orden cronologico pero no anade "
    "informacion predictiva util para un modelo batch que no conoce el contexto temporal "
    "de cada transaccion al momento del scoring. Se descarta antes del split.\n\n"
    "2. Transformacion de Amount: Amount tiene distribucion muy sesgada a la derecha "
    "(la mayoria de transacciones son de poco monto, con outliers grandes). Para la "
    "Regresion Logistica aplicamos log1p(Amount) seguido de StandardScaler. Los modelos "
    "de arboles (XGBoost, LightGBM) son invariantes a transformaciones monotonas, por lo "
    "que reciben Amount sin transformar.\n\n"
    "3. Split estratificado: se usa stratify=y en train_test_split con test_size=0.20 y "
    "random_state=42, garantizando que ambos splits mantengan la proporcion 0.173% de "
    "fraudes. Sin esto, un split aleatorio podria producir un test set con cero fraudes.\n\n"
    "4. Manejo del desbalance: LogReg usa class_weight='balanced'. XGBoost recibe "
    "scale_pos_weight = (n_negativos / n_positivos) ~577. LightGBM usa is_unbalance=True "
    "o scale_pos_weight segun la configuracion (el experimento revelo que las dos primeras "
    "configuraciones LGBM colapsaron con estas estrategias)."
)

# ============================================================
# 6. ARQUITECTURA MLOPS
# ============================================================
pdf.chapter("4. Arquitectura MLOps")
pdf.body(
    "La arquitectura del proyecto tiene dos planos complementarios que se intersectan en "
    "params.yaml:"
)
pdf.section("Plano DVC: linaje de datos y pipeline reproducible")
pdf.body(
    "DVC gestiona el grafo de dependencias del pipeline. El archivo dvc.yaml define 4 stages:\n\n"
    "  pull_raw -> preprocess -> train -> evaluate\n\n"
    "Cada stage especifica sus dependencias (deps), parametros (params) y salidas (outs). "
    "DVC computa un hash de cada dep y out; si el hash no cambia, el stage se salta. "
    "Un cambio en params.yaml invalida los stages que dependen de esos parametros, y "
    "dvc repro los reconstruye en orden topologico.\n\n"
    "El archivo dvc.lock registra los hashes exactos de cada stage tras la ultima ejecucion, "
    "funcionando como 'snapshot' reproducible del estado del pipeline. Versionarlo en Git "
    "junto con params.yaml garantiza que cualquier commit puede reproducirse."
)
pdf.section("Plano MLflow: linaje de experimentos")
pdf.body(
    "MLflow tracking server con backend SQLite (mlflow.db). Cada llamada a train_one() "
    "abre un mlflow.start_run() que registra:\n"
    "- Parametros: model_type, learning_rate, n_estimators, max_depth / num_leaves.\n"
    "- Metricas: pr_auc, roc_auc, best_threshold, f1_at_threshold, precision, recall.\n"
    "- Artefactos: modelo serializado (sklearn/xgboost/lightgbm flavor) con signature e "
    "  input_example; CSV de predicciones para reconstruir curvas ROC/PR.\n\n"
    "En la fase de despliegue en nube, se configura MLFLOW_S3_ENDPOINT_URL para que "
    "MLflow use el bucket R2 como artifact store, manteniendo identica la API de tracking."
)
pdf.section("Punto de union: params.yaml")
pdf.body(
    "params.yaml es la fuente de verdad compartida:\n"
    "- DVC lo usa para detectar cambios que invalidan stages (instruccion params: en dvc.yaml).\n"
    "- train_all.py lo lee con yaml.safe_load() y re-envia cada hiperparametro via "
    "mlflow.log_param(), creando el puente auditorio entre el pipeline de datos y el "
    "experimento de ML.\n\n"
    "Cuando un investigador cambia un hiperparametro en params.yaml y ejecuta dvc repro, "
    "el stage train se invalida, se re-entrena con los nuevos parametros, y MLflow "
    "registra un nuevo run con esos exactos valores. No hay riesgo de desincronizacion "
    "entre lo que dice el archivo de configuracion y lo que fue realmente usado."
)

# ============================================================
# 7. VERSIONADO DE DATOS CON DVC
# ============================================================
pdf.chapter("5. Versionado de datos con DVC")
pdf.section("Inicializacion y pipeline")
pdf.body(
    "DVC se inicializa con dvc init sobre el repositorio Git. Esto crea el directorio .dvc/ "
    "con el archivo config (donde se configuran remotes) y un cache local content-addressable. "
    "Los archivos de datos grandes NO entran a Git; solo sus metadatos (.dvc o entradas en "
    "dvc.lock) se versionan. El cache usa una estructura de directorios basada en el hash "
    "MD5 del contenido del archivo."
)
pdf.section("dvc.yaml — definicion del pipeline")
pdf.code(dvc_yaml_text, max_lines=60)

pdf.section("Ejecucion: dvc repro (salida resumida)")
pdf.code(dvc_repro_text, max_lines=50)

# ============================================================
# 8. CONFIGURACION DE CLOUDFLARE R2
# ============================================================
pdf.chapter("6. Configuracion de Cloudflare R2")
pdf.body(
    "Cloudflare R2 expone una API completamente compatible con S3, lo que significa que "
    "cualquier herramienta que soporte S3 (DVC, boto3, el SDK de MLflow) puede usarla "
    "sin modificaciones en el codigo, solo cambiando el endpoint URL."
)
pdf.section("Creacion del bucket")
pdf.code(
    "# Bucket creado con wrangler CLI (ya ejecutado)\n"
    "wrangler r2 bucket create luci-mlops-fraud --location wnam\n\n"
    "# Verificacion\n"
    "wrangler r2 bucket list\n"
    "# -> luci-mlops-fraud   created: 2026-05-05  location: wnam"
)
pdf.section("Modelo de autenticacion")
pdf.body(
    "Las credenciales de R2 se generan en el dashboard de Cloudflare:\n"
    "R2 -> Manage API Tokens -> Create Token con permisos Object Read & Write.\n\n"
    "Esto genera un Access Key ID y un Secret Access Key permanentes (sin TTL), "
    "a diferencia de las credenciales temporales de AWS Academy. Estos valores se "
    "configuran como variables de entorno o en ~/.aws/credentials bajo un profile dedicado.\n\n"
    "El endpoint URL para este account es:\n"
    "  https://<account_id>.r2.cloudflarestorage.com"
)
pdf.section("Configuracion de DVC remote (pendiente de credenciales S3)")
pdf.code(
    "# Configurar remote en DVC apuntando a R2\n"
    "dvc remote add -d r2remote s3://luci-mlops-fraud/dvc-cache\n"
    "dvc remote modify r2remote endpointurl https://<account_id>.r2.cloudflarestorage.com\n"
    "# Credenciales (usar --local para que no entren a Git)\n"
    "dvc remote modify --local r2remote access_key_id     <R2_ACCESS_KEY_ID>\n"
    "dvc remote modify --local r2remote secret_access_key <R2_SECRET_ACCESS_KEY>\n\n"
    "# Subir artefactos al remote\n"
    "dvc push\n\n"
    "# Configurar MLflow artifact store vía R2\n"
    "export MLFLOW_S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com\n"
    "export AWS_ACCESS_KEY_ID=<R2_ACCESS_KEY_ID>\n"
    "export AWS_SECRET_ACCESS_KEY=<R2_SECRET_ACCESS_KEY>\n"
    "mlflow server --backend-store-uri sqlite:///mlflow.db \\\n"
    "              --default-artifact-root s3://luci-mlops-fraud/mlartifacts"
)
pdf.body(
    "NOTA: El bucket luci-mlops-fraud esta creado y accesible. La fase de push a R2 esta "
    "pendiente de configurar el token S3 generado en el dashboard. Una vez wired, el comando "
    "dvc push y el artifact store de MLflow funcionan identicamente a S3 de AWS."
)

# ============================================================
# 9. ENTRENAMIENTO Y MLFLOW
# ============================================================
pdf.chapter("7. Entrenamiento y MLflow")
pdf.section("Hiperparametros del sweep (params.yaml)")
pdf.code(params_yaml_text, max_lines=80)

pdf.section("Metricas registradas por run")
pdf.body(
    "Cada run de MLflow registra las siguientes metricas:\n\n"
    "- pr_auc: Area bajo la curva Precision-Recall. METRICA PRIMARIA.\n"
    "- roc_auc: Area bajo la curva ROC.\n"
    "- best_threshold: umbral que maximiza F1 en el test set.\n"
    "- f1_at_threshold: F1 en el umbral optimo.\n"
    "- precision_at_threshold: precision en el umbral optimo.\n"
    "- recall_at_threshold: recall en el umbral optimo.\n\n"
    "Por que PR-AUC como metrica primaria y no ROC-AUC?\n\n"
    "Con 0.17% de positivos, la clase negativa domina abrumadoramente. ROC-AUC mide la "
    "habilidad del modelo para separar positivos de negativos en TODA la distribucion, "
    "incluyendo la enorme region de verdaderos negativos que es trivialmente facil de "
    "clasificar correctamente. Un modelo que asigna scores bajos a casi todo tendra un "
    "ROC-AUC alto simplemente porque el denominador de la tasa de falsos positivos es "
    "gigante (284,315 negativos).\n\n"
    "PR-AUC solo mide la calidad del modelo en la region que importa al analista de fraude: "
    "cuando el modelo dice 'sospecho fraude', cuantas veces tiene razon (precision), y "
    "de todos los fraudes reales, cuantos detecta (recall). Es por esto que los dos modelos "
    "lgbm-l31 y lgbm-l63 tienen ROC-AUC > 0.88 pero PR-AUC < 0.05: ROC-AUC les da "
    "credito por clasificar bien los 284k negativos, ocultando que casi nunca detectan "
    "correctamente un fraude real."
)
pdf.section("Snippet clave: train.py — train_one()")
pdf.code(train_snippet)

# ============================================================
# 10. LEADERBOARD
# ============================================================
pdf.chapter("8. Comparacion de modelos (leaderboard)")
pdf.body(
    "Los 7 runs ordenados por PR-AUC descendente. Las columnas clave son PR-AUC (metrica "
    "primaria), ROC-AUC y F1 en el umbral optimo de cada modelo."
)
# Render leaderboard table as monospaced code block
header = f"{'#':>2}  {'run_name':<32}  {'model':>5}  {'PR-AUC':>7}  {'ROC-AUC':>8}  {'F1@thr':>7}  {'Prec':>7}  {'Recall':>7}"
sep    = "-" * len(header)
rows   = [header, sep]
for rank, (name, v) in enumerate(leaderboard, 1):
    rows.append(
        f"{rank:>2}  {name:<32}  {v['model']:>5}  {v['pr_auc']:>7.4f}  "
        f"{v['roc_auc']:>8.4f}  {v['f1_at_threshold']:>7.4f}  "
        f"{v['precision_at_threshold']:>7.4f}  {v['recall_at_threshold']:>7.4f}"
    )
pdf.code("\n".join(rows))

pdf.section("Hallazgo pedagogico: colapso de lgbm-l31 y lgbm-l63")
pdf.body(
    "Los dos primeros runs de LightGBM (lgbm-l31-lr10-n300-isunb y lgbm-l63-lr05-n500-spw) "
    "tienen PR-AUC de 0.0368 y 0.0146 respectivamente, mientras que su ROC-AUC es 0.926 y "
    "0.889. Este patron es la prueba pedagogica central del proyecto.\n\n"
    "El colapso se origina en la interaccion entre el desbalance extremo y los parametros de "
    "manejo de clases. lgbm-l31 usa is_unbalance=True con learning_rate=0.1 y num_leaves=31: "
    "la combinacion produce umbrales de decision que se 'pegan' a 1.0 (threshold=1.0 en la "
    "tabla), lo que significa que el modelo necesita certeza absoluta para declarar fraude. "
    "El resultado: detecta casi todos los fraudes (recall=0.888) pero con precision "
    "desastrosa (0.041), generando avalanchas de falsos positivos.\n\n"
    "lgbm-l63-lr05-n500-spw usa scale_pos_weight (is_unbalance=False) con los mismos "
    "sintomas. El tercer run LGBM (lgbm-l127 con 1000 estimadores y learning_rate=0.01) "
    "converge correctamente: PR-AUC=0.8783.\n\n"
    "Sin tracking individual por run en MLflow, este modo de falla seria invisible: el "
    "promedio de PR-AUC de los tres LGBMs seria ~0.31, aparentando performance mediocre "
    "en lugar de revelar que dos configuraciones estan completamente rotas. MLflow permite "
    "diagnosticar el problema a nivel de run, no solo de modelo."
)

# ============================================================
# 11. REGISTRO Y DESPLIEGUE
# ============================================================
pdf.chapter("9. Registro de modelo y despliegue")
pdf.body(
    "Una vez completados los 7 runs, register_best.py selecciona automaticamente el run "
    "con mayor PR-AUC y lo registra en el MLflow Model Registry."
)
pdf.section("API de aliases (MLflow 3.x) vs stages deprecados")
pdf.body(
    "Hasta MLflow 2.x, los modelos se promovian entre estados mediante transition_model_version_stage(), "
    "con stages fijos: None -> Staging -> Production -> Archived. Esta API fue deprecada "
    "en MLflow 3.x y sera removida en una futura version mayor.\n\n"
    "La nueva API usa aliases arbitrarios:\n"
    "  client.set_registered_model_alias(model_name, 'staging', version)\n\n"
    "Ventajas de aliases:\n"
    "- Multiples canarios simultaneos: 'canary-region-1', 'canary-region-2', 'prod', "
    "'staging' pueden coexistir sobre distintas versiones.\n"
    "- Sin transiciones one-at-a-time: promover 'prod' a una nueva version no bloquea "
    "los demas aliases.\n"
    "- Carga idiomatica: mlflow.pyfunc.load_model('models:/fraud-detector@staging') "
    "resuelve automaticamente la version apuntada por el alias."
)
pdf.section("Codigo: register_best.py")
pdf.code(register_snippet)
pdf.section("Salida de register_best.py")
pdf.code(register_out_text)
pdf.section("Smoke test: carga y prediccion por alias")
pdf.code(smoke_test_text)

# ============================================================
# 12. EVIDENCIAS
# ============================================================
pdf.chapter("10. Evidencias")

screenshots = sorted(SHOTS.glob("*.png")) if SHOTS.exists() else []
if screenshots:
    pdf.section("Screenshots de MLflow UI")
    captions = {
        "01_experiments_list.png": "MLflow UI: lista de experimentos — 'credit-fraud' con 7 runs registrados",
        "02_runs_table.png":       "MLflow UI: tabla de runs ordenados, destacando el colapso de lgbm-l31 y lgbm-l63",
    }
    for img_path in screenshots:
        caption = captions.get(img_path.name, img_path.stem.replace("_", " "))
        pdf.img(img_path, width=180, caption=caption)
else:
    pdf.body(
        "Screenshots pendientes — generar tras dvc repro local. "
        "Re-ejecutar generate_proyecto_final_pdf.py una vez que los screenshots esten "
        "disponibles en mlops/evidence/screenshots/"
    )

pdf.section("Salida de dvc repro (ultimas 50 lineas)")
pdf.code(dvc_repro_text, max_lines=50)

pdf.section("Salida de register_best.py")
pdf.code(register_out_text)

pdf.section("Smoke test: predicciones del modelo cargado por alias")
pdf.code(smoke_test_text)

pdf.section("Snippet: evaluate.py — patron MlflowClient.search_runs")
pdf.code(evaluate_snippet)

# ============================================================
# 13. CONCLUSIONES
# ============================================================
pdf.chapter("11. Conclusiones")
pdf.body(
    "El deliverable minimo exigido por el profesor — un modelo entrenado y funcionando con "
    "tracking de experimentos — esta completamente cumplido por el pipeline local:\n\n"
    "- dvc repro ejecuta las 4 etapas (pull_raw -> preprocess -> train -> evaluate) de "
    "forma reproducible y versionada.\n"
    "- Los 7 runs quedan registrados en MLflow con parametros, metricas y artefactos.\n"
    "- El mejor modelo (XGBoost, PR-AUC=0.8819) esta registrado con alias 'staging' y "
    "es cargable y funcional via mlflow.pyfunc.load_model.\n\n"
    "El storage en nube via Cloudflare R2 esta listo arquitectonicamente: el bucket "
    "luci-mlops-fraud esta creado, la configuracion de DVC remote y MLFLOW_S3_ENDPOINT_URL "
    "esta documentada, y el unico paso pendiente es generar el token S3 permanente desde "
    "el dashboard de R2 y exportarlo como variable de entorno. Una vez hecho esto, "
    "dvc push y el artifact store de MLflow quedan operativos sin ningun cambio de codigo.\n\n"
    "La pipeline es completamente reproducible: un clon del repositorio con dvc pull "
    "(cuando R2 este wired) seguido de dvc repro reconstruye metricas identicas, ya que "
    "params.yaml, dvc.lock y el hash del dataset estan versionados en Git.\n\n"
    "Aprendizajes clave del proyecto:\n"
    "- PR-AUC no es opcional cuando el dataset es tan desbalanceado. El colapso de lgbm-l31 "
    "y lgbm-l63 (ROC-AUC=0.89 pero PR-AUC=0.01) es la evidencia mas convincente.\n"
    "- El tracking por run individual en MLflow permite diagnosticar modos de falla que "
    "son invisibles en promedios agregados.\n"
    "- La API de aliases de MLflow 3 es superior a los stages: mas flexible, mas expresiva "
    "y preparada para despliegues multi-canario.\n"
    "- DVC + params.yaml crea un puente auditorio entre el versionado de datos/configuracion "
    "y el tracking de experimentos: toda la cadena es trazable desde el raw data hasta el "
    "modelo registrado."
)

# ============================================================
# 14. REFERENCIAS
# ============================================================
pdf.chapter("12. Referencias")
refs = [
    ("MLflow Model Registry — Aliases API",
     "https://mlflow.org/docs/latest/model-registry.html#model-aliases"),
    ("MLflow Tracking — log_metric, log_param",
     "https://mlflow.org/docs/latest/python_api/mlflow.html"),
    ("DVC Pipelines — dvc.yaml reference",
     "https://dvc.org/doc/user-guide/project-structure/dvcyaml-files"),
    ("DVC Remote Storage — S3 configuration",
     "https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3"),
    ("Cloudflare R2 — S3 API compatibility",
     "https://developers.cloudflare.com/r2/api/s3/api/"),
    ("Cloudflare R2 — Tokens y autenticacion",
     "https://developers.cloudflare.com/r2/api/s3/tokens/"),
    ("OpenML Dataset 1597 — Credit Card Fraud Detection",
     "https://www.openml.org/search?type=data&id=1597"),
    ("XGBoost — scale_pos_weight para datasets desbalanceados",
     "https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html"),
    ("LightGBM — is_unbalance vs scale_pos_weight",
     "https://lightgbm.readthedocs.io/en/latest/Parameters.html#is_unbalance"),
]
for title, url in refs:
    pdf.set_font("Main", "B", 10)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Mono", "", 8)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 5, url, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

# ============================================================
# SAVE
# ============================================================
out_pdf = ROOT / "Tarea_ProyectoFinal_MLflowDVCR2_EduardoGarcia_FernandoRamos.pdf"
pdf.output(str(out_pdf))
print(f"Wrote {out_pdf} ({os.path.getsize(out_pdf):,} bytes)")
