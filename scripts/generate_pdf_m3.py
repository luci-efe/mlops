#!/usr/bin/env python3
"""M3 supplement PDF — Despliegue del modelo en Cloudflare + Cartel.

Generates `Suplemento_M3_Cloudflare_Despliegue_EduardoGarcia_FernandoRamos.pdf`
documenting the inference deployment (Cloudflare Container + Worker), the
ONNX export, the parity verification, and the poster artifact. Same Adwaita
typography as the previous milestone PDFs.
"""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[2]
MLOPS = ROOT / "mlops"
EVID_M3 = MLOPS / "evidence" / "m3"
OUT_PDF = MLOPS / "Suplemento_M3_Cloudflare_Despliegue_EduardoGarcia_FernandoRamos.pdf"

FONT_R = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
FONT_I = "/usr/share/fonts/Adwaita/AdwaitaSans-Italic.ttf"
FONT_M = "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf"


class PDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.add_font("Main", "", FONT_R)
        self.add_font("Main", "B", FONT_R)
        self.add_font("Main", "I", FONT_I)
        self.add_font("Mono", "", FONT_M)
        self.set_auto_page_break(True, margin=20)

    def header(self) -> None:
        if self.page_no() > 1:
            self.set_font("Main", "", 8)
            self.set_text_color(110, 110, 110)
            self.cell(0, 10, "Suplemento M3 — Despliegue en Cloudflare | ISAA | Primavera 2026",
                      align="C", new_x="LMARGIN", new_y="NEXT")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Main", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def h1(self, text: str) -> None:
        self.set_font("Main", "B", 18)
        self.set_text_color(0, 80, 160)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def h2(self, text: str) -> None:
        self.set_font("Main", "B", 13)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Main", "", 11)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code(self, text: str) -> None:
        self.set_fill_color(245, 245, 248)
        self.set_font("Mono", "", 9)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 4.5, text, fill=True, border=1)
        self.ln(2)


def fmt_json(d: dict) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False)


def main() -> None:
    parity = json.loads((EVID_M3 / "inference_parity.json").read_text())
    onnx_meta = json.loads((EVID_M3 / "onnx_export_parity.json").read_text())
    curl_evidence = json.loads((EVID_M3 / "curl_predictions.json").read_text())
    worker_url = (EVID_M3 / "worker_url.txt").read_text().strip()

    pdf = PDF()
    pdf.add_page()

    # Cover
    pdf.set_font("Main", "B", 22)
    pdf.set_text_color(0, 80, 160)
    pdf.ln(15)
    pdf.cell(0, 12, "Suplemento Milestone 3", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Main", "B", 15)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 9, "Despliegue del modelo en Cloudflare", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 9, "y generación del cartel ITESO", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)
    pdf.set_font("Main", "", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "Eduardo García López  ·  Fernando Ramos Ríos", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, "ITESO — ISAA — Primavera 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.body(
        "Este documento complementa el reporte del Proyecto Final entregado en M2. "
        "Cubre exclusivamente las tareas del Milestone 3: (1) exportación del modelo "
        "registrado fraud-detector@staging a ONNX como artefacto portátil; "
        "(2) construcción y publicación de un contenedor de inferencia en Cloudflare; "
        "(3) verificación empírica de paridad contra el modelo local; (4) generación "
        "del cartel para el Congreso Ingenierías SUJ."
    )

    pdf.add_page()
    pdf.h1("1. Arquitectura de despliegue")
    pdf.body(
        "El modelo XGBoost ganador (xgb-d6-lr05-n500, PR-AUC 0.8819) fue exportado a "
        "ONNX y servido desde un contenedor Docker (python:3.11-slim + FastAPI + "
        "onnxruntime) publicado en el registro privado de Cloudflare. El contenedor "
        "se materializa como un Durable Object detrás de un Worker que actúa de "
        "proxy HTTP, expuesto en una URL pública *.workers.dev. La elección de "
        "Cloudflare en lugar de AWS se debe a la suspensión de la cuenta académica "
        "de uno de los autores; el pivote es parte abierta de la narrativa del "
        "proyecto."
    )
    arch_img = EVID_M3 / "architecture.png"
    if arch_img.exists():
        pdf.image(str(arch_img), x=15, w=180)
    pdf.ln(2)
    pdf.h2("Componentes")
    pdf.body(
        f"• Endpoint: {worker_url}\n"
        "• Worker (TS, ESM): mlops/deployment/src/index.ts — usa @cloudflare/containers, "
        "ruta toda petición a getContainer(env.FRAUD_DETECTOR)\n"
        "• Container (Python): mlops/deployment/container_src/main.py — FastAPI con "
        "POST /predict, GET /health, GET /\n"
        "• Modelo: ONNX 850 KB (opset 15, zipmap=False) horneado dentro de la imagen\n"
        "• Instance type: basic (1/4 vCPU, 1 GiB RAM, 4 GB disco), max 3 instancias\n"
        "• sleepAfter: 5 minutos (la primera petición tras inactividad paga arranque "
        "en frío de ~30 s)"
    )

    pdf.add_page()
    pdf.h1("2. Exportación a ONNX y paridad local")
    pdf.body(
        "El modelo fue cargado vía mlflow.xgboost.load_model('models:/fraud-detector@staging'), "
        "lo que devuelve el XGBClassifier nativo sin necesidad de envolver/desenvolver "
        "pyfunc. Para que onnxruntime acepte los nombres de variables, renombramos "
        "los atributos del booster (V1..V28, Amount) al patrón f0..f28 que exige "
        "onnxmltools. Convertimos con skl2onnx (registrando manualmente el "
        "converter de XGBoost) usando target_opset={\"\": 15, \"ai.onnx.ml\": 3} y "
        "options={id(model): {\"zipmap\": False}}; sin esto, el grafo ONNX habría "
        "incluido un nodo ZipMap incompatible con runtimes WASM/JS."
    )
    pdf.h2("Resultado de la paridad")
    pdf.code(fmt_json({k: v for k, v in onnx_meta.items() if k != "onnx_path"}))
    pdf.body(
        f"Los 200 ejemplos del subconjunto de prueba coincidieron con el modelo "
        f"original dentro de 1×10⁻⁴; la diferencia máxima absoluta fue "
        f"{onnx_meta['max_abs_diff']:.2e}, casi al ruido de punto flotante de "
        f"32 bits. ONNX queda así validado como artefacto portátil; el archivo "
        f"se subió también al bucket R2 luci-mlops-fraud/inference/ para "
        f"trazabilidad."
    )

    pdf.add_page()
    pdf.h1("3. Verificación contra el endpoint en producción")
    pdf.body(
        f"El script scripts/test_inference.py muestrea {parity['n_rows']} filas del "
        f"conjunto de prueba (incluyendo {parity['n_positives_sampled']} fraudes "
        f"reales, sobremuestreados para no quedarnos solo con negativos), las envía "
        f"al endpoint, y compara la probabilidad devuelta contra "
        f"sklearn.predict_proba del mismo modelo cargado localmente. El criterio "
        f"binario es: PASS sólo si las 100 filas caen dentro de tolerancia y la "
        f"decisión de clase (umbral 0.9274) coincide en todas."
    )
    pdf.h2("Métricas de paridad")
    pdf.code(fmt_json(parity))
    pdf.h2("Evidencia muestra (curl_predictions.json)")
    for row in curl_evidence[:3]:
        block = (
            f"row {row['row_index_in_X_test']}  (y_true={row['y_true']})\n"
            f"  POST /predict\n"
            f"    → probability={row['response']['probability']:.6g}\n"
            f"    → prediction={row['response']['prediction']}, "
            f"latency={row['latency_ms']} ms\n"
            f"  sklearn local: {row['sklearn_proba']:.6g}\n"
        )
        pdf.code(block)

    pdf.add_page()
    pdf.h1("4. Cartel para el Congreso Ingenierías SUJ")
    pdf.body(
        "El cartel fue generado programáticamente a partir de la plantilla oficial "
        "2026_Formato_cartel_CongresoIngSUJ.pptx (35.4 × 47.2 in, portrait, una sola "
        "diapositiva) usando python-pptx. El script scripts/generate_cartel.py "
        "respeta la tipografía y colores de la plantilla; sólo reemplaza el texto "
        "Lorem ipsum por contenido en español y agrega las cinco figuras "
        "(arquitectura, leaderboard PR-AUC, curvas PR, comparación de corridas en "
        "MLflow UI, contenido del bucket R2). La conversión a PDF se hace con "
        "libreoffice --headless --convert-to pdf; la vista previa PNG con "
        "pdftoppm a 100 dpi."
    )
    preview = EVID_M3 / "cartel_preview.png"
    if preview.exists():
        pdf.image(str(preview), x=40, w=130)
    pdf.body(
        "Archivos generados: \n"
        "• mlops/Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos.pptx\n"
        "• mlops/Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos.pdf\n"
        "• mlops/evidence/m3/cartel_preview.png"
    )

    pdf.add_page()
    pdf.h1("5. Reproducir desde cero")
    pdf.body(
        "Nota importante: el `account_id` de Cloudflare en `deployment/wrangler.jsonc` "
        "está fijado al de los autores. Para reproducir desde otra cuenta, reemplazar "
        "ese valor y volver a desplegar. Además, la primera petición al endpoint tras "
        "una hora de inactividad puede tardar ~30 s (arranque en frío del contenedor); "
        "peticiones subsecuentes responden en menos de 500 ms."
    )
    pdf.code(
        "# Pre-requisitos: Python 3.11 venv, Docker corriendo, wrangler 4.56+\n"
        "cd mlops\n"
        "source .venv/bin/activate\n"
        "\n"
        "# 1. Exportar a ONNX y verificar paridad local\n"
        "python scripts/export_onnx.py\n"
        "\n"
        "# 2. Subir ONNX a R2 (artefacto portátil)\n"
        "CLOUDFLARE_ACCOUNT_ID=<id> wrangler r2 object put \\\n"
        "  luci-mlops-fraud/inference/fraud-detector-v1.onnx \\\n"
        "  --file models/fraud-detector-v1.onnx --remote\n"
        "\n"
        "# 3. Desplegar contenedor + Worker (build + push + deploy)\n"
        "cd deployment && npm install\n"
        "CLOUDFLARE_ACCOUNT_ID=<id> npx wrangler deploy\n"
        "\n"
        "# 4. Verificar paridad contra el endpoint en vivo\n"
        "WORKER_URL=https://fraud-detector.eduardo-lalo1999.workers.dev \\\n"
        "  python scripts/test_inference.py\n"
        "\n"
        "# 5. Generar cartel + PDF + preview\n"
        "python scripts/generate_cartel.py\n"
    )
    pdf.h2("Comprobación rápida del endpoint")
    pdf.code(
        f"curl -X POST {worker_url}/predict \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"features\":[-0.674,1.408,-1.111,...,23.0]}'\n"
        "\n"
        "# → {\"probability\":1.55e-06,\"prediction\":0,\n"
        "#    \"threshold\":0.9274014830589294,\"model_version\":\"v1\"}"
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
