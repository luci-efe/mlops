"""Generate the ITESO Congreso Ing. SUJ cartel from the official template.

Replaces text in the 8 named text boxes with Spanish content describing this
project (MLOps pipeline + Cloudflare Container deployment), and inserts figure
images (architecture diagram, PR-AUC leaderboard, PR curves, MLflow UI screenshot,
R2 contents pie, deployed inference response) over the slide.

Output:
  mlops/Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos.pptx
  mlops/evidence/m3/cartel_preview.png  (rendered via LibreOffice)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
MLOPS_ROOT = ROOT / "mlops"
TEMPLATE = ROOT / "2026_Formato_cartel_CongresoIngSUJ.pptx"
OUT_PPTX = MLOPS_ROOT / "Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos.pptx"
EVIDENCE_DIR = MLOPS_ROOT / "evidence" / "m3"
SHOTS_DIR = MLOPS_ROOT / "evidence" / "screenshots"

# ---------------------------------------------------------------------------
# Spanish content
# ---------------------------------------------------------------------------

TITLE = (
    "Detector de fraude con tarjeta de crédito: pipeline MLOps reproducible "
    "con DVC, MLflow y despliegue en Cloudflare"
)

AUTORES = (
    "Eduardo García López y Fernando Ramos\n"
    "ITESO — Integración de Servicios de Aprendizaje Automático (ISAA) — Primavera 2026"
)

INTRO = (
    "El fraude con tarjeta de crédito ocurre en menos del 0.2% de las "
    "transacciones (OpenML 1597, ULB, ~284 807 transacciones, 29 atributos "
    "numéricos): un problema severamente desbalanceado que exige métricas "
    "centradas en la clase positiva, en particular PR-AUC.\n\n"
    "El pivote de AWS a Cloudflare fue forzado por la suspensión de una "
    "cuenta académica, pero refuerza el mensaje MLOps: entrenar en Python, "
    "exportar a ONNX, servir como contenedor en el edge."
)

METODOLOGIA = (
    "Pipeline en cinco etapas, orquestado por DVC sobre Python 3.11:\n\n"
    "1. Ingesta + preprocesamiento (OpenML 1597, partición estratificada 80/20).\n"
    "2. Entrenamiento de 7 corridas (LogReg baseline, LightGBM, XGBoost) variando "
    "hiperparámetros y técnicas de balanceo.\n"
    "3. Registro en MLflow — métricas, artefactos, y alias fraud-detector@staging "
    "sobre el mejor modelo.\n"
    "4. Sincronización a Cloudflare R2 (bucket luci-mlops-fraud, 75 objetos, "
    "~167 MB) vía la API de gestión REST.\n"
    "5. Despliegue: ONNX (opset 15, zipmap=False) en contenedor Docker "
    "(FastAPI + onnxruntime) detrás de Worker + Durable Object."
)

RESULTADOS = (
    "Mejor modelo — xgb-d6-lr05-n500: PR-AUC 0.8819, ROC-AUC 0.9789. "
    "Umbral 0.9274 ⇒ F1 0.874, precisión 0.941, recall 0.816.\n\n"
    "Inferencia en vivo: el endpoint en Cloudflare coincide con sklearn en "
    "100/100 filas dentro de 1×10⁻⁴, latencia mediana 253 ms (p95 613 ms)."
)

CONCLUSIONES = (
    "Un pipeline MLOps disciplinado (DVC + MLflow + registro por alias + "
    "verificación de paridad) lleva un modelo desde notebook hasta endpoint "
    "público sin perder reproducibilidad ni precisión.\n\n"
    "ONNX desacopla el entrenamiento de la plataforma de inferencia; los "
    "Containers de Cloudflare son alternativa creíble a SageMaker para "
    "prototipos académicos.\n\n"
    "Repo: github.com/LaloLalo1999/mlops · "
    "Endpoint: fraud-detector.eduardo-lalo1999.workers.dev"
)

REFERENCIAS = (
    "Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015). "
    "Calibrating probability with undersampling for unbalanced "
    "classification. IEEE SSCI.\n"
    "Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting "
    "system. KDD '16.\n"
    "MLflow Documentation (2026). mlflow.org/docs/latest.\n"
    "DVC Documentation (2026). dvc.org/doc.\n"
    "Cloudflare Workers + Containers (2026). developers.cloudflare.com.\n"
    "ONNX Runtime (2026). onnxruntime.ai/docs."
)

AGRADECIMIENTOS = (
    "A ITESO y al profesor de Integración de Servicios de Aprendizaje "
    "Automático por el marco académico y la flexibilidad para validar "
    "el despliegue en Cloudflare en lugar de AWS. A nuestros compañeros "
    "de ISAA por la retroalimentación durante las tareas previas que "
    "alimentaron este proyecto final."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_text(shape, text: str, font_size: int, *, bold: bool = False,
             align: PP_ALIGN | None = None) -> None:
    tf = shape.text_frame
    tf.clear()
    # Use a single paragraph per logical paragraph; first para already exists.
    paragraphs = text.split("\n")
    for i, raw in enumerate(paragraphs):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align is not None:
            para.alignment = align
        run = para.add_run()
        run.text = raw
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    tf.word_wrap = True


def find_shape(slide, name: str):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    pres = Presentation(TEMPLATE)
    slide = pres.slides[0]

    set_text(find_shape(slide, "Rectangle 180"), TITLE, font_size=44, bold=True,
             align=PP_ALIGN.CENTER)
    set_text(find_shape(slide, "Text Box 14"), AUTORES, font_size=24,
             align=PP_ALIGN.CENTER)
    set_text(find_shape(slide, "Text Box 7"), INTRO, font_size=18)
    set_text(find_shape(slide, "Text Box 12"), METODOLOGIA, font_size=18)
    set_text(find_shape(slide, "Text Box 11"), CONCLUSIONES, font_size=18)
    set_text(find_shape(slide, "Text Box 13"), RESULTADOS, font_size=18)
    set_text(find_shape(slide, "Text Box 15"), REFERENCIAS, font_size=14)
    set_text(find_shape(slide, "Text Box 16"), AGRADECIMIENTOS, font_size=16)

    # Figures: insert as new pictures over white space within / below sections.
    # Coordinates from the template inspection.
    figures = [
        # (image_path, left_in, top_in, width_in) — placed below the section text
        # so they don't overlap. Section text was shortened above to leave room.
        # Slide is 35.4 x 47.2 in.
        # Methodology bottom (Text Box 12 occupies 13.7..33.2, 12.4..28.0)
        (EVIDENCE_DIR / "architecture.png",         14.0, 22.5, 18.7),
        # Resultados bottom (Text Box 13 occupies 13.7..33.1, 29.6..37.8)
        (SHOTS_DIR / "08_pr_auc_leaderboard.png",   14.2, 34.0, 9.0),
        (SHOTS_DIR / "10_pr_curves.png",            23.8, 34.0, 9.0),
        # Intro bottom (Text Box 7 occupies 2.4..12.5, 12.3..24.9)
        (SHOTS_DIR / "03_runs_compare.png",          2.8, 19.6,  9.5),
        # Conclusiones bottom (Text Box 11 occupies 2.4..12.5, 26.7..37.9)
        (SHOTS_DIR / "09_r2_contents_pie.png",       3.5, 33.5,  7.7),
    ]

    for img, left, top, width in figures:
        if not img.exists():
            print(f"  skipping missing image: {img}")
            continue
        pic = slide.shapes.add_picture(str(img), Inches(left), Inches(top),
                                        width=Inches(width))
        # ensure it isn't behind the background image (Imagen 26)
        spTree = pic._element.getparent()
        spTree.remove(pic._element)
        spTree.append(pic._element)
        print(f"  added {img.name}: {pic.width / 914400:.1f}\" x {pic.height / 914400:.1f}\"")

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    pres.save(OUT_PPTX)
    print(f"wrote {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.0f} KB)")

    # Convert to PDF + PNG preview via LibreOffice.
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("Converting to PDF...")
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(MLOPS_ROOT), str(OUT_PPTX),
    ], check=True, timeout=180)
    pdf_path = MLOPS_ROOT / OUT_PPTX.with_suffix(".pdf").name
    print(f"PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")

    print("Rendering preview PNG (150 dpi)...")
    subprocess.run([
        "pdftoppm", "-r", "100", "-singlefile", "-png",
        str(pdf_path), str(EVIDENCE_DIR / "cartel_preview"),
    ], check=True, timeout=120)
    preview = EVIDENCE_DIR / "cartel_preview.png"
    print(f"PNG: {preview} ({preview.stat().st_size / 1024:.0f} KB)")

    summary = {
        "pptx": str(OUT_PPTX.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "preview": str(preview.relative_to(ROOT)),
        "title": TITLE,
        "authors": ["Eduardo García López", "Fernando Ramos"],
    }
    (EVIDENCE_DIR / "cartel_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("OK")


if __name__ == "__main__":
    main()
