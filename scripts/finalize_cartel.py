#!/usr/bin/env python3
"""Final poster/cartel iteration.

This script does two things without requiring python-pptx:
1. Produces a fixed, print-ready PDF that preserves the current Congreso Ing. SUJ
   visual template, but redraws the content panels so the information is readable.
2. Produces a fixed PPTX by XML-editing the current PPTX: section text is corrected,
   oversized figures are repositioned, and cluttering figures are moved off-canvas.

The PDF is the canonical final cartel deliverable. The PPTX is kept editable.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import wrap

ROOT = Path(__file__).resolve().parents[1]
PPTX_IN = ROOT / "Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos.pptx"
PPTX_OUT = ROOT / "Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos_FINAL.pptx"
PDF_OUT = ROOT / "Cartel_ProyectoFinal_EduardoGarcia_FernandoRamos_FINAL.pdf"
PREVIEW = ROOT / "evidence" / "m3" / "cartel_final_preview.jpg"
WORK = ROOT / "evidence" / "m3" / "cartel_assets"
BACKGROUND = ROOT / "evidence" / "m3" / "cartel_preview.png"
IMAGES = {
    "architecture": ROOT / "evidence" / "m3" / "architecture.png",
    "leaderboard": ROOT / "evidence" / "screenshots" / "08_pr_auc_leaderboard.png",
    "pr_curves": ROOT / "evidence" / "screenshots" / "10_pr_curves.png",
}

# Poster size from the existing PPTX/template, in inches.
W_IN, H_IN = 35.43, 47.24
PT = 72.0
W_PT, H_PT = W_IN * PT, H_IN * PT
CREAM = (242/255, 239/255, 230/255)
BLUE = (0/255, 61/255, 96/255)
DARK = (28/255, 28/255, 28/255)
MUTED = (85/255, 85/255, 85/255)
LIGHT_BLUE = (229/255, 240/255, 247/255)
GREEN = (224/255, 243/255, 229/255)
RED = (255/255, 232/255, 232/255)

TITLE = "Detector de fraude con tarjeta de credito: pipeline MLOps reproducible con DVC, MLflow y despliegue en Cloudflare"
AUTHORS = "Eduardo Garcia Lopez y Fernando Ramos"
COURSE = "ITESO - Integracion de Servicios de Aprendizaje Automatico (ISAA) - Primavera 2026"

INTRO = (
    "El fraude con tarjeta de credito es un problema extremadamente desbalanceado: en OpenML 1597 hay cerca de "
    "284,807 transacciones y menos de 0.2% son fraude. Por eso se optimizo PR-AUC y no accuracy.\n\n"
    "Objetivo: construir un flujo reproducible que lleve el modelo desde datos versionados hasta una API publica de inferencia."
)

RESULTS = (
    "Mejor modelo: XGBoost xgb-d6-lr05-n500. PR-AUC 0.8819, ROC-AUC 0.9789. "
    "Umbral operativo 0.9274: F1 0.874, precision 0.941 y recall 0.816.\n\n"
    "Validacion de despliegue: el endpoint de Cloudflare coincide con sklearn en 100/100 filas dentro de tolerancia 1e-4. "
    "Latencia mediana 253 ms; p95 613 ms."
)

METHODS = (
    "1. Ingesta y preprocesamiento con DVC.\n"
    "2. Entrenamiento de 7 corridas en MLflow: Logistic Regression, XGBoost y LightGBM.\n"
    "3. Registro del mejor modelo como fraud-detector@staging.\n"
    "4. Exportacion a ONNX para desacoplar entrenamiento e inferencia.\n"
    "5. Servicio FastAPI + onnxruntime en Cloudflare Containers detras de Worker y Durable Object."
)

CONCLUSIONS = (
    "DVC + MLflow + registro por alias permitieron reproducibilidad de datos, experimentos y artefactos. "
    "ONNX mantuvo paridad numerica entre entrenamiento e inferencia.\n\n"
    "El proyecto demuestra una alternativa academica viable a servicios como SageMaker: un modelo entrenado localmente, "
    "versionado, publicado en R2 y servido desde el edge con una API publica.\n\n"
    "API: fraud-detector.eduardo-lalo1999.workers.dev"
)

REFS = (
    "Dal Pozzolo et al. (2015), IEEE SSCI.  Chen & Guestrin (2016), KDD.  "
    "MLflow, DVC, ONNX Runtime y Cloudflare Workers/Containers docs (2026)."
)
ACK = (
    "A ITESO y al profesor de ISAA por el marco academico y la flexibilidad para validar el despliegue en Cloudflare."
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, timeout=180)


def convert_to_jpeg(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    run(["sips", "-s", "format", "jpeg", str(src), "--out", str(dest)])
    return dest


def jpg_size(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)], text=True)
    w = int(re.search(r"pixelWidth:\s*(\d+)", out).group(1))
    h = int(re.search(r"pixelHeight:\s*(\d+)", out).group(1))
    return w, h


def esc_pdf_text(s: str) -> str:
    b = s.encode("latin-1", "replace").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b.decode("latin-1")


class PDF:
    def __init__(self) -> None:
        self.objects: list[bytes] = []
        self.pages: list[int] = []
        self.images: dict[str, tuple[int, int, int]] = {}
        self.content: list[str] = []

    def add_obj(self, data: bytes) -> int:
        self.objects.append(data)
        return len(self.objects)

    def image(self, name: str, path: Path) -> None:
        data = path.read_bytes()
        w, h = jpg_size(path)
        obj = self.add_obj(
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n".encode("latin-1")
            + data
            + b"\nendstream"
        )
        self.images[name] = (obj, w, h)

    def rgb(self, c: tuple[float, float, float]) -> str:
        return f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}"

    def rect(self, x: float, y_top: float, w: float, h: float, fill: tuple[float,float,float], stroke: tuple[float,float,float]|None=None, lw: float=1.0) -> None:
        xpt, ypt, wpt, hpt = x*PT, H_PT - (y_top+h)*PT, w*PT, h*PT
        self.content.append(f"q {self.rgb(fill)} rg {xpt:.2f} {ypt:.2f} {wpt:.2f} {hpt:.2f} re f Q")
        if stroke:
            self.content.append(f"q {self.rgb(stroke)} RG {lw:.2f} w {xpt:.2f} {ypt:.2f} {wpt:.2f} {hpt:.2f} re S Q")

    def text(self, s: str, x: float, y_top: float, size: float=14, color: tuple[float,float,float]=DARK, font: str="F1", align: str="left", max_w: float|None=None, leading: float|None=None) -> float:
        leading = leading or size*1.22
        lines: list[str] = []
        for para in s.split("\n"):
            if para == "":
                lines.append("")
                continue
            if max_w:
                chars = max(8, int(max_w*72/(size*0.48)))
                lines.extend(wrap(para, width=chars, break_long_words=False))
            else:
                lines.append(para)
        y = y_top
        for line in lines:
            if line == "":
                y += leading/72*0.65
                continue
            approx_w = len(line)*size*0.48/72
            tx = x
            if align == "center" and max_w:
                tx = x + max(0, (max_w-approx_w)/2)
            elif align == "right" and max_w:
                tx = x + max(0, max_w-approx_w)
            self.content.append(f"BT {self.rgb(color)} rg /{font} {size:.2f} Tf {tx*PT:.2f} {H_PT-y*PT:.2f} Td ({esc_pdf_text(line)}) Tj ET")
            y += leading/72
        return y

    def draw_image(self, name: str, x: float, y_top: float, w: float, h: float|None=None) -> None:
        _, iw, ih = self.images[name]
        if h is None:
            h = w * ih / iw
        self.content.append(f"q {w*PT:.2f} 0 0 {h*PT:.2f} {x*PT:.2f} {H_PT-(y_top+h)*PT:.2f} cm /{name} Do Q")

    def panel(self, title: str, x: float, y: float, w: float, h: float) -> None:
        self.rect(x, y, w, 1.08, BLUE)
        self.text(title, x, y+0.28, 24, (1,1,1), "F2", "center", w)
        self.rect(x, y+1.08, w, h-1.08, (1,1,1), BLUE, 1.4)

    def save(self, path: Path) -> None:
        font1 = self.add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font2 = self.add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        contents = "\n".join(self.content).encode("latin-1", "replace")
        cont_obj = self.add_obj(f"<< /Length {len(contents)} >>\nstream\n".encode()+contents+b"\nendstream")
        xobjs = " ".join(f"/{name} {obj} 0 R" for name,(obj,_,_) in self.images.items())
        resources = f"<< /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> /XObject << {xobjs} >> >>"
        page_obj = self.add_obj(f"<< /Type /Page /Parent PAGES 0 R /MediaBox [0 0 {W_PT:.2f} {H_PT:.2f}] /Resources {resources} /Contents {cont_obj} 0 R >>".encode())
        pages_obj = self.add_obj(f"<< /Type /Pages /Kids [{page_obj} 0 R] /Count 1 >>".encode())
        catalog_obj = self.add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode())
        # replace placeholder
        self.objects[page_obj-1] = self.objects[page_obj-1].replace(b"PAGES 0 R", f"{pages_obj} 0 R".encode())
        offsets=[]
        out=bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for i,obj in enumerate(self.objects, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode()+obj+b"\nendobj\n")
        xref=len(out)
        out.extend(f"xref\n0 {len(self.objects)+1}\n0000000000 65535 f \n".encode())
        for off in offsets:
            out.extend(f"{off:010d} 00000 n \n".encode())
        out.extend(f"trailer << /Size {len(self.objects)+1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        path.write_bytes(out)


def make_pdf() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    bg = convert_to_jpeg(BACKGROUND, WORK / "template_background.jpg")
    arch = convert_to_jpeg(IMAGES["architecture"], WORK / "architecture.jpg")
    leader = convert_to_jpeg(IMAGES["leaderboard"], WORK / "leaderboard.jpg")
    pr = convert_to_jpeg(IMAGES["pr_curves"], WORK / "pr_curves.jpg")

    pdf = PDF()
    pdf.image("Bg", bg)
    pdf.image("Arch", arch)
    pdf.image("Leader", leader)
    pdf.image("PR", pr)
    pdf.draw_image("Bg", 0, 0, W_IN, H_IN)

    # Hide the broken old content, preserving the official green header/footer and logos.
    pdf.rect(0, 5.25, W_IN, 39.75, CREAM)
    pdf.text(TITLE, 2.5, 6.18, 34, DARK, "F2", "center", 30.5, 39)
    pdf.text(AUTHORS, 2.5, 9.75, 17, DARK, "F1", "center", 30.5)
    pdf.text(COURSE, 2.5, 10.32, 14.5, DARK, "F1", "center", 30.5)

    pdf.panel("INTRODUCCION", 2.35, 11.45, 10.15, 12.6)
    pdf.text(INTRO, 3.05, 13.1, 17.2, DARK, "F1", "left", 8.75, 21)
    # Three key-value cards
    card_y = 20.35
    for i,(label,val,col) in enumerate([
        ("Dataset", "OpenML 1597", LIGHT_BLUE),
        ("Clase positiva", "< 0.2%", RED),
        ("Metrica clave", "PR-AUC", GREEN),
    ]):
        pdf.rect(3.0, card_y+i*0.95, 8.8, 0.72, col, BLUE, 0.8)
        pdf.text(label, 3.25, card_y+0.22+i*0.95, 10.5, MUTED, "F2")
        pdf.text(val, 7.4, card_y+0.22+i*0.95, 10.5, DARK, "F2")

    pdf.panel("RESULTADOS", 13.72, 11.45, 19.49, 16.25)
    pdf.text(RESULTS, 14.55, 13.15, 17.0, DARK, "F1", "left", 17.85, 21)
    pdf.draw_image("Leader", 14.45, 20.55, 8.8)
    pdf.draw_image("PR", 23.65, 20.40, 8.2)

    pdf.panel("MATERIALES Y METODOS", 2.35, 25.45, 10.15, 11.65)
    pdf.text(METHODS, 3.0, 27.15, 13.8, DARK, "F1", "left", 8.85, 16.5)
    pdf.draw_image("Arch", 3.0, 32.35, 8.9)

    pdf.panel("CONCLUSIONES", 13.72, 28.40, 19.49, 8.7)
    pdf.text(CONCLUSIONS, 14.55, 30.1, 16.0, DARK, "F1", "left", 17.85, 20)

    pdf.panel("REFERENCIAS", 2.35, 38.25, 16.05, 5.5)
    pdf.text(REFS, 3.0, 39.95, 11.4, DARK, "F1", "left", 14.7, 14)
    pdf.panel("AGRADECIMIENTOS", 19.45, 38.25, 13.75, 5.5)
    pdf.text(ACK, 20.1, 39.95, 11.4, DARK, "F1", "left", 12.4, 14)

    pdf.save(PDF_OUT)
    print(f"Wrote {PDF_OUT} ({PDF_OUT.stat().st_size/1024:.0f} KB)")
    # QuickLook thumbnail for QA if available.
    try:
        outdir = WORK / "ql"
        if outdir.exists(): shutil.rmtree(outdir)
        outdir.mkdir(parents=True)
        run(["qlmanage", "-t", "-s", "1800", "-o", str(outdir), str(PDF_OUT)])
        thumbs = list(outdir.glob("*.png")) + list(outdir.glob("*.jpg"))
        if thumbs:
            shutil.copy2(thumbs[0], PREVIEW)
            print(f"Preview {PREVIEW}")
    except Exception as e:
        print(f"Preview generation skipped: {e}")


def set_shape_text(sp: ET.Element, text: str, size: int = 1200, bold: bool = False) -> None:
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    tx_body = sp.find("a:txBody", ns)
    if tx_body is None:
        return
    # Remove old paragraphs.
    for p in list(tx_body.findall("a:p", ns)):
        tx_body.remove(p)
    for para_text in text.split("\n"):
        p = ET.SubElement(tx_body, "{http://schemas.openxmlformats.org/drawingml/2006/main}p")
        r = ET.SubElement(p, "{http://schemas.openxmlformats.org/drawingml/2006/main}r")
        rpr = ET.SubElement(r, "{http://schemas.openxmlformats.org/drawingml/2006/main}rPr", {"lang": "es-MX", "sz": str(size)})
        if bold:
            rpr.set("b", "1")
        solid = ET.SubElement(rpr, "{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill")
        ET.SubElement(solid, "{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr", {"val": "1C1C1C"})
        t = ET.SubElement(r, "{http://schemas.openxmlformats.org/drawingml/2006/main}t")
        t.text = para_text


def edit_pptx() -> None:
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    ET.register_namespace("p", ns["p"])
    ET.register_namespace("a", ns["a"])
    ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    with zipfile.ZipFile(PPTX_IN) as zin:
        xml = zin.read("ppt/slides/slide1.xml")
        root = ET.fromstring(xml)
        for sp in root.findall(".//p:sp", ns):
            cnv = sp.find("p:nvSpPr/p:cNvPr", ns)
            if cnv is None: continue
            name, sid = cnv.attrib.get("name"), cnv.attrib.get("id")
            if name == "Rectangle 180": set_shape_text(sp, TITLE, 2600, True)
            elif sid == "8": set_shape_text(sp, AUTHORS + "\n" + COURSE, 1700)
            elif sid == "14": set_shape_text(sp, INTRO, 1100)
            elif sid == "6": set_shape_text(sp, RESULTS, 1050)
            elif sid == "5": set_shape_text(sp, METHODS, 1000)
            elif sid == "7": set_shape_text(sp, CONCLUSIONS, 1000)
            elif sid == "13": set_shape_text(sp, REFS, 850)
            elif sid == "12": set_shape_text(sp, ACK, 900)
        # Correct section labels in grouped text boxes.
        labels = {"31":"INTRODUCCION", "35":"RESULTADOS", "39":"MATERIALES Y METODOS", "50":"CONCLUSIONES", "63":"REFERENCIAS", "68":"AGRADECIMIENTOS"}
        for sp in root.findall(".//p:sp", ns):
            cnv = sp.find("p:nvSpPr/p:cNvPr", ns)
            if cnv is not None and cnv.attrib.get("id") in labels:
                set_shape_text(sp, labels[cnv.attrib["id"]], 2400 if cnv.attrib["id"] in {"31","35","39","50"} else 1000, True)
        # Reposition useful pictures and move clutter off-canvas.
        pos = {
            "69": (3.0, 32.35, 8.9, 3.43),     # architecture
            "70": (14.45, 20.55, 8.8, 3.84),    # leaderboard
            "71": (23.65, 20.40, 8.2, 6.16),    # PR curves
            "72": (60, 60, 1, 1),               # hide MLflow screenshot
            "73": (62, 62, 1, 1),               # hide R2 pie
        }
        for pic in root.findall(".//p:pic", ns):
            cnv = pic.find("p:nvPicPr/p:cNvPr", ns)
            if cnv is None or cnv.attrib.get("id") not in pos: continue
            x,y,w,h = pos[cnv.attrib["id"]]
            off = pic.find(".//a:xfrm/a:off", ns); ext = pic.find(".//a:xfrm/a:ext", ns)
            off.set("x", str(int(x*914400))); off.set("y", str(int(y*914400)))
            ext.set("cx", str(int(w*914400))); ext.set("cy", str(int(h*914400)))
        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        new_xml = new_xml.replace("Fernando Ramos Ríos".encode("utf-8"), "Fernando Ramos".encode("utf-8"))
        new_xml = new_xml.replace(b"Fernando Ramos Rios", b"Fernando Ramos")
        with zipfile.ZipFile(PPTX_OUT, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "ppt/slides/slide1.xml":
                    data = new_xml
                zout.writestr(item, data)
    print(f"Wrote {PPTX_OUT} ({PPTX_OUT.stat().st_size/1024:.0f} KB)")


def main() -> None:
    make_pdf()
    edit_pptx()


if __name__ == "__main__":
    main()
