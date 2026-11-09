"""pdf_generator.py
Generador dinámico del informe PDF (tabla de épocas + procedimiento paso a paso).
Recibe un objeto Perceptron ya entrenado; no conoce nada de Streamlit.
"""
import io
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from perceptron import Perceptron

_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


def _register_fonts():
    """Registra una fuente Unicode (para η, Σ, subíndices). Si no hay, usa Helvetica + ASCII."""
    for regular, bold in _FONT_CANDIDATES:
        if os.path.exists(regular) and os.path.exists(bold):
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                          italic="Body", boldItalic="Body-Bold")
            return "Body", "Body-Bold", True
    return "Helvetica", "Helvetica-Bold", False


def _n(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _p(v: float) -> str:
    """Número entre paréntesis si es negativo."""
    return f"({_n(v)})" if v < 0 else _n(v)


def generate_pdf(model: Perceptron) -> bytes:
    font, bold, unicode_ok = _register_fonts()
    eta = "η" if unicode_ok else "eta"
    dot = "·" if unicode_ok else "*"
    minus = "−" if unicode_ok else "-"
    sub = (lambda a, i: f"{a}{'₁₂'[i - 1]}") if unicode_ok else (lambda a, i: f"{a}{i}")

    ss = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=ss["Title"], fontName=bold, fontSize=17)
    h2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=bold, fontSize=13,
                        spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName=bold, fontSize=10.5,
                        spaceBefore=6, spaceAfter=2)
    body = ParagraphStyle("B", parent=ss["BodyText"], fontName=font, fontSize=9.5, leading=13)
    mono = ParagraphStyle("M", parent=body, leftIndent=12, fontSize=9.5, leading=13.5)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Perceptrón monocapa - compuerta AND")
    story = [Paragraph("Perceptrón Monocapa – Compuerta AND", title),
             Paragraph("Informe del procedimiento de entrenamiento", body), Spacer(1, 8)]

    # ---- Hiperparámetros ----
    story.append(Paragraph("1. Valores iniciales e hiperparámetros", h2))
    story.append(Paragraph(
        f"{sub('w', 1)} = {_n(model.w1_init)} &nbsp;&nbsp; {sub('w', 2)} = {_n(model.w2_init)} "
        f"&nbsp;&nbsp; b = {_n(model.b_init)} &nbsp;&nbsp; {eta} = {_n(model.eta)}", body))
    story.append(Paragraph(
        "Función de activación escalón: f(z) = 1 si z ≥ 0, f(z) = 0 si z &lt; 0."
        if unicode_ok else "Funcion escalon: f(z)=1 si z>=0, f(z)=0 si z<0.", body))
    story.append(Paragraph(
        f"Resultado: {'convergió' if model.converged else 'NO convergió'} en "
        f"{len(model.epochs)} época(s). Pesos finales: {sub('w', 1)} = {_n(model.w1)}, "
        f"{sub('w', 2)} = {_n(model.w2)}, b = {_n(model.b)}.", body))

    # ---- Tabla de épocas ----
    story.append(Paragraph("2. Tabla resumen por épocas", h2))
    rows = [["Época", sub("w", 1), sub("w", 2), "b", "Errores"]]
    rows += [[str(e.epoch), _n(e.w1), _n(e.w2), _n(e.b), str(e.errors)] for e in model.epochs]
    t = Table(rows, hAlign="LEFT", colWidths=[2.2 * cm] * 5)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
    ]))
    story.append(t)

    # ---- Procedimiento ----
    story.append(Paragraph("3. Procedimiento paso a paso", h2))
    story.append(Paragraph(
        f"z = ({sub('x', 1)}{dot}{sub('w', 1)}) + ({sub('x', 2)}{dot}{sub('w', 2)}) + b; "
        f"y = f(z); e = t {minus} y; w(nuevo) = w(previo) + {eta}{dot}e{dot}x; "
        f"b(nuevo) = b(previo) + {eta}{dot}e", body))

    for ep in model.epochs:
        block = [Paragraph(f"Época {ep.epoch}", h3)]
        for s in model.steps_of_epoch(ep.epoch):
            lines = [
                f"<b>Muestra {s.sample}:</b> {sub('x', 1)} = {_n(s.x1)}, {sub('x', 2)} = {_n(s.x2)}, "
                f"t = {s.t} &nbsp;|&nbsp; {sub('w', 1)} = {_n(s.w1_prev)}, "
                f"{sub('w', 2)} = {_n(s.w2_prev)}, b = {_n(s.b_prev)}",
                f"z = ({_n(s.x1)}{dot}{_p(s.w1_prev)}) + ({_n(s.x2)}{dot}{_p(s.w2_prev)}) + "
                f"{_p(s.b_prev)} = {_n(s.z)}",
                f"y = f({_n(s.z)}) = {s.y} &nbsp;&nbsp; e = {s.t} {minus} {s.y} = {s.e}",
            ]
            if s.e == 0:
                lines.append("e = 0 → no hay actualización." if unicode_ok else "e = 0: sin actualizacion.")
            else:
                lines += [
                    f"{sub('w', 1)}(n) = {_p(s.w1_prev)} + {_n(model.eta)}{dot}{_p(s.e)}{dot}{_n(s.x1)} = {_n(s.w1_new)}",
                    f"{sub('w', 2)}(n) = {_p(s.w2_prev)} + {_n(model.eta)}{dot}{_p(s.e)}{dot}{_n(s.x2)} = {_n(s.w2_new)}",
                    f"b(n) = {_p(s.b_prev)} + {_n(model.eta)}{dot}{_p(s.e)} = {_n(s.b_new)}",
                ]
            block.append(KeepTogether([Paragraph(lines[0], body)] +
                                      [Paragraph(l, mono) for l in lines[1:]] + [Spacer(1, 4)]))
        block.append(Paragraph(
            f"<b>Fin de época {ep.epoch}:</b> {sub('w', 1)} = {_n(ep.w1)}, {sub('w', 2)} = {_n(ep.w2)}, "
            f"b = {_n(ep.b)}, errores = {ep.errors}", body))
        story.extend(block)

    doc.build(story)
    return buf.getvalue()
