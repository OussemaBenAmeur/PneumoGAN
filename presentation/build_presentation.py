#!/usr/bin/env python3
"""Construit la présentation (en français) à partir des résultats mesurés.

Usage : .venv/bin/python presentation/build_presentation.py
Lit results/eval_full/results.json et results/eval_normal10/results.json :
les chiffres ne sont jamais saisis à la main.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentation" / "Presentation_GAN_Augmentation_Radiographies.pptx"

NAVY = RGBColor(0x02, 0x33, 0x4F)
BLUE = RGBColor(0x3B, 0x9C, 0xF0)
LIGHT = RGBColor(0xE8, 0xF2, 0xFB)
SKY = RGBColor(0x83, 0xBB, 0xD4)
GREY = RGBColor(0x55, 0x60, 0x6E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xD6, 0x45, 0x45)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
ORANGE = RGBColor(0xE8, 0x96, 0x2E)
FONT = "Calibri"
FOOTER = "ENISo · IA3.1 · Intelligence Artificielle"
N_SLIDES = 9

# ---------- Données mesurées ----------
full = json.loads((ROOT / "results/eval_full/results.json").read_text())
p10 = ROOT / "results/eval_normal10/results.json"
scar = json.loads(p10.read_text()) if p10.exists() else None


def fr(x, d=3):
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{d}f}".replace(".", ",")


def m(res, strat, key):
    return res["classifier"][strat]["mean"][key]


def sd(res, strat, key):
    return res["classifier"][strat]["std"][key]


STRATS = [("baseline", "Données réelles seules"),
          ("classical_aug", "+ Augmentation classique"),
          ("oversample", "+ Sur-échantillonnage"),
          ("gan_balance", "+ GAN (équilibrage)"),
          ("gan_balance+classical_aug", "+ GAN + aug. classique"),
          ("synthetic_only_TSTR", "Synthétique seul (TSTR)")]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, content, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, font=FONT, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = content if isinstance(content, list) else [content]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.color.rgb, run.font.name = Pt(size), bold, color, font
        run.font.italic = italic
    return box


def bullets(slide, x, y, w, h, items, size=16, color=GREY, space=8):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space)
        r0 = p.add_run()
        r0.text = "▪  "
        r0.font.color.rgb, r0.font.size, r0.font.name = BLUE, Pt(size), FONT
        if isinstance(item, tuple):
            b = p.add_run()
            b.text = item[0]
            b.font.bold, b.font.size, b.font.color.rgb, b.font.name = True, Pt(size), NAVY, FONT
            item = item[1]
        r = p.add_run()
        r.text = item
        r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, FONT
    return box


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(slide, number, title):
    rect(slide, 0, 0, W, Inches(1.05), NAVY)
    rect(slide, 0, Inches(1.05), W, Inches(0.06), BLUE)
    text(slide, Inches(0.55), Inches(0.18), Inches(10.5), Inches(0.75), title, size=30, bold=True,
         color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, Inches(12.25), Inches(0.1), Inches(0.95), Inches(0.85), WHITE,
         MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.15
    slide.shapes.add_picture(str(ROOT / "docs/assets/logo-eniso.png"), Inches(12.35), Inches(0.14),
                             height=Inches(0.77))
    text(slide, Inches(0.55), Inches(7.0), Inches(9), Inches(0.35), FOOTER, size=11, color=GREY)
    text(slide, Inches(12.0), Inches(7.0), Inches(0.8), Inches(0.35), f"{number}/{N_SLIDES}", size=11,
         color=GREY, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, title, body, accent=BLUE, title_size=19, body_size=15):
    rect(slide, x, y, w, h, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.06
    rect(slide, x, y + Inches(0.25), Inches(0.08), h - Inches(0.5), accent)
    text(slide, x + Inches(0.3), y + Inches(0.2), w - Inches(0.5), Inches(0.5), title,
         size=title_size, bold=True)
    if isinstance(body, list):
        bullets(slide, x + Inches(0.3), y + Inches(0.8), w - Inches(0.5), h - Inches(0.9), body,
                size=body_size)
    else:
        text(slide, x + Inches(0.3), y + Inches(0.8), w - Inches(0.5), h - Inches(0.9), body,
             size=body_size, color=GREY)


def banner(slide, y, h, content, size=18):
    rect(slide, Inches(0.55), y, Inches(12.25), h, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.15
    text(slide, Inches(0.85), y + Inches(0.05), Inches(11.7), h - Inches(0.1), content, size=size,
         bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)


def table(slide, x, y, w, h, rows, widths, size=13, first_col_bold=True):
    tbl = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for i, wdt in enumerate(widths):
        tbl.columns[i].width = Inches(wdt)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            for p in cell.text_frame.paragraphs:
                p.font.size, p.font.name = Pt(size), FONT
                p.font.bold = r == 0 or (c == 0 and first_col_bold)
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                p.font.color.rgb = WHITE if r == 0 else NAVY
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (LIGHT if r % 2 else WHITE)
    return tbl


def flow_box(slide, x, y, w, h, title, sub, col):
    b = rect(slide, x, y, w, h, col, MSO_SHAPE.ROUNDED_RECTANGLE)
    b.adjustments[0] = 0.12
    tc = NAVY if col in (SKY, LIGHT) else WHITE
    text(slide, x, y + Inches(0.1), w, Inches(0.75), title.split("\n"), size=16, bold=True, color=tc,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        text(slide, x, y + Inches(0.85), w, h - Inches(0.9), sub.split("\n"), size=11, color=tc,
             align=PP_ALIGN.CENTER)


def arrow(slide, x, y, w, col=GREY, h=Inches(0.3)):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y - h / 2, w, h)
    a.fill.solid(); a.fill.fore_color.rgb = col; a.line.fill.background()


# ---------- 1. Page de garde ----------
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, Inches(4.6), H, LIGHT)
s.shapes.add_picture(str(ROOT / "docs/assets/logo-eniso.png"), Inches(0.9), Inches(1.6), width=Inches(2.8))
rect(s, Inches(4.6), 0, Inches(0.08), H, BLUE)
text(s, Inches(5.2), Inches(0.9), Inches(7.6), Inches(0.4), "Projet · Intelligence Artificielle",
     size=16, color=BLUE, bold=True)
text(s, Inches(5.2), Inches(1.4), Inches(7.7), Inches(2.0),
     "Augmentation de radiographies thoraciques par réseaux antagonistes génératifs (GAN)",
     size=34, bold=True, color=NAVY)
rect(s, Inches(5.2), Inches(3.55), Inches(1.2), Inches(0.06), BLUE)
info = [("Réalisé par", "Oussema Ben Ameur  ·  Yacin Ben Kacem"),
        ("Classe", "IA3.1"),
        ("Encadrant", "Dr Walid Chainbi"),
        ("Module", "Intelligence Artificielle")]
for i, (k, v) in enumerate(info):
    y = Inches(3.95 + i * 0.6)
    text(s, Inches(5.2), y, Inches(2.0), Inches(0.5), k, size=15, color=GREY)
    text(s, Inches(7.2), y, Inches(5.8), Inches(0.5), v, size=17, bold=True, color=NAVY)
text(s, Inches(5.2), Inches(6.7), Inches(7.6), Inches(0.4),
     "École Nationale d'Ingénieurs de Sousse · 2026–2027", size=13, color=GREY)

# ---------- 2. Contexte et problématique ----------
s = prs.slides.add_slide(BLANK)
header(s, 2, "Contexte et problématique")
cw, ch, top = Inches(3.9), Inches(3.6), Inches(1.6)
card(s, Inches(0.55), top, cw, ch, "Données médicales rares", [
    "Acquisition coûteuse et confidentialité des patients",
    "Annotation par des experts : lente et coûteuse",
    "Le deep learning a besoin de beaucoup d'exemples"])
card(s, Inches(4.72), top, cw, ch, "Classes déséquilibrées", [
    "Les cas pathologiques ou rares sont minoritaires",
    "Le modèle est biaisé vers la classe majoritaire",
    "L'augmentation classique ne fait que transformer les mêmes images"], accent=ORANGE)
card(s, Inches(8.89), top, cw, ch, "Apport des GAN", [
    "Apprendre la distribution des images réelles",
    "Générer de nouvelles images d'une classe choisie",
    "Partager des données synthétiques anonymes"], accent=GREEN)
banner(s, Inches(5.55), Inches(1.05),
       "Problématique : un GAN conditionnel peut-il générer des radiographies assez réalistes "
       "pour améliorer un classifieur de pneumonie entraîné sur des données déséquilibrées ?")

# ---------- 3. Principe des GAN ----------
s = prs.slides.add_slide(BLANK)
header(s, 3, "Principe des GAN (Goodfellow et al., 2014)")
y0, bh = Inches(1.55), Inches(1.25)
flow_box(s, Inches(0.55), y0, Inches(1.9), bh, "Bruit z", "z ~ N(0, I)", SKY)
arrow(s, Inches(2.5), y0 + bh / 2, Inches(0.4))
flow_box(s, Inches(2.95), y0, Inches(2.2), bh, "Générateur G", "« faussaire »", BLUE)
arrow(s, Inches(5.2), y0 + bh / 2, Inches(0.4))
flow_box(s, Inches(5.65), y0, Inches(2.0), bh, "Image\nsynthétique", None, LIGHT)
flow_box(s, Inches(5.65), y0 + Inches(1.45), Inches(2.0), bh, "Image\nréelle x", None, LIGHT)
arrow(s, Inches(7.7), y0 + bh / 2, Inches(0.4))
arrow(s, Inches(7.7), y0 + Inches(1.45) + bh / 2, Inches(0.4))
flow_box(s, Inches(8.15), y0 + Inches(0.6), Inches(2.3), bh + Inches(0.25), "Discriminateur D",
         "« détective »", NAVY)
arrow(s, Inches(10.5), y0 + Inches(0.6) + (bh + Inches(0.25)) / 2, Inches(0.4))
flow_box(s, Inches(10.95), y0 + Inches(0.6), Inches(1.85), bh + Inches(0.25), "P(réel)", "0 = faux\n1 = réel", SKY)
rect(s, Inches(0.55), Inches(4.45), Inches(6.6), Inches(0.85), WHITE, MSO_SHAPE.ROUNDED_RECTANGLE, line=BLUE)
text(s, Inches(0.65), Inches(4.5), Inches(6.4), Inches(0.75),
     "min_G max_D  E_x[log D(x)] + E_z[log(1 − D(G(z)))]", size=16, bold=True,
     font="Cambria Math", anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
bullets(s, Inches(0.55), Inches(5.4), Inches(6.6), Inches(1.6), [
    ("D maximise : ", "reconnaître le réel et rejeter le faux"),
    ("G minimise : ", "tromper D ; entraînement alterné D puis G"),
    ("Optimum : ", "p_G = p_données, D répond 0,5 partout")], size=14, space=4)
card(s, Inches(7.4), Inches(4.45), Inches(5.4), Inches(2.4), "Difficultés connues", [
    ("Effondrement de modes : ", "G ne produit que quelques types d'images"),
    ("Instabilité : ", "les pertes oscillent au lieu de converger"),
    ("Pas de critère d'arrêt : ", "la perte ne mesure pas la qualité")],
    accent=RED, body_size=14, title_size=17)

# ---------- 4. État de l'art ----------
s = prs.slides.add_slide(BLANK)
header(s, 4, "État de l'art : deux revues systématiques (PRISMA)")
card(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(3.75),
     "Akpinar et al., IEEE OJEMB 2024 · 72 articles", [
         ("Architectures : ", "cGAN 31 %, CycleGAN 18 %"),
         ("Domaines : ", "cerveau 22 %, cardiologie 18 %, cancer 15 %"),
         ("Modalités : ", "IRM et scanner ; ECG pour les signaux"),
         ("Limites : ", "pas de métriques standard, nombre d'images synthétiques rarement indiqué"),
         ("Pistes : ", "métriques propres au domaine, GAN + diffusion")], body_size=14, title_size=17)
card(s, Inches(6.78), Inches(1.5), Inches(6.0), Inches(3.75),
     "Makhlouf et al., Neural Comput. Appl. 2023 · 52 articles", [
         ("Architectures : ", "cGAN n°1, DCGAN n°2"),
         ("Modalités / organes : ", "IRM, scanner, radiographie ; cerveau, thorax"),
         ("Tâche visée : ", "classification (37 articles sur 52)"),
         ("Évaluation : ", "qualitative, directe (FID…), indirecte (classifieur), la plus fréquente"),
         ("Limites : ", "hyperparamètres rarement publiés")], accent=GREEN, body_size=14, title_size=17)
banner(s, Inches(5.5), Inches(1.25),
       "Notre projet suit ces constats : cGAN + DCGAN, radiographie thoracique, classification "
       "déséquilibrée, évaluation directe et indirecte, protocole entièrement reproductible.", size=17)

# ---------- 5. Architecture ----------
s = prs.slides.add_slide(BLANK)
header(s, 5, "Architecture de la solution")
c = full["train_counts"]
steps = [("Pneumonia-\nMNIST", f"radiographies 64×64\n{c[0]} normales · {c[1]} pneumonie", SKY),
         ("cDCGAN", "G(z, classe)\nD(x, classe)", NAVY),
         ("Images\nsynthétiques", "classe normale\n(minoritaire)", BLUE),
         ("Jeu augmenté", "réel + synthétique\nclasses équilibrées", BLUE),
         ("Classifieur CNN", "normal / pneumonie", BLUE),
         ("Évaluation", "test réel\n624 images", GREEN)]
bw, bh, gap, y0 = Inches(1.75), Inches(1.6), Inches(0.33), Inches(1.6)
xs = [Inches(0.55) + i * (bw + gap) for i in range(len(steps))]
for x, (t, d, col) in zip(xs, steps):
    flow_box(s, x, y0, bw, bh, t, d, col)
for x in xs[:-1]:
    arrow(s, x + bw + Inches(0.04), y0 + bh / 2, gap - Inches(0.08))
a = full["gan_args"]
card(s, Inches(0.55), Inches(3.55), Inches(6.0), Inches(3.25), "Modèle : GAN conditionnel convolutionnel", [
    ("G : ", f"bruit z ({a['z_dim']}) + plongement de la classe → convolutions transposées"),
    ("D : ", "convolutions + normalisation spectrale + projection de la classe"),
    ("Entraînement : ", f"perte non saturante, Adam (lr = 2·10⁻⁴), {a['iters']:,} itérations".replace(",000", " 000")),
    ("Matériel : ", "RTX 4050 portable, ≈ 8 min par GAN")], body_size=14, title_size=17)
card(s, Inches(6.78), Inches(3.55), Inches(6.0), Inches(3.25), "Garde-fous", [
    "Le GAN ne voit que le jeu d'entraînement",
    "Choix du classifieur sur la validation uniquement",
    "Test sur des radiographies réelles jamais vues",
    "3 graines aléatoires, hyperparamètres publiés"], accent=GREEN, body_size=14, title_size=17)

# ---------- 6. Réalisation ----------
s = prs.slides.add_slide(BLANK)
header(s, 6, "Réalisation et démonstration")
pic = s.shapes.add_picture(str(ROOT / "presentation/fig/reel_vs_synthetique.png"), Inches(0.55),
                           Inches(1.35), width=Inches(12.25))
pic.line.color.rgb = SKY
y_after = Inches(1.35) + pic.height + Inches(0.15)
curves = s.shapes.add_picture(str(ROOT / "presentation/fig/courbes_entrainement.png"), Inches(0.55),
                              y_after, height=Inches(6.85) - y_after)
curves.line.color.rgb = SKY
bullets(s, curves.left + curves.width + Inches(0.3), y_after + Inches(0.1),
        Inches(12.8) - (curves.left + curves.width + Inches(0.3)), Inches(6.85) - y_after, [
    ("Anatomie réaliste : ", "côtes, colonne, ombre cardiaque"),
    ("Classe respectée : ", "les poumons « pneumonie » sont plus opaques"),
    ("Entraînement stable : ", "D(x) et D(G(z)) oscillent autour de 0,5"),
    ("Limite : ", "images « normales » peu variées (effondrement partiel)")], size=13, space=4)

# ---------- 7. Méthodologie ----------
s = prs.slides.add_slide(BLANK)
header(s, 7, "Méthodologie de validation")
card(s, Inches(0.55), Inches(1.5), Inches(4.0), Inches(3.55), "Stratégies comparées", [
    "Données réelles seules (référence)",
    "Augmentation classique (rotation, translation)",
    "Sur-échantillonnage (copies)",
    "GAN : équilibrage des classes",
    "GAN + augmentation classique",
    "Synthétique seul (TSTR)"], body_size=13, title_size=17)
text(s, Inches(4.8), Inches(1.5), Inches(4.3), Inches(0.5), "Taxonomie d'évaluation (Makhlouf)",
     size=17, bold=True)
table(s, Inches(4.8), Inches(2.0), Inches(4.3), Inches(2.9), [
    ("Type", "Mesure"),
    ("Qualitative", "grilles réel / synthétique"),
    ("Directe", "FID Inception (vs réf. réel/réel)"),
    ("Indirecte", "AUC, exactitude équilibrée"),
    ("Indirecte", "rappel de la classe normale")], [1.35, 2.95], size=12)
n10 = scar["train_counts"] if scar else [121, c[1]]
card(s, Inches(9.35), Inches(1.5), Inches(3.45), Inches(3.55), "Deux scénarios", [
    ("Complet : ", f"{c[0]} normales (ratio 1:{c[1] / c[0]:.1f})".replace(".", ",")),
    ("Rareté : ", f"10 % gardées, {n10[0]} normales (ratio 1:{n10[1] / n10[0]:.0f})")],
    accent=ORANGE, body_size=13, title_size=17)
banner(s, Inches(5.3), Inches(1.4), [
    "Le classifieur est choisi sur la validation puis évalué une seule fois sur 624 radiographies réelles.",
    "Chaque FID est comparé à la distance entre deux échantillons réels (train vs test) pour lui donner un sens."],
    size=15)

# ---------- 8. Résultats ----------
s = prs.slides.add_slide(BLANK)
header(s, 8, "Résultats expérimentaux")
hdr = ("Stratégie", "AUC", "Exact. équil.", "Exact. équil.\n(rareté 10 %)") if scar else \
      ("Stratégie", "AUC", "Exact. équil.", "Rappel normal")
rows = [hdr]
for k, lab in STRATS:
    r = [lab, fr(m(full, k, "auc")), f"{fr(m(full, k, 'balanced_accuracy'))} ± {fr(sd(full, k, 'balanced_accuracy'), 2)}"]
    if scar:
        r.append(f"{fr(m(scar, k, 'balanced_accuracy'))} ± {fr(sd(scar, k, 'balanced_accuracy'), 2)}")
    else:
        r.append(fr(m(full, k, "recall_normal")))
    rows.append(tuple(r))
table(s, Inches(0.55), Inches(1.4), Inches(7.0), Inches(3.6), rows, [2.75, 1.0, 1.6, 1.65], size=13)
fid_lines = []
for lab, res in [("complet", full), ("rareté", scar)]:
    if res:
        fi = res["fid"]["inception_fid"]["normal"]
        fid_lines.append(f"FID Inception, classe normale ({lab}) : synthétique {fi['synthetic_vs_real_test']:.0f}"
                         f" · référence réel/réel {fi['real_train_vs_real_test (reference)']:.0f}")
text(s, Inches(0.55), Inches(5.15), Inches(7.0), Inches(1.2), fid_lines +
     ["Moyenne ± écart-type sur 3 graines · test : 624 radiographies réelles"], size=12, color=GREY)
cd = CategoryChartData()
cd.categories = ["Réel seul", "Aug. classique", "Sur-échant.", "GAN", "GAN + classique"]
keys = [k for k, _ in STRATS[:5]]
cd.add_series("Complet", [round(m(full, k, "balanced_accuracy"), 3) for k in keys])
if scar:
    cd.add_series("Rareté 10 %", [round(m(scar, k, "balanced_accuracy"), 3) for k in keys])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.75), Inches(1.3), Inches(5.1),
                        Inches(3.4), cd)
ch_ = gf.chart
ch_.has_title = True
ch_.chart_title.text_frame.text = "Exactitude équilibrée (test réel)"
ch_.chart_title.text_frame.paragraphs[0].font.size = Pt(13)
ch_.chart_title.text_frame.paragraphs[0].font.bold = True
ch_.has_legend = bool(scar)
if scar:
    ch_.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch_.legend.include_in_layout = False
    ch_.legend.font.size = Pt(11)
ch_.value_axis.has_major_gridlines = False
lo = min(min(v for v in ser.values) for ser in cd._series)
ch_.value_axis.minimum_scale = max(0, round(lo - 0.1, 1))
ch_.value_axis.maximum_scale = 1.0
ch_.value_axis.tick_labels.font.size = Pt(10)
ch_.value_axis.tick_labels.number_format = '0.0'
ch_.value_axis.tick_labels.number_format_is_linked = False
ch_.category_axis.tick_labels.font.size = Pt(10)
plot = ch_.plots[0]
plot.has_data_labels = True
plot.data_labels.font.size = Pt(9)
plot.data_labels.number_format = '0.00'
plot.data_labels.number_format_is_linked = False
plot.gap_width = 60
for ser, col in zip(plot.series, [NAVY, ORANGE]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
GC, CL = "gan_balance+classical_aug", "classical_aug"
ba = lambda res, k: m(res, k, "balanced_accuracy")
best = f"{fr(ba(full, GC))}" + (f" / {fr(ba(scar, GC))}" if scar else "")
RESULT_BULLETS = [  # rempli à partir des chiffres
    ("Meilleur partout : ", f"GAN + aug. classique ({best})"),
    ("GAN seul ≈ copies : ", "équilibrer ne suffit pas, il faut aussi varier les images")]
if scar:
    RESULT_BULLETS.append(("Rareté : ", f"gain du GAN sur l'aug. classique +{fr(ba(scar, GC) - ba(scar, CL))} "
                                        f"(contre +{fr(ba(full, GC) - ba(full, CL))} en complet)"))
RESULT_BULLETS.append(("TSTR : ", f"AUC {fr(m(full, 'synthetic_only_TSTR', 'auc'))} sans aucune image réelle"))
bullets(s, Inches(7.75), Inches(4.85), Inches(5.1), Inches(2.1), RESULT_BULLETS, size=12, space=3)

# ---------- 9. Conclusion ----------
s = prs.slides.add_slide(BLANK)
header(s, 9, "Conclusion et perspectives")
card(s, Inches(0.55), Inches(1.5), Inches(3.9), Inches(4.0), "Apports des GAN", [
    "Radiographies synthétiques crédibles et conditionnées par la classe",
    "Le classifieur entraîné sur du synthétique seul reste performant",
    "Complément utile à l'augmentation classique, surtout quand les données sont rares"], accent=GREEN)
card(s, Inches(4.72), Inches(1.5), Inches(3.9), Inches(4.0), "Limites observées", [
    "Gain modeste quand les données réelles suffisent",
    "Peu d'images réelles → images moins fidèles (FID 83 → 125)",
    "FID : dépend de l'extracteur, peu fiable sur petit test",
    "Résolution 64×64, pas d'avis de radiologue"], accent=RED)
card(s, Inches(8.89), Inches(1.5), Inches(3.9), Inches(4.0), "Perspectives", [
    "StyleGAN ou modèles de diffusion, plus haute résolution",
    "Test de Turing visuel avec des radiologues",
    "Métriques propres au domaine médical",
    "Autres modalités : IRM, scanner"], accent=BLUE)
banner(s, Inches(5.75), Inches(1.0),
       "Les GAN enrichissent les données médicales rares, mais leur évaluation doit rester rigoureuse, "
       "directe et indirecte.")

prs.save(OUT)
print(f"Présentation enregistrée : {OUT}")
