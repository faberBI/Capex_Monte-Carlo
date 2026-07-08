"""
dcf_report.py
=============
Generatore di REPORT WORD (.docx) per il comitato investimenti, a partire dai
risultati del motore dcf_core.

Filosofia: il report e' un deliverable decisionale, non un dump di numeri. Contiene
la sintesi esecutiva, i risultati con la loro barra d'errore (intervalli di confidenza),
i driver del valore, il tornado di sensibilita', il DSCR e una nota metodologica onesta.

Dipendenze: python-docx, matplotlib, numpy, pandas (+ dcf_core).
Uso tipico (in app):
    buf = BytesIO()
    dcf_report.build_report(df, cfg, res, buf, project_name="...", reliability=ci)
    st.download_button(..., data=buf.getvalue(), file_name="report.docx", mime=...)
"""

from io import BytesIO
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import dcf_core as core

# --- palette (facilmente ri-brandizzabile: basta cambiare questi due colori) ---
ACCENT = "1F3864"     # blu scuro per intestazioni tabella / titoli
ACCENT_RGB = RGBColor(0x1F, 0x38, 0x64)
GREEN = "#2E7D32"
RED = "#C62828"
BAR = "#4C78A8"
GRID = "#D9D9D9"


# ==========================================================================
# helper docx
# ==========================================================================
def _shade(cell, hex_fill):
    """Colore di sfondo di una cella (ShadingType CLEAR)."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _set_cell_text(cell, text, bold=False, color=None, size=10, align="left"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color


def _table(doc, headers, rows, col_widths=None, first_col_left=True):
    """Tabella con intestazione colorata e righe alternate."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for j, h in enumerate(headers):
        _shade(hdr[j], ACCENT)
        _set_cell_text(hdr[j], h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                       size=10, align="left" if (j == 0 and first_col_left) else "center")
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            _set_cell_text(cells[j], val, bold=(j == 0),
                           align="left" if (j == 0 and first_col_left) else "center")
            if i % 2 == 1:
                _shade(cells[j], "F2F4F8")
    if col_widths:
        for j, w in enumerate(col_widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    return t


def _heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = ACCENT_RGB
    return h


def _add_fig(doc, fig, width=6.4):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    doc.add_picture(buf, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _fmt(x, currency="€", dec=0):
    return f"{currency}{x:,.{dec}f}"


# ==========================================================================
# grafici (self-contained, non dipendono da capex.visuals)
# ==========================================================================
def _chart_npv(npv, currency):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(npv, bins=60, color=BAR, alpha=0.85, edgecolor="white", linewidth=0.3)
    mean, p5, p1 = npv.mean(), np.percentile(npv, 5), np.percentile(npv, 1)
    ax.axvline(mean, color="black", lw=1.4, label=f"Media {_fmt(mean, currency)}")
    ax.axvline(p5, color=RED, lw=1.2, ls="--", label=f"VaR 95% {_fmt(p5, currency)}")
    ax.axvline(p1, color=RED, lw=1.0, ls=":", label=f"VaR 99% {_fmt(p1, currency)}")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_xlabel("NPV"); ax.set_ylabel("Frequenza")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    return fig


def _chart_drivers(importance):
    labels = [d["driver"] for d in importance][::-1]
    rhos = [d["spearman"] for d in importance][::-1]
    contribs = [d["contribution"] for d in importance][::-1]
    colors = [GREEN if r >= 0 else RED for r in rhos]
    fig, ax = plt.subplots(figsize=(6.4, max(2.2, 0.5 * len(labels) + 1)))
    ax.barh(labels, rhos, color=colors)
    ax.axvline(0, color="black", lw=0.8); ax.set_xlim(-1, 1)
    for i, (r, c) in enumerate(zip(rhos, contribs)):
        ax.text(r + (0.02 if r >= 0 else -0.02), i, f"{c:.0%}", va="center",
                ha="left" if r >= 0 else "right", fontsize=8)
    ax.set_xlabel("Correlazione di rango con l'NPV (segno = direzione)")
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def _chart_tornado(tor, currency):
    bars = tor["bars"][::-1]
    labels = [b["label"] for b in bars]
    fig, ax = plt.subplots(figsize=(6.4, max(2.2, 0.5 * len(labels) + 1)))
    for i, b in enumerate(bars):
        left, right = min(b["low"], b["high"]), max(b["low"], b["high"])
        ax.barh(i, right - left, left=left, color=BAR, alpha=0.85)
    ax.axvline(tor["base"], color=RED, ls="--", lw=1,
               label=f"NPV base {_fmt(tor['base'], currency)}")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("NPV"); ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.25)
    return fig


# ==========================================================================
# costruzione report
# ==========================================================================
def build_report(df, cfg, res, out, project_name="Progetto", currency="€",
                 reliability=None, author="Conflux", subtitle=None):
    """Costruisce il report Word e lo salva su `out` (path o file-like).
    reliability: output di core.estimate_with_ci (opzionale). Se presente, i risultati
    principali mostrano gli intervalli di confidenza."""
    npv = np.asarray(res["npv"], float)
    doc = Document()

    # --- stile di base ---
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    # --- titolo ---
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = title.add_run("Report di valutazione — Analisi DCF Monte Carlo")
    r.bold = True; r.font.size = Pt(20); r.font.color.rgb = ACCENT_RGB
    sub = doc.add_paragraph()
    rs = sub.add_run(subtitle or f"{project_name}")
    rs.font.size = Pt(13); rs.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    meta = doc.add_paragraph()
    rm = meta.add_run(f"{author}  ·  {date.today().strftime('%d/%m/%Y')}  ·  "
                      f"framework {res['framework']}  ·  {cfg.n_sim:,} simulazioni/batch")
    rm.font.size = Pt(9); rm.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    # riga separatrice
    pborder = doc.add_paragraph()
    pPr = pborder._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr"); bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1"); bottom.set(qn("w:color"), ACCENT)
    pbdr.append(bottom); pPr.append(pbdr)

    # --- metriche di base ---
    mean = float(npv.mean()); p5 = float(np.percentile(npv, 5))
    p1 = float(np.percentile(npv, 1)); p_loss = float(np.mean(npv < 0))
    dscr = res.get("dscr"); has_debt = res.get("has_debt", False)
    dscr_min = float(np.nanmin(dscr)) if has_debt else None
    dscr_mean = float(np.nanmean(dscr)) if has_debt else None

    # --- sintesi esecutiva ---
    _heading(doc, "Sintesi esecutiva", 1)
    verdict = ("valore atteso positivo" if mean > 0 else "valore atteso negativo")
    risk_txt = (f"con una probabilita' di NPV negativo del {p_loss:.0%}")
    p = doc.add_paragraph()
    p.add_run(
        f"Su {int(reliability['total_n']) if reliability else cfg.n_sim:,} simulazioni, "
        f"il progetto «{project_name}» mostra un {verdict} pari a {_fmt(mean, currency)} "
        f"({risk_txt}). La perdita nello scenario sfavorevole (VaR 95%) e' "
        f"{_fmt(p5, currency)}; nello scenario estremo (VaR 99%) e' {_fmt(p1, currency)}."
    )
    if has_debt:
        p.add_run(f" La copertura del servizio del debito (DSCR) minima e' {dscr_min:.2f} "
                  f"(media {dscr_mean:.2f}, giudizio {core.dscr_rating(dscr_min).split(' ')[-1]}).")
    doc.add_paragraph(
        "I risultati incorporano correlazione tra i fattori, dipendenza di coda (copula t "
        "quando selezionata), distribuzioni per fattore e attualizzazione coerente col "
        "framework. Le stime sono accompagnate dal relativo intervallo di confidenza.",
    ).runs[0].font.size = Pt(9.5)

    # --- risultati principali ---
    _heading(doc, "Risultati principali", 1)
    if reliability:
        rows = []
        for k, m in reliability["metrics"].items():
            if k == "P(NPV<0)":
                val = f"{m['value']:.1%}"
                ci = f"{m['ci_low']:.1%} … {m['ci_high']:.1%}"
                hw = f"±{m['half_width']*100:.1f} pp"
            elif k == "Dev. standard":
                val = _fmt(m["value"], currency); ci = f"{_fmt(m['ci_low'], currency)} … {_fmt(m['ci_high'], currency)}"
                hw = f"±{_fmt(m['half_width'], currency)}"
            else:
                val = _fmt(m["value"], currency); ci = f"{_fmt(m['ci_low'], currency)} … {_fmt(m['ci_high'], currency)}"
                hw = f"±{_fmt(m['half_width'], currency)}"
            rows.append([k, val, hw, ci])
        _table(doc, ["Metrica", "Stima", "Errore (IC 95%)", "Intervallo di confidenza 95%"],
               rows, col_widths=[2.0, 1.5, 1.4, 1.9])
        doc.add_paragraph(
            f"Intervalli calcolati su {reliability['n_batches']} batch indipendenti "
            f"({reliability['total_n']:,} simulazioni totali), campionamento "
            f"{'Latin Hypercube' if reliability['sampling']=='lhs' else 'Monte Carlo casuale'}. "
            "L'errore standard e' la deviazione tra i batch: valido anche con LHS."
        ).runs[0].font.size = Pt(8.5)
    else:
        rows = [
            ["E[NPV] (media)", _fmt(mean, currency)],
            ["Mediana", _fmt(float(np.median(npv)), currency)],
            ["VaR 95% (P5)", _fmt(p5, currency)],
            ["VaR 99% (P1)", _fmt(p1, currency)],
            ["P(NPV<0)", f"{p_loss:.1%}"],
            ["Dev. standard", _fmt(float(npv.std(ddof=1)), currency)],
        ]
        _table(doc, ["Metrica", "Stima"], rows, col_widths=[3.0, 2.0])

    _add_fig(doc, _chart_npv(npv, currency))
    doc.add_paragraph("Distribuzione dell'NPV simulato con media e soglie di rischio (VaR).").runs[0].font.size = Pt(8.5)

    # --- driver ---
    _heading(doc, "Driver del valore", 1)
    imp = core.driver_importance(res)
    if imp:
        rows = [[d["driver"], f"{d['spearman']:+.2f}", f"{d['contribution']:.0%}",
                 "spinge in alto" if d["spearman"] >= 0 else "spinge in basso"] for d in imp]
        _table(doc, ["Driver", "Corr. di rango", "Contributo", "Direzione"], rows,
               col_widths=[2.2, 1.6, 1.4, 1.6])
        _add_fig(doc, _chart_drivers(imp))
        doc.add_paragraph(
            "Correlazione di rango (Spearman) tra ciascun fattore e l'NPV sulle simulazioni. "
            "Il contributo (rho² normalizzato) e' robusto anche con fattori correlati."
        ).runs[0].font.size = Pt(8.5)

    # --- tornado ---
    _heading(doc, "Sensibilita' (tornado)", 1)
    tor = core.tornado_oneway(df, cfg, p_low=0.10, p_high=0.90, include_assumptions=True)
    rows = [[b["label"], _fmt(b["low"], currency), _fmt(b["high"], currency),
             _fmt(b["swing"], currency)] for b in tor["bars"]]
    _table(doc, ["Fattore / ipotesi", "NPV (basso)", "NPV (alto)", "Ampiezza"], rows,
           col_widths=[2.4, 1.5, 1.5, 1.4])
    _add_fig(doc, _chart_tornado(tor, currency))
    doc.add_paragraph(
        f"NPV al variare di un elemento alla volta tra P10 e P90 (base = {_fmt(tor['base'], currency)}), "
        "tenendo gli altri al valore 'piano'. Include le ipotesi scalari."
    ).runs[0].font.size = Pt(8.5)

    # --- DSCR ---
    if has_debt:
        _heading(doc, "Analisi del credito (DSCR)", 1)
        rows = [
            ["DSCR minimo (peggior anno)", f"{dscr_min:.2f}"],
            ["DSCR medio", f"{dscr_mean:.2f}"],
            ["Giudizio (sul minimo)", core.dscr_rating(dscr_min)],
        ]
        _table(doc, ["Indicatore", "Valore"], rows, col_widths=[3.5, 1.8])
        doc.add_paragraph(
            "DSCR = CFADS / servizio del debito, cioe' cassa disponibile PRIMA del servizio "
            "del debito sul totale di interessi e quota capitale (definizione bancaria)."
        ).runs[0].font.size = Pt(8.5)

    # --- piano di finanziamento ---
    fund = res.get("funding")
    if fund is not None:
        _heading(doc, "Piano di finanziamento (equity / senior debt)", 1)
        rows = [
            ["Fabbisogno di costruzione", _fmt(fund["total_need"], currency)],
            ["Senior debt (su capex)", f"{_fmt(fund['debt_capex'], currency)}  (gearing {fund['gearing_capex']:.0%})"],
            ["Interessi in costruzione (IDC)", _fmt(fund["idc_total"], currency)],
            ["Debito a fine costruzione (COD)", _fmt(fund["debt_at_cod"], currency)],
            ["Equity", _fmt(fund["total_equity"], currency)],
        ]
        _table(doc, ["Voce", "Valore"], rows, col_widths=[3.4, 2.4])
        eq = res.get("equity_injection")
        years_lbl = [str(int(y)) for y in res["years_col"]]
        srows = [[years_lbl[i], _fmt(fund["debt_inflow"][i], currency),
                  _fmt(eq[i] if eq is not None else 0.0, currency),
                  _fmt(fund["debt_repayment"][i], currency)] for i in range(len(years_lbl))]
        _table(doc, ["Anno", "Drawdown debito", "Equity", "Rimborso debito"], srows,
               col_widths=[1.2, 1.9, 1.6, 1.7])
        doc.add_paragraph(
            "Tiraggio derivato dal cronoprogramma di capex. Il senior debt e' dimensionato sul "
            "caso base; i sovracosti di capex sono a carico dell'equity."
        ).runs[0].font.size = Pt(8.5)

    # --- ipotesi e parametri ---
    doc.add_page_break()
    _heading(doc, "Ipotesi e parametri", 1)
    tv_label = {"none": "Nessuno", "gordon": f"Gordon (g={cfg.tv_growth:.1%})",
                "multiple": f"Multiplo EBITDA ×{cfg.tv_multiple:g}"}.get(cfg.tv_method.lower(),
                                                                          cfg.tv_method)
    dist_label = {"triangular": "Triangolare", "pert": "PERT", "normal": "Normale",
                  "lognormal": "Lognormale", "uniform": "Uniforme", "empirical": "Empirica"}

    def dl(spec):
        d = (spec or {}).get("dist", "triangular")
        s = dist_label.get(d, d)
        if d == "pert":
            s += f" (λ={spec.get('lam', 4)})"
        return s

    rows = [
        ["Framework", "FCFF (unlevered, WACC)" if res["framework"] == "FCFF" else "FCFE (levered, Ke)"],
        ["Tasso di sconto", f"{cfg.discount_rate:.2%}"],
        ["Aliquota fiscale", f"{cfg.tax_rate:.1%}"],
        ["Valore terminale", tv_label],
        ["Distribuzione ricavi", dl(cfg.dist_revenue)],
        ["Distribuzione costi variabili", dl(cfg.dist_cost)],
        ["Distribuzione disposal", dl(cfg.dist_disposal)],
        ["Copula", "t di Student (df=%.0f)" % cfg.copula_df if cfg.copula == "t" else "Gaussiana"],
        ["Correlazione ricavi↔costi", f"{cfg.corr_rev_cost:+.2f}"],
        ["Persistenza AR(1)", f"{cfg.persistence:.2f}"],
        ["Shift temporale", "attivo" if cfg.enable_shift else "disattivo"],
        ["Campionamento", "Latin Hypercube" if cfg.sampling == "lhs" else "Monte Carlo casuale"],
        ["Simulazioni per batch", f"{cfg.n_sim:,}"],
    ]
    _table(doc, ["Parametro", "Valore"], rows, col_widths=[3.0, 3.0])

    # --- nota metodologica ---
    _heading(doc, "Nota metodologica", 1)
    notes = [
        "Framework: FCFF si attualizza al WACC con imposte sull'EBIT; FCFE si attualizza al "
        "costo dell'equity (Ke) e include il servizio del debito e lo scudo fiscale sugli interessi.",
        "Correlazione e code: i fattori sono legati da una copula (gaussiana o t di Student) con "
        "persistenza temporale AR(1). La copula t introduce dipendenza di coda — scenari in cui "
        "piu' fattori vanno male insieme — che la gaussiana sottostima.",
        "Distribuzioni: le tre stime min/piano/max sono mappate su ciascuna distribuzione con "
        "convenzioni esplicite. La lognormale adatta min e max come 5° e 95° percentile della "
        "magnitudo (segno preservato); la normale centra sul valore 'piano' e ignora l'asimmetria.",
        "Campionamento Latin Hypercube: stratifica le estrazioni riducendo la varianza delle "
        "stime a parita' di simulazioni.",
        "Intervalli di confidenza: calcolati per replicazione (piu' batch indipendenti), metodo "
        "valido anche con LHS dove la formula std/√n non si applica.",
        "Valore terminale: se calcolato con Gordon sull'ultimo anno e con shift attivo, l'ultimo "
        "flusso puo' non rappresentare lo stato stazionario — valutare la normalizzazione.",
    ]
    for n in notes:
        para = doc.add_paragraph(style="List Bullet")
        run = para.add_run(n); run.font.size = Pt(9.5)

    disc = doc.add_paragraph()
    rd = disc.add_run(
        "Documento generato automaticamente a supporto dell'analisi. Non costituisce consulenza "
        "finanziaria o raccomandazione d'investimento; i risultati dipendono dalle ipotesi in input."
    )
    rd.italic = True; rd.font.size = Pt(8.5); rd.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.save(out)
    return out


if __name__ == "__main__":
    # smoke test: genera un report di esempio
    import pandas as pd
    df = pd.DataFrame({
        "Anno": np.arange(2026, 2032),
        "Revenues min": [0, 80, 180, 260, 300, 320], "Revenues piano": [0, 100, 220, 320, 380, 400],
        "Revenues max": [0, 130, 270, 390, 460, 500],
        "Cost var min": [0, -70, -150, -210, -240, -250], "Cost var piano": [0, -55, -120, -170, -200, -210],
        "Cost var max": [0, -40, -90, -130, -150, -160],
        "Costs fixed": [-20, -25, -30, -30, -30, -30], "Amort, & Depreciation": [0, -40, -40, -40, -40, -40],
        "Capex": [-200, -50, 0, 0, 0, 0],
        "Disposal & Capex Saving min": [0, 0, 0, 0, 0, 10], "Disposal & Capex Saving": [0, 0, 0, 0, 0, 20],
        "Disposal & Capex Saving max": [0, 0, 0, 0, 0, 35],
        "Change in working cap,": [0, -10, -15, -8, -5, 5], "Debt inflow": [150, 0, 0, 0, 0, 0],
        "Debt repayment": [0, 30, 30, 30, 30, 30], "Interest rate": [0.05] * 6,
    })
    cfg = core.SimConfig(n_sim=8000, seed=42, framework="FCFE", tv_method="gordon", tv_growth=0.02,
                         discount_rate=0.10, corr_rev_cost=0.6, persistence=0.3,
                         dist_revenue={"dist": "pert"}, dist_cost={"dist": "lognormal"},
                         copula="t", copula_df=5.0, sampling="lhs")
    res = core.run_simulations(df, cfg)
    ci = core.estimate_with_ci(df, cfg, n_batches=15)
    build_report(df, cfg, res, "/mnt/user-data/outputs/report_esempio.docx",
                 project_name="Progetto Fibra — Business Case", reliability=ci)
    print("Report generato: /mnt/user-data/outputs/report_esempio.docx")
