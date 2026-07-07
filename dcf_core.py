"""
dcf_core.py
============
Motore di simulazione Monte Carlo per la valutazione DCF (NPV @Risk).

Modulo PURO: nessuna dipendenza da Streamlit. Serve per:
  - poter testare la correttezza dei numeri con unit test / smoke test;
  - riusare la logica fuori dall'app (batch, notebook, API).

Correzioni rispetto alla versione originale (dcf.py):
  1) Framework FCFF/FCFE coerente e selezionabile (niente ibridi).
  2) DSCR = CFADS / servizio del debito (cassa PRIMA del servizio del debito).
  3) Monte Carlo con correlazione (copula gaussiana) + persistenza AR(1).
  4) Shift temporale a livello di PROGETTO: una sola estrazione di ritardo
     applicata in modo coerente a ricavi, costi e capex.
  5) Valore terminale opzionale (Gordon growth oppure multiplo di uscita).

CONVENZIONI DI SEGNO nel file Excel di input (invariate: i tuoi file continuano a funzionare):
  Ricavi (Revenues ...)             -> POSITIVI
  Costi variabili (Cost var ...)    -> NEGATIVI
  Costi fissi (Costs fixed)         -> NEGATIVI
  Amm.ti (Amort, & Depreciation)    -> NEGATIVI
  Capex                             -> NEGATIVI
  Disposal & Capex Saving           -> POSITIVI
  Change in working cap,            -> segno di cassa (uscita = NEGATIVO)
  Debt inflow / Debt repayment      -> POSITIVI (magnitudo)
  Interest rate                     -> tasso periodale (es. 0.05)
"""

from dataclasses import dataclass, field
import numpy as np


# ==========================================================================
# 1. UTILITY STATISTICHE
# ==========================================================================
def _ndtr(x):
    """CDF normale standard, vettorizzata. Usa scipy se presente, altrimenti erf."""
    try:
        from scipy.special import ndtr
        return ndtr(x)
    except Exception:  # pragma: no cover - fallback senza scipy
        import math
        vec = np.vectorize(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))))
        return vec(x)


def nearest_correlation(R):
    """Proietta R sul cono delle matrici semidefinite positive e la riscala
    a diagonale unitaria. Evita il crash di Cholesky se l'utente inserisce
    correlazioni non coerenti (es. tutte molto alte)."""
    R = np.asarray(R, float)
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-10, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return A


def triangular_ppf(u, lo, mode, hi):
    """Inversa della CDF triangolare (vettorizzata).
    lo/mode/hi sono per-anno e vengono broadcastati su u [n_sim, years].
    Gestisce il caso degenere lo==mode==hi (nessun crash, ritorna il valore)."""
    u = np.asarray(u, float)
    lo = np.broadcast_to(np.asarray(lo, float), u.shape)
    mode = np.broadcast_to(np.asarray(mode, float), u.shape)
    hi = np.broadcast_to(np.asarray(hi, float), u.shape)

    span = hi - lo
    degenerate = span <= 0
    safe_span = np.where(degenerate, 1.0, span)
    Fc = np.where(degenerate, 0.0, (mode - lo) / safe_span)  # CDF alla moda

    left = u < Fc
    left_val = lo + np.sqrt(np.clip(u * span * (mode - lo), 0.0, None))
    right_val = hi - np.sqrt(np.clip((1.0 - u) * span * (hi - mode), 0.0, None))
    out = np.where(left, left_val, right_val)
    out = np.where(degenerate, lo, out)
    return out


# --------------------------------------------------------------------------
# Libreria di distribuzioni marginali (tutte con la stessa interfaccia PPF)
# Le tre stime min/mode/max ("piano") vengono mappate su ciascuna distribuzione
# con convenzioni esplicite e documentate.
# --------------------------------------------------------------------------
_U_EPS = 1e-12
_Z95 = 1.6448536269514722  # quantile normale al 95%


def _clip01(u):
    return np.clip(np.asarray(u, float), _U_EPS, 1.0 - _U_EPS)


def _ndtri(p):
    """Inversa della CDF normale standard (quantile)."""
    try:
        from scipy.special import ndtri
        return ndtri(p)
    except Exception:  # pragma: no cover
        # fallback: bisezione su erf (raro)
        from scipy.optimize import brentq
        import math
        f = np.vectorize(lambda q: brentq(lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))) - q,
                                          -8, 8))
        return f(p)


def _betaincinv(a, b, y):
    from scipy.special import betaincinv
    return betaincinv(a, b, y)


def pert_ppf(u, lo, mode, hi, lam=4.0):
    """PPF della PERT (Beta riparametrizzata sui tre punti).
    lam=4 -> PERT standard; lam piu' alto = piu' peso sulla moda (PERT modificata).
    Rispetto alla triangolare da' meno peso agli estremi: piu' adatta a stime da esperto."""
    u = np.asarray(u, float)
    lo = np.broadcast_to(np.asarray(lo, float), u.shape)
    mode = np.broadcast_to(np.asarray(mode, float), u.shape)
    hi = np.broadcast_to(np.asarray(hi, float), u.shape)
    span = hi - lo
    degenerate = span <= 0
    safe = np.where(degenerate, 1.0, span)
    alpha = 1.0 + lam * (mode - lo) / safe
    beta_ = 1.0 + lam * (hi - mode) / safe
    x = _betaincinv(alpha, beta_, _clip01(u))
    return np.where(degenerate, lo, lo + x * span)


def uniform_ppf(u, lo, hi):
    """PPF uniforme su [lo, hi] (la moda viene ignorata)."""
    u = np.asarray(u, float)
    lo = np.broadcast_to(np.asarray(lo, float), u.shape)
    hi = np.broadcast_to(np.asarray(hi, float), u.shape)
    return lo + u * (hi - lo)


def normal_ppf_range(u, lo, mode, hi, z=_Z95):
    """Normale adattata ai tre punti: media = moda ('piano'),
    dev.st. = (max - min) / (2*z95). L'eventuale asimmetria dei tre punti e' ignorata."""
    u = np.asarray(u, float)
    lo = np.broadcast_to(np.asarray(lo, float), u.shape)
    mode = np.broadcast_to(np.asarray(mode, float), u.shape)
    hi = np.broadcast_to(np.asarray(hi, float), u.shape)
    sd = (hi - lo) / (2.0 * z)
    degenerate = sd <= 0
    return np.where(degenerate, mode, mode + np.where(degenerate, 0.0, sd) * _ndtri(_clip01(u)))


def lognormal_ppf_range(u, lo, mode, hi, z=_Z95):
    """Lognormale adattata cosi' che min e max siano i percentili 5 e 95 della MAGNITUDO.
    Il segno del fattore (ricavi + / costi -) viene preservato. Utile per grandezze
    che non cambiano segno e hanno coda destra (es. costi che possono esplodere)."""
    u = np.asarray(u, float)
    lo = np.broadcast_to(np.asarray(lo, float), u.shape)
    mode = np.broadcast_to(np.asarray(mode, float), u.shape)
    hi = np.broadcast_to(np.asarray(hi, float), u.shape)
    sign = np.sign(lo + mode + hi)                       # segno coerente del fattore
    a = np.minimum(np.abs(lo), np.abs(hi))
    b = np.maximum(np.abs(lo), np.abs(hi))
    degenerate = (sign == 0) | (a <= 0) | (b <= a)
    la = np.log(np.where(a <= 0, 1.0, a))
    lb = np.log(np.where(b <= 0, 1.0, b))
    mu = (la + lb) / 2.0
    sigma = (lb - la) / (2.0 * z)
    mag = np.exp(mu + sigma * _ndtri(_clip01(u)))
    return np.where(degenerate, mode, sign * mag)


def empirical_ppf(u, samples):
    """PPF empirica (per interpolazione dei quantili) da uno storico di osservazioni."""
    s = np.sort(np.asarray(samples, float).ravel())
    if s.size == 0:
        return np.zeros_like(np.asarray(u, float))
    return np.quantile(s, _clip01(u), method="linear")


# distribuzioni disponibili (per popolare i menu della UI)
DISTRIBUTIONS = ("triangular", "pert", "normal", "lognormal", "uniform", "empirical")


def sample_marginal(u2d, spec, lo, mode, hi):
    """Applica la marginale scelta agli uniformi correlati u2d [n_sim, years].
    spec: dict con chiave 'dist' e parametri opzionali ('lam' per PERT, 'samples' per empirica).
    Cosi' la struttura di correlazione (copula) resta invariata mentre la forma
    della distribuzione cambia per ciascun fattore."""
    spec = spec or {}
    dist = spec.get("dist", "triangular")
    if dist == "triangular":
        return triangular_ppf(u2d, lo, mode, hi)
    if dist == "pert":
        return pert_ppf(u2d, lo, mode, hi, spec.get("lam", 4.0))
    if dist == "normal":
        return normal_ppf_range(u2d, lo, mode, hi)
    if dist == "lognormal":
        return lognormal_ppf_range(u2d, lo, mode, hi)
    if dist == "uniform":
        return uniform_ppf(u2d, lo, hi)
    if dist == "empirical":
        samples = spec.get("samples", None)
        if samples is None or len(samples) == 0:
            return triangular_ppf(u2d, lo, mode, hi)   # fallback prudente
        return empirical_ppf(u2d, samples)
    return triangular_ppf(u2d, lo, mode, hi)


def correlated_uniforms(n_sim, years, corr, persistence, rng,
                        copula="gaussian", copula_df=8.0):
    """Copula (gaussiana o t di Student) con persistenza AR(1).

    corr        : matrice 3x3 di correlazione tra {ricavi, costi var, disposal}
    persistence : coefficiente AR(1) sull'orizzonte temporale (0 = anni indipendenti)
    copula      : 'gaussian' | 't'
    copula_df   : gradi di liberta' della t (piu' bassi = code piu' spesse / dipendenza di coda)

    Ritorna un array uniforme [n_sim, years, 3]. Le marginali vengono imposte a valle.

    Perche' la copula t: la gaussiana ha dipendenza di coda nulla, cioe' sottostima la
    probabilita' che piu' fattori vadano male INSIEME in uno scenario estremo (la lezione
    del 2008). La t introduce dipendenza di coda: proprio cio' che serve a un tool di rischio.
    """
    L = np.linalg.cholesky(nearest_correlation(corr))
    phi = float(np.clip(persistence, -0.999, 0.999))
    z = np.empty((n_sim, years, 3))
    prev = None
    for t in range(years):
        shock = rng.standard_normal((n_sim, 3)) @ L.T  # innovazioni correlate, varianza unitaria
        if t == 0:
            z[:, t, :] = shock
        else:
            z[:, t, :] = phi * prev + np.sqrt(1.0 - phi ** 2) * shock
        prev = z[:, t, :]

    if str(copula).lower() == "t":
        # copula t di Student "piena": UN fattore chi-quadro per simulazione, condiviso
        # fra tutte le voci e tutti gli anni -> in uno scenario estremo TUTTO va male
        # insieme (regime di crisi). df alto -> torna alla gaussiana.
        nu = max(float(copula_df), 2.1)
        w = rng.chisquare(nu, size=(n_sim, 1, 1))
        z_t = z / np.sqrt(w / nu)
        from scipy.stats import t as _t
        return _t.cdf(z_t, nu)
    return _ndtr(z)


# ==========================================================================
# 2. LETTURA / VALIDAZIONE INPUT
# ==========================================================================
REQUIRED_TRIPLES = {
    "revenue":  ("Revenues min", "Revenues piano", "Revenues max"),
    "cost_var": ("Cost var min", "Cost var piano", "Cost var max"),
    "disposal": ("Disposal & Capex Saving min", "Disposal & Capex Saving",
                 "Disposal & Capex Saving max"),
}
SINGLE_COLS = {
    "costs_fixed":    "Costs fixed",
    "amort":          "Amort, & Depreciation",
    "capex":          "Capex",
    "change_wc":      "Change in working cap,",
    "debt_inflow":    "Debt inflow",
    "debt_repayment": "Debt repayment",
    "interest_rate":  "Interest rate",
}


def _col(df, name, default, n):
    """Legge una colonna; se manca ritorna un vettore costante = default."""
    if name in df.columns:
        return df[name].fillna(default).to_numpy(dtype=float)
    return np.full(n, default, dtype=float)


def validate_columns(df):
    """Ritorna (ok, missing). `missing` = colonne attese ma assenti che verrebbero
    lette silenziosamente come 0. Serve a bloccare/avvisare invece di produrre
    numeri sbagliati con un'aria convincente."""
    present = set(df.columns)
    missing = []
    for cols in REQUIRED_TRIPLES.values():
        missing += [c for c in cols if c not in present]
    for c in SINGLE_COLS.values():
        if c not in present:
            missing.append(c)
    return (len([c for cols in REQUIRED_TRIPLES.values() for c in cols
                 if c not in present]) == 0, missing)


# ==========================================================================
# 3. CONFIGURAZIONE
# ==========================================================================
@dataclass
class SimConfig:
    # --- framework e attualizzazione ---
    framework: str = "FCFE"          # "FCFF" (unlevered, WACC) | "FCFE" (levered, Ke)
    discount_rate: float = 0.10      # WACC se FCFF, Ke se FCFE
    tax_rate: float = 0.25
    n_sim: int = 2000
    seed: int = 0                    # 0 = random

    # --- correlazione Monte Carlo ---
    corr_rev_cost: float = 0.5       # ricavi <-> costi variabili (di norma > 0)
    corr_rev_disp: float = 0.0
    corr_cost_disp: float = 0.0
    persistence: float = 0.3         # AR(1) anno-su-anno

    # --- shift temporale (a livello progetto) ---
    enable_shift: bool = True
    shift_probs: tuple = (0.3, 0.5, 0.2)   # P(0), P(+1 anno), P(+2 anni)
    shift_rev_pct: float = 100.0
    shift_cs_pct: float = 100.0
    shift_capex_pct: float = 100.0

    # --- valore terminale ---
    tv_method: str = "none"          # "none" | "gordon" | "multiple"
    tv_growth: float = 0.02          # g per Gordon
    tv_multiple: float = 6.0         # multiplo su EBITDA ultimo anno

    # --- distribuzioni marginali per fattore di rischio ---
    # ogni spec e' un dict: {"dist": <tipo>, "lam": <PERT>, "samples": <empirica>}
    dist_revenue: dict = field(default_factory=lambda: {"dist": "triangular"})
    dist_cost: dict = field(default_factory=lambda: {"dist": "triangular"})
    dist_disposal: dict = field(default_factory=lambda: {"dist": "triangular"})

    # --- struttura di dipendenza (copula) ---
    copula: str = "gaussian"         # "gaussian" | "t"
    copula_df: float = 8.0           # gradi di liberta' della t (bassi = code piu' spesse)

    # --- performance ---
    irr_subsample: int = 3000        # n. simulazioni per la curva IRR (npf.irr e' lento)


# ==========================================================================
# 3b. PREPARAZIONE INPUT (parte deterministica: unica fonte di verita')
# ==========================================================================
def _prep_inputs(df):
    """Legge dal DataFrame tutte le grandezze deterministiche e le terne (min/mode/max),
    e calcola una sola volta lo schema del debito. Usato sia dalla simulazione sia
    dalle analisi di sensibilita', cosi' non possono divergere."""
    years = df.shape[0]

    def trip(key):
        a, b, c = REQUIRED_TRIPLES[key]
        mn = _col(df, a, 0.0, years)
        md = _col(df, b, 0.0, years)
        mx = _col(df, c, 0.0, years)
        lo = np.minimum.reduce([mn, md, mx])
        hi = np.maximum.reduce([mn, md, mx])
        mid = mn + md + mx - lo - hi
        return lo, mid, hi

    costs_fixed    = _col(df, SINGLE_COLS["costs_fixed"], 0.0, years)
    amort          = _col(df, SINGLE_COLS["amort"], 0.0, years)
    capex          = _col(df, SINGLE_COLS["capex"], 0.0, years)
    change_wc      = _col(df, SINGLE_COLS["change_wc"], 0.0, years)
    debt_inflow    = _col(df, SINGLE_COLS["debt_inflow"], 0.0, years)
    debt_repayment = _col(df, SINGLE_COLS["debt_repayment"], 0.0, years)
    interest_rate  = _col(df, SINGLE_COLS["interest_rate"], 0.05, years)

    interest = np.zeros(years)
    stock = 0.0
    for y in range(years):
        interest[y] = -stock * interest_rate[y]
        stock = max(stock + debt_inflow[y] - debt_repayment[y], 0.0)

    return {
        "years": years,
        "years_col": df.iloc[:, 0].to_numpy(),
        "revenue": trip("revenue"),
        "cost": trip("cost_var"),
        "disposal": trip("disposal"),
        "costs_fixed": costs_fixed, "amort": amort, "capex": capex, "change_wc": change_wc,
        "debt_inflow": debt_inflow, "debt_repayment": debt_repayment,
        "interest": interest, "net_borrowing": debt_inflow - debt_repayment, "debt_end": stock,
    }


# ==========================================================================
# 4. MOTORE DI SIMULAZIONE
# ==========================================================================
def run_simulations(df, cfg: SimConfig):
    rng = np.random.default_rng(None if cfg.seed == 0 else int(cfg.seed))
    prep = _prep_inputs(df)
    years = prep["years"]
    years_col = prep["years_col"]
    n_sim = int(cfg.n_sim)

    rev_lo, rev_mid, rev_hi = prep["revenue"]
    cs_lo, cs_mid, cs_hi = prep["cost"]
    dp_lo, dp_mid, dp_hi = prep["disposal"]

    costs_fixed    = prep["costs_fixed"]
    amort          = prep["amort"]
    capex          = prep["capex"]
    change_wc      = prep["change_wc"]
    debt_repayment = prep["debt_repayment"]
    interest       = prep["interest"]        # negativo = interessi pagati (deterministico)
    debt_end       = prep["debt_end"]        # debito residuo a fine orizzonte
    net_borrowing  = prep["net_borrowing"]   # nuovo debito netto per anno

    # ---- estrazioni CORRELATE (copula gaussiana o t + AR1) ----
    R = np.array([
        [1.0,               cfg.corr_rev_cost, cfg.corr_rev_disp],
        [cfg.corr_rev_cost, 1.0,               cfg.corr_cost_disp],
        [cfg.corr_rev_disp, cfg.corr_cost_disp, 1.0],
    ])
    u = correlated_uniforms(n_sim, years, R, cfg.persistence, rng,
                            copula=cfg.copula, copula_df=cfg.copula_df)  # [n_sim, years, 3]

    # ---- marginali selezionabili per fattore (la copula resta invariata) ----
    revenue_orig = sample_marginal(u[:, :, 0], cfg.dist_revenue, rev_lo, rev_mid, rev_hi)
    cs_orig      = sample_marginal(u[:, :, 1], cfg.dist_cost, cs_lo, cs_mid, cs_hi)
    disposal     = sample_marginal(u[:, :, 2], cfg.dist_disposal, dp_lo, dp_mid, dp_hi)  # non shiftato
    capex_orig   = np.broadcast_to(capex, (n_sim, years)).copy()

    # ---- SHIFT temporale a livello di PROGETTO ----
    # Un'unica estrazione di ritardo per (sim, anno) applicata a ricavi, costi e capex:
    # se il progetto slitta, slittano INSIEME. Le percentuali possono differire per voce.
    delay_severity = None
    if cfg.enable_shift:
        probs = np.asarray(cfg.shift_probs, float)
        probs = probs / probs.sum()
        n_shift = rng.choice(np.array([0, 1, 2]), size=(n_sim, years), p=probs)
        delay_severity = n_shift.sum(axis=1)                     # anni totali di ritardo per sim
        target = np.minimum(np.arange(years)[None, :] + n_shift, years - 1)
        i_idx = np.broadcast_to(np.arange(n_sim)[:, None], (n_sim, years)).ravel()
        j_idx = target.ravel()

        def _shift(flow, pct):
            moved = flow * (pct / 100.0)
            out = flow - moved
            np.add.at(out, (i_idx, j_idx), moved.ravel())  # accumula i pezzi spostati
            return out

        revenue = _shift(revenue_orig, cfg.shift_rev_pct)
        cs      = _shift(cs_orig, cfg.shift_cs_pct)
        capex_s = _shift(capex_orig, cfg.shift_capex_pct)
    else:
        revenue, cs, capex_s = revenue_orig.copy(), cs_orig.copy(), capex_orig.copy()

    # ---- CONTO ECONOMICO e CASSA (vettorizzato su tutte le simulazioni) ----
    cf = costs_fixed[None, :]
    am = amort[None, :]
    it = interest[None, :]
    wc = change_wc[None, :]

    ebitda = revenue + cs + cf          # costi gia' negativi
    ebit   = ebitda + am                # amort gia' negativo
    ebt    = ebit + it                  # interessi gia' negativi (base imponibile levered)

    tax_unlev = np.maximum(ebit, 0.0) * cfg.tax_rate   # imposta UNLEVERED (su EBIT)
    tax_lev   = np.maximum(ebt, 0.0) * cfg.tax_rate    # imposta LEVERED (su EBT) -> scudo fiscale
    # Nota: si assume nessun credito d'imposta quando la base e' negativa (no loss carry-forward).

    # CFADS: cassa disponibile PRIMA del servizio del debito, con imposte cash (levered).
    cfads = ebitda - tax_lev + capex_s + disposal + wc

    # FCFF (unlevered): nessun flusso di finanziamento, imposta su EBIT -> si sconta al WACC.
    fcff = ebitda - tax_unlev + capex_s + disposal + wc
    # FCFE (levered): CFADS - servizio del debito + nuovo debito -> si sconta al Ke.
    fcfe = cfads + it + net_borrowing[None, :]

    fcf = fcff if cfg.framework.upper() == "FCFF" else fcfe

    # ---- ATTUALIZZAZIONE ----
    disc = (1.0 + cfg.discount_rate) ** np.arange(1, years + 1)
    fcf_pv = fcf / disc[None, :]
    npv = fcf_pv.sum(axis=1)

    # ---- VALORE TERMINALE ----
    tv = np.zeros(n_sim)
    tv_warning = None
    method = cfg.tv_method.lower()
    if method == "gordon":
        if cfg.discount_rate > cfg.tv_growth:
            # Gordon sul flusso dell'ultimo anno, coerente col framework selezionato
            tv = fcf[:, -1] * (1.0 + cfg.tv_growth) / (cfg.discount_rate - cfg.tv_growth)
        else:
            tv_warning = "Gordon non valido: g >= tasso di sconto. Valore terminale impostato a 0."
    elif method == "multiple":
        tv = cfg.tv_multiple * ebitda[:, -1]                 # multiplo su EBITDA -> enterprise value
        if cfg.framework.upper() == "FCFE":
            tv = tv - debt_end                               # da EV a equity value: - debito residuo
    tv_pv = tv / disc[-1]
    npv = npv + tv_pv

    # ---- DSCR (definizione corretta) ----
    debt_service = -it + debt_repayment[None, :]             # interessi + quota capitale (magnitudo)
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(debt_service > 1e-9, cfads / debt_service, np.nan)
    has_debt = bool(np.isfinite(dscr).any())

    # ---- campioni per driver (per l'analisi di sensibilita' Monte Carlo) ----
    # aggrego ogni fattore stocastico come contributo attualizzato: monotono con l'NPV.
    driver_samples = {
        "Ricavi": (revenue / disc[None, :]).sum(axis=1),
        "Costi variabili": (cs / disc[None, :]).sum(axis=1),
        "Disposal": (disposal / disc[None, :]).sum(axis=1),
    }
    if cfg.enable_shift and delay_severity is not None:
        driver_samples["Ritardo (anni)"] = delay_severity.astype(float)

    return {
        "years_col": years_col,
        "npv": npv,
        "fcf": fcf,
        "fcf_pv": fcf_pv,
        "cfads": cfads,
        "ebitda": ebitda,
        "dscr": dscr,
        "has_debt": has_debt,
        "tv_pv": tv_pv,
        "tv_warning": tv_warning,
        "debt_end": debt_end,
        "framework": cfg.framework.upper(),
        "driver_samples": driver_samples,
        # medie per il grafico "originale vs shift"
        "revenue_orig_mean": revenue_orig.mean(axis=0),
        "revenue_shift_mean": revenue.mean(axis=0),
        "cs_orig_mean": cs_orig.mean(axis=0),
        "cs_shift_mean": cs.mean(axis=0),
        "capex_orig_mean": capex_orig.mean(axis=0),
        "capex_shift_mean": capex_s.mean(axis=0),
    }


# ==========================================================================
# 5. METRICHE DERIVATE (payback, IRR, PPI)
# ==========================================================================
def compute_payback(fcf_pv):
    """Payback ATTUALIZZATO: anni fino a cumulato PV >= 0, con interpolazione lineare."""
    n_sim, years = fcf_pv.shape
    cum = np.cumsum(fcf_pv, axis=1)
    pb = np.full(n_sim, np.nan)
    for i in range(n_sim):
        c = cum[i]
        for j in range(years):
            if c[j] >= 0:
                prev = c[j - 1] if j > 0 else 0.0
                denom = c[j] - prev
                pb[i] = j + (-prev / denom if denom != 0 else 0.0)
                break
    return pb


def compute_irr_curve(fcf, subsample):
    """IRR cumulata per anno su un sottocampione (npf.irr e' iterativo e costoso).
    Le simulazioni sono i.i.d., quindi i primi `subsample` sono rappresentativi."""
    import numpy_financial as npf
    n_sim, years = fcf.shape
    k = min(int(subsample), n_sim)
    irr = np.zeros((k, years))
    for i in range(k):
        row = fcf[i]
        for j in range(years):
            sub = row[:j + 1]
            if np.any(sub < 0) and np.any(sub > 0):
                r = npf.irr(sub)
                irr[i, j] = np.clip(r, -1, 5) if (r is not None and np.isfinite(r)) else 0.0
    return irr


def compute_ppi(fcf_pv, df, discount_rate):
    """Profitability index cumulato: NPV cumulato / costo cumulato (PV)."""
    n_years = fcf_pv.shape[1]
    costs_fixed = _col(df, SINGLE_COLS["costs_fixed"], 0.0, n_years)
    capex = _col(df, SINGLE_COLS["capex"], 0.0, n_years)
    disc = (1.0 + discount_rate) ** np.arange(1, n_years + 1)
    cost_total = (np.abs(costs_fixed) + np.abs(capex)) / disc
    cost_cum = np.cumsum(cost_total)
    cost_cum = np.where(cost_cum == 0, np.nan, cost_cum)
    npv_cum = np.cumsum(fcf_pv, axis=1)
    return npv_cum / cost_cum[None, :]


def dscr_rating(x):
    """Rating sintetico dal DSCR medio."""
    if not np.isfinite(x):
        return "N/A (nessun debito)"
    if x < 1:
        return "🔴 Default Risk"
    if x < 1.2:
        return "🟠 Weak"
    if x < 1.5:
        return "🟡 Acceptable"
    return "🟢 Strong"


# ==========================================================================
# 6. SENSIBILITA' E DRIVER (cosa muove l'NPV)
# ==========================================================================
def _shift_deterministic(flow, k):
    """Trasla un vettore per-anno di k anni in avanti (clamp all'ultimo anno)."""
    flow = np.asarray(flow, float)
    if k <= 0:
        return flow.copy()
    out = np.zeros_like(flow)
    idx = np.minimum(np.arange(flow.size) + k, flow.size - 1)
    np.add.at(out, idx, flow)
    return out


def _npv_scenario(prep, revenue, cost, disposal, capex,
                  framework, discount_rate, tax_rate, tv_method, tv_growth, tv_multiple):
    """NPV DETERMINISTICO di un singolo scenario (flussi per-anno gia' risolti).
    Stesse formule della simulazione, ma senza dimensione Monte Carlo. Riusa lo
    schema del debito da _prep_inputs, quindi resta coerente con run_simulations."""
    years = prep["years"]
    cf = prep["costs_fixed"]; am = prep["amort"]; it = prep["interest"]
    wc = prep["change_wc"]; nb = prep["net_borrowing"]; debt_end = prep["debt_end"]

    ebitda = revenue + cost + cf
    ebit = ebitda + am
    ebt = ebit + it
    tax_unlev = np.maximum(ebit, 0.0) * tax_rate
    tax_lev = np.maximum(ebt, 0.0) * tax_rate

    if framework.upper() == "FCFF":
        fcf = ebitda - tax_unlev + capex + disposal + wc
    else:
        cfads = ebitda - tax_lev + capex + disposal + wc
        fcf = cfads + it + nb

    disc = (1.0 + discount_rate) ** np.arange(1, years + 1)
    npv = float(np.sum(fcf / disc))

    m = tv_method.lower()
    tv = 0.0
    if m == "gordon" and discount_rate > tv_growth:
        tv = fcf[-1] * (1.0 + tv_growth) / (discount_rate - tv_growth)
    elif m == "multiple":
        tv = tv_multiple * ebitda[-1]
        if framework.upper() == "FCFE":
            tv = tv - debt_end
    npv += tv / disc[-1]
    return npv


def _factor_at(prep, cfg, factor, p):
    """Valore per-anno del fattore al percentile p, secondo la sua distribuzione scelta."""
    lo, mode, hi = prep[factor]
    spec = {"revenue": cfg.dist_revenue, "cost": cfg.dist_cost,
            "disposal": cfg.dist_disposal}[factor]
    u = np.full(prep["years"], float(p))
    return sample_marginal(u, spec, lo, mode, hi)


def tornado_oneway(df, cfg, p_low=0.10, p_high=0.90,
                   vary_discount=0.015, vary_tax=0.05, vary_growth=0.01, vary_multiple=1.0,
                   include_assumptions=True):
    """Tornado DETERMINISTICO: si varia un elemento alla volta tra 'basso' e 'alto'
    tenendo gli altri al valore 'piano'. Ordina le barre per ampiezza dello swing sull'NPV.

    - Fattori di rischio: portati ai percentili p_low / p_high della loro distribuzione.
    - Ritardo: da 0 al ritardo massimo previsto.
    - Ipotesi scalari (opz.): tasso di sconto, aliquota, valore terminale.
    """
    prep = _prep_inputs(df)
    rev0 = prep["revenue"][1]; cost0 = prep["cost"][1]; disp0 = prep["disposal"][1]
    capex0 = prep["capex"]

    def npv_of(rev=rev0, cost=cost0, disp=disp0, capex=capex0, framework=cfg.framework,
               dr=cfg.discount_rate, tax=cfg.tax_rate, tvm=cfg.tv_method,
               tvg=cfg.tv_growth, tvx=cfg.tv_multiple):
        return _npv_scenario(prep, rev, cost, disp, capex, framework, dr, tax, tvm, tvg, tvx)

    base = npv_of()
    bars = []

    # --- fattori di rischio (usano la distribuzione scelta) ---
    for factor, label in [("revenue", "Ricavi"), ("cost", "Costi variabili"),
                          ("disposal", "Disposal")]:
        flo = _factor_at(prep, cfg, factor, p_low)
        fhi = _factor_at(prep, cfg, factor, p_high)
        kw_lo = {"revenue": "rev", "cost": "cost", "disposal": "disp"}[factor]
        bars.append({"label": label,
                     "low": npv_of(**{kw_lo: flo}),
                     "high": npv_of(**{kw_lo: fhi})})

    # --- ritardo di progetto ---
    if cfg.enable_shift:
        kmax = len(cfg.shift_probs) - 1
        if kmax > 0:
            bars.append({
                "label": f"Ritardo (+{kmax} anni)",
                "low": npv_of(rev=_shift_deterministic(rev0, kmax),
                              cost=_shift_deterministic(cost0, kmax),
                              capex=_shift_deterministic(capex0, kmax)),
                "high": base,
            })

    # --- ipotesi scalari ---
    if include_assumptions:
        bars.append({"label": f"Tasso di sconto (±{vary_discount:.1%})",
                     "low": npv_of(dr=cfg.discount_rate + vary_discount),
                     "high": npv_of(dr=max(cfg.discount_rate - vary_discount, 1e-6))})
        bars.append({"label": f"Aliquota (±{vary_tax:.0%})",
                     "low": npv_of(tax=min(cfg.tax_rate + vary_tax, 0.99)),
                     "high": npv_of(tax=max(cfg.tax_rate - vary_tax, 0.0))})
        if cfg.tv_method.lower() == "gordon":
            bars.append({"label": f"g terminale (±{vary_growth:.1%})",
                         "low": npv_of(tvg=cfg.tv_growth - vary_growth),
                         "high": npv_of(tvg=min(cfg.tv_growth + vary_growth,
                                                cfg.discount_rate - 1e-4))})
        elif cfg.tv_method.lower() == "multiple":
            bars.append({"label": f"Multiplo EBITDA (±{vary_multiple:g})",
                         "low": npv_of(tvx=max(cfg.tv_multiple - vary_multiple, 0.0)),
                         "high": npv_of(tvx=cfg.tv_multiple + vary_multiple)})

    for b in bars:
        b["swing"] = abs(b["high"] - b["low"])
    bars.sort(key=lambda d: d["swing"], reverse=True)
    return {"base": base, "bars": bars}


def driver_importance(res):
    """Importanza dei driver dal Monte Carlo: correlazione di rango (Spearman) tra ogni
    driver e l'NPV, piu' un contributo normalizzato (rho^2 a somma 100%).

    La correlazione di rango e' robusta con input correlati (a differenza della
    scomposizione di varianza classica) ed e' il tornado 'a coefficienti' usato dai
    tool professionali. Il segno indica la direzione dell'effetto sull'NPV."""
    from scipy.stats import rankdata
    npv = np.asarray(res["npv"], float)
    samples = res.get("driver_samples", {})
    npv_rank = rankdata(npv)
    out = []
    for name, vals in samples.items():
        v = np.asarray(vals, float)
        if v.size == 0 or np.allclose(v, v.flat[0]):
            rho = 0.0
        else:
            rho = float(np.corrcoef(rankdata(v), npv_rank)[0, 1])
            if not np.isfinite(rho):
                rho = 0.0
        out.append({"driver": name, "spearman": rho})
    ss = sum(d["spearman"] ** 2 for d in out) or 1.0
    for d in out:
        d["contribution"] = d["spearman"] ** 2 / ss
    out.sort(key=lambda d: abs(d["spearman"]), reverse=True)
    return out


TWO_WAY_PARAMS = ("discount_rate", "tax_rate", "tv_growth", "tv_multiple",
                  "scale_revenue", "scale_cost", "scale_capex")


def two_way_sensitivity(df, cfg, x_param, x_values, y_param, y_values):
    """Griglia di NPV (deterministico, fattori al 'piano') al variare di DUE ipotesi.
    param ammessi: vedi TWO_WAY_PARAMS. Gli 'scale_*' sono moltiplicatori (1.0 = piano).
    Ritorna una matrice [len(y_values), len(x_values)]."""
    prep = _prep_inputs(df)
    rev0 = prep["revenue"][1]; cost0 = prep["cost"][1]; disp0 = prep["disposal"][1]
    capex0 = prep["capex"]

    def npv_with(ov):
        rev = rev0 * ov.get("scale_revenue", 1.0)
        cost = cost0 * ov.get("scale_cost", 1.0)
        capex = capex0 * ov.get("scale_capex", 1.0)
        return _npv_scenario(prep, rev, cost, disp0, capex, cfg.framework,
                             ov.get("discount_rate", cfg.discount_rate),
                             ov.get("tax_rate", cfg.tax_rate), cfg.tv_method,
                             ov.get("tv_growth", cfg.tv_growth),
                             ov.get("tv_multiple", cfg.tv_multiple))

    grid = np.zeros((len(y_values), len(x_values)))
    for iy, yv in enumerate(y_values):
        for ix, xv in enumerate(x_values):
            grid[iy, ix] = npv_with({x_param: xv, y_param: yv})
    return grid


# ==========================================================================
# 7. COSTO DEL CAPITALE (CAPM / WACC)
# ==========================================================================
def relever_beta(beta_unlevered, tax_rate, debt_equity, beta_debt=0.0):
    """Rileveraggio di Hamada (con beta del debito opzionale).
    beta_L = beta_U + (beta_U - beta_D) * (1 - t) * (D/E).
    Con beta_D = 0 diventa il classico beta_U * [1 + (1 - t) * D/E]."""
    return beta_unlevered + (beta_unlevered - beta_debt) * (1.0 - tax_rate) * debt_equity


def unlever_beta(beta_levered, tax_rate, debt_equity, beta_debt=0.0):
    """Deleveraggio (inverso di Hamada). Serve se parti da un beta levered di mercato
    e vuoi risalire al beta operativo (asset beta)."""
    return (beta_levered + beta_debt * (1.0 - tax_rate) * debt_equity) / \
           (1.0 + (1.0 - tax_rate) * debt_equity)


def capm_cost_of_equity(rf, erp, beta_levered, extra_premium=0.0):
    """CAPM: Ke = Rf + beta_L * ERP (+ eventuali premi: country, size, specific).
    Se l'ERP e' gia' 'totale paese' (mature + country risk premium), non aggiungere il CRP due volte."""
    return rf + beta_levered * erp + extra_premium


def wacc(ke, kd_pretax, tax_rate, debt_equity):
    """WACC con pesi a valore (target). debt_equity = D/E.
    Se D/E = 0 -> WACC = Ke (il ramo 'senza debito' cade fuori da solo)."""
    de = debt_equity
    wd = de / (1.0 + de)          # D / (D + E)
    we = 1.0 - wd                 # E / (D + E)
    return we * ke + wd * kd_pretax * (1.0 - tax_rate)


def cost_of_capital(framework, rf, erp, tax_rate, debt_equity,
                    listed=False, beta_levered=None, beta_unlevered=None,
                    kd_pretax=0.0, beta_debt=0.0, extra_premium=0.0):
    """Calcola Ke (CAPM) e WACC e restituisce il tasso di sconto coerente col framework.

      listed=True  -> usi il beta levered osservato sul mercato (beta_levered).
      listed=False -> parti dal beta unlevered di settore (Damodaran) e lo rilevraggi
                      alla struttura D/E dell'impresa (beta_unlevered).

    Regola sul tasso:
      FCFF -> WACC   (che coincide con Ke se D/E = 0)
      FCFE -> Ke

    Nota: il CAPM (Ke) serve SEMPRE, anche come input del WACC. Non e' un'alternativa al WACC.
    """
    de = max(float(debt_equity), 0.0)
    if listed:
        if beta_levered is None:
            raise ValueError("Impresa quotata: serve beta_levered.")
        bL = float(beta_levered)
        bU = unlever_beta(bL, tax_rate, de, beta_debt)
    else:
        if beta_unlevered is None:
            raise ValueError("Impresa non quotata: serve beta_unlevered (settore Damodaran).")
        bU = float(beta_unlevered)
        bL = relever_beta(bU, tax_rate, de, beta_debt)

    ke = capm_cost_of_equity(rf, erp, bL, extra_premium)
    w = wacc(ke, kd_pretax, tax_rate, de)
    rate = w if framework.upper() == "FCFF" else ke
    return {
        "beta_unlevered": bU,
        "beta_levered": bL,
        "ke": ke,
        "kd_after_tax": kd_pretax * (1.0 - tax_rate),
        "wacc": w,
        "discount_rate": rate,
        "framework": framework.upper(),
        "has_debt": de > 0,
    }


# ==========================================================================
# 8. SMOKE TEST (esegui: python3 dcf_core.py)
# ==========================================================================
if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("SMOKE TEST dcf_core")
    print("=" * 70)

    # --- dataset sintetico: progetto a 6 anni con debito ---
    years = 6
    df = pd.DataFrame({
        "Anno": np.arange(2026, 2026 + years),
        "Revenues min":   [0,  80, 180, 260, 300, 320],
        "Revenues piano": [0, 100, 220, 320, 380, 400],
        "Revenues max":   [0, 130, 270, 390, 460, 500],
        "Cost var min":   [0,  -70, -150, -210, -240, -250],   # negativi
        "Cost var piano": [0,  -55, -120, -170, -200, -210],
        "Cost var max":   [0,  -40,  -90, -130, -150, -160],
        "Costs fixed":    [-20, -25, -30, -30, -30, -30],       # negativi
        "Amort, & Depreciation": [0, -40, -40, -40, -40, -40],  # negativi
        "Capex":          [-200, -50, 0, 0, 0, 0],              # negativi
        "Disposal & Capex Saving min": [0, 0, 0, 0, 0, 10],
        "Disposal & Capex Saving":     [0, 0, 0, 0, 0, 20],
        "Disposal & Capex Saving max": [0, 0, 0, 0, 0, 35],
        "Change in working cap,": [0, -10, -15, -8, -5, 5],
        "Debt inflow":    [150, 0, 0, 0, 0, 0],
        "Debt repayment": [0, 30, 30, 30, 30, 30],
        "Interest rate":  [0.05] * years,
    })

    ok, missing = validate_columns(df)
    print(f"\nValidazione colonne: ok={ok}, mancanti={missing}")

    # ---- TEST 1: coerenza NPV = somma flussi scontati + TV ----
    cfg = SimConfig(framework="FCFE", discount_rate=0.10, n_sim=20000, seed=42,
                    tv_method="gordon", tv_growth=0.02)
    res = run_simulations(df, cfg)
    recompute = res["fcf_pv"].sum(axis=1) + res["tv_pv"]
    err = np.max(np.abs(res["npv"] - recompute))
    print(f"\n[TEST 1] NPV == sum(FCF_PV) + TV_PV   -> max err = {err:.2e}  "
          f"{'OK' if err < 1e-6 else 'FAIL'}")

    # ---- TEST 2: FCFE vs FCFF (identita' del bridge quando basi imponibili > 0) ----
    cfg_ff = SimConfig(framework="FCFF", n_sim=20000, seed=42, enable_shift=False,
                       persistence=0.0, corr_rev_cost=0.5, tv_method="none")
    cfg_fe = SimConfig(framework="FCFE", n_sim=20000, seed=42, enable_shift=False,
                       persistence=0.0, corr_rev_cost=0.5, tv_method="none")
    r_ff = run_simulations(df, cfg_ff)
    r_fe = run_simulations(df, cfg_fe)
    # bridge: FCFE = FCFF - interessi*(1-t) + nuovo debito, quando EBIT>0 e EBT>0
    print(f"\n[TEST 2] FCFF medio  = {r_ff['fcf'].mean(axis=0).round(1)}")
    print(f"         FCFE medio  = {r_fe['fcf'].mean(axis=0).round(1)}")
    print(f"         (FCFE include incasso debito anno 1 e servizio del debito: atteso)")

    # ---- TEST 3: DSCR ben definito e maggiore del vecchio numeratore sbagliato ----
    dscr = r_fe["dscr"]
    cfads = r_fe["cfads"]
    fcf_lev = r_fe["fcf"]
    # ricostruisco il servizio del debito per confronto
    print(f"\n[TEST 3] has_debt = {r_fe['has_debt']}")
    print(f"         DSCR medio (nuovo, CFADS/DS)      = {np.nanmean(dscr):.2f}")
    print(f"         'DSCR' col vecchio numeratore FCFE = "
          f"{np.nanmean(np.where(np.isfinite(dscr), fcf_lev/np.maximum(cfads-fcf_lev,1e-9), np.nan)):.2f} "
          f"(sottostimato, come atteso)")

    # ---- TEST 4: la correlazione riduce la diversificazione -> code piu' spesse ----
    base = SimConfig(n_sim=40000, seed=7, enable_shift=False, tv_method="none")
    indep = SimConfig(**{**base.__dict__, "corr_rev_cost": 0.0, "persistence": 0.0})
    corr  = SimConfig(**{**base.__dict__, "corr_rev_cost": 0.8, "persistence": 0.6})
    npv_indep = run_simulations(df, indep)["npv"]
    npv_corr  = run_simulations(df, corr)["npv"]
    print(f"\n[TEST 4] Std(NPV) indipendente = {npv_indep.std():.1f}")
    print(f"         Std(NPV) correlato     = {npv_corr.std():.1f}   "
          f"({'OK: code piu spesse' if npv_corr.std() > npv_indep.std() else 'controllare'})")
    print(f"         VaR5 indipendente = {np.percentile(npv_indep,5):.1f} | "
          f"VaR5 correlato = {np.percentile(npv_corr,5):.1f}")

    # ---- TEST 5: lo shift e' condiviso tra le voci (stessa struttura di ritardo) ----
    cfg_shift = SimConfig(n_sim=5000, seed=1, enable_shift=True,
                          shift_probs=(0.0, 1.0, 0.0),  # ritardo deterministico di +1 anno
                          shift_rev_pct=100, shift_cs_pct=100, shift_capex_pct=100,
                          tv_method="none")
    rs = run_simulations(df, cfg_shift)
    # con +1 anno certo, la media shiftata deve essere la originale traslata di un anno
    rev_o = rs["revenue_orig_mean"]; rev_s = rs["revenue_shift_mean"]
    shifted_expected = np.concatenate([[0.0], rev_o[:-1]])
    shifted_expected[-1] += rev_o[-1]  # l'ultimo anno accumula (clamp)
    err_shift = np.max(np.abs(rev_s - shifted_expected))
    print(f"\n[TEST 5] Shift +1 anno certo, ricavi traslati -> max err = {err_shift:.2e} "
          f"{'OK' if err_shift < 1e-6 else 'FAIL'}")

    # ---- TEST 6: metriche derivate girano senza errori ----
    pb = compute_payback(r_fe["fcf_pv"])
    irr = compute_irr_curve(r_fe["fcf"], subsample=1000)
    ppi = compute_ppi(r_fe["fcf_pv"], df, 0.10)
    print(f"\n[TEST 6] payback mediano = {np.nanmedian(pb):.2f} anni | "
          f"IRR p50 ultimo anno = {np.nanpercentile(irr[:, -1],50):.1%} | "
          f"PPI p50 ultimo anno = {np.nanpercentile(ppi[:, -1],50):.2f}")

    # ---- TEST 7: costo del capitale (CAPM / WACC) ----
    print("\n[TEST 7] Costo del capitale")
    # esempio: beta_U=0.75, Rf=3%, ERP=6%, t=25%, D/E=0.6, Kd=5%
    coc = cost_of_capital("FCFE", rf=0.03, erp=0.06, tax_rate=0.25, debt_equity=0.6,
                          listed=False, beta_unlevered=0.75, kd_pretax=0.05)
    print(f"         beta_L = {coc['beta_levered']:.4f} (atteso 1.0875) | "
          f"Ke = {coc['ke']:.4%} (atteso 9.5250%)")
    coc_ff = cost_of_capital("FCFF", rf=0.03, erp=0.06, tax_rate=0.25, debt_equity=0.6,
                             listed=False, beta_unlevered=0.75, kd_pretax=0.05)
    print(f"         WACC = {coc_ff['wacc']:.4%} (atteso ~7.3594%) | "
          f"tasso FCFF = {coc_ff['discount_rate']:.4%}")
    # no debito: WACC deve coincidere con Ke
    coc0 = cost_of_capital("FCFF", rf=0.03, erp=0.06, tax_rate=0.25, debt_equity=0.0,
                           listed=False, beta_unlevered=0.75, kd_pretax=0.05)
    same = abs(coc0["wacc"] - coc0["ke"]) < 1e-12
    print(f"         D/E=0 -> WACC({coc0['wacc']:.4%}) == Ke({coc0['ke']:.4%})  "
          f"{'OK' if same else 'FAIL'}")
    # roundtrip: unlever(relever(bU)) == bU
    bU = 0.9
    bL = relever_beta(bU, 0.27, 0.8, beta_debt=0.1)
    bU2 = unlever_beta(bL, 0.27, 0.8, beta_debt=0.1)
    print(f"         roundtrip unlever/relever: {bU} -> {bL:.4f} -> {bU2:.4f}  "
          f"{'OK' if abs(bU-bU2)<1e-12 else 'FAIL'}")

    # ---- TEST 8: distribuzioni, copula t e sensibilita' ----
    print("\n[TEST 8] Distribuzioni / copula / sensibilita'")
    # 8a: le marginali producono forme diverse ma restano nel supporto [min,max] dove limitato
    uu = np.linspace(0.001, 0.999, 100001)
    tri = triangular_ppf(uu, 0.0, 3.0, 10.0)
    prt = pert_ppf(uu, 0.0, 3.0, 10.0)
    unf = uniform_ppf(uu, 0.0, 10.0)
    print(f"         media triangolare={tri.mean():.3f} | PERT={prt.mean():.3f} "
          f"(PERT piu' vicina alla moda) | uniforme={unf.mean():.3f}")
    print(f"         supporti in [0,10]: tri[{tri.min():.2f},{tri.max():.2f}] "
          f"pert[{prt.min():.2f},{prt.max():.2f}] "
          f"{'OK' if tri.min()>=-1e-6 and tri.max()<=10+1e-6 and prt.min()>=-1e-6 and prt.max()<=10+1e-6 else 'FAIL'}")
    # lognormale su costi negativi: segno preservato
    ln = lognormal_ppf_range(np.array([0.5, 0.95]), -250.0, -210.0, -160.0)
    print(f"         lognormale su costi negativi -> {ln.round(1)} (segno negativo preservato) "
          f"{'OK' if np.all(ln < 0) else 'FAIL'}")

    # 8b: la copula t produce code CONGIUNTE piu' pesanti (VaR 1%) della gaussiana
    base8 = SimConfig(n_sim=80000, seed=11, enable_shift=False, tv_method="none",
                      corr_rev_cost=0.7, persistence=0.3)
    g = run_simulations(df, SimConfig(**{**base8.__dict__, "copula": "gaussian"}))
    t = run_simulations(df, SimConfig(**{**base8.__dict__, "copula": "t", "copula_df": 3.0}))
    v_g, v_t = np.percentile(g["npv"], 1), np.percentile(t["npv"], 1)
    print(f"         VaR 1% NPV: gaussiana={v_g:.1f} | t(df=3)={v_t:.1f}  "
          f"(std quasi uguale {g['npv'].std():.0f}~{t['npv'].std():.0f}, ma coda t piu' spessa) "
          f"{'OK' if v_t < v_g else 'controllare'}")

    # 8c: driver importance su un run con distribuzioni miste
    cfg8 = SimConfig(n_sim=40000, seed=5, framework="FCFE", tv_method="none",
                     dist_revenue={"dist": "pert"}, dist_cost={"dist": "lognormal"},
                     dist_disposal={"dist": "uniform"}, corr_rev_cost=0.5)
    res8 = run_simulations(df, cfg8)
    imp = driver_importance(res8)
    print("         Driver per importanza (|Spearman|):")
    for d in imp:
        print(f"            {d['driver']:<18} rho={d['spearman']:+.3f}  contributo={d['contribution']:.0%}")

    # 8d: tornado deterministico
    tor = tornado_oneway(df, cfg8, p_low=0.10, p_high=0.90)
    print(f"         Tornado (base NPV={tor['base']:.1f}), barre per swing:")
    for b in tor["bars"][:5]:
        print(f"            {b['label']:<24} [{b['low']:.1f} .. {b['high']:.1f}]  swing={b['swing']:.1f}")

    # 8e: sensibilita' a due vie (tasso di sconto x g terminale) su un caso Gordon
    cfg8g = SimConfig(framework="FCFE", tv_method="gordon", tv_growth=0.02, discount_rate=0.10)
    grid = two_way_sensitivity(df, cfg8g, "discount_rate", [0.08, 0.10, 0.12],
                               "tv_growth", [0.00, 0.02])
    print(f"         Griglia NPV 2 vie (righe=g, col=tasso): shape={grid.shape}, "
          f"monotona nel tasso={'OK' if np.all(np.diff(grid, axis=1) < 0) else 'controllare'}")

    print("\n" + "=" * 70)
    print("Metriche di sintesi (FCFE, con Gordon):")
    print(f"  Expected NPV      = {res['npv'].mean():,.1f}")
    print(f"  VaR 95% (CaR)     = {np.percentile(res['npv'],5):,.1f}")
    print(f"  P(NPV<0)          = {np.mean(res['npv']<0):.1%}")
    print(f"  DSCR medio        = {np.nanmean(res['dscr']):.2f}  -> {dscr_rating(np.nanmean(res['dscr']))}")
    print("=" * 70)
    print("TUTTI I TEST ESEGUITI.")
