"""
llm_commentary.py
=================
Sintesi e discussione dei risultati DCF generate da un LLM (OpenAI di default),
GROUNDED sui numeri del motore.

Principio: il motore calcola le cifre e le impacchetta in un riepilogo strutturato
(build_results_digest); l'LLM riceve SOLO quel riepilogo e scrive il layer narrativo /
decisionale, con l'istruzione esplicita di non inventare numeri. Le tabelle e i grafici
del report restano deterministici. Cosi' si ha prosa da analista + numeri a prova di revisore.

Robusto: se manca la chiave API o la chiamata fallisce, ritorna None e il chiamante
(report o app) usa il testo template. Non solleva mai eccezioni verso l'esterno.

Uso:
    from llm_commentary import generate_investment_commentary
    testo = generate_investment_commentary(res, cfg, df, reliability=ci, model="gpt-4o")
    if testo: st.markdown(testo)
"""

import os
import numpy as np

try:
    import dcf_core as core
except Exception:  # pragma: no cover
    core = None


# ==========================================================================
# 1. RIEPILOGO NUMERICO (deterministico) -> input per l'LLM
# ==========================================================================
def build_results_digest(res, cfg, df=None, reliability=None, currency="€",
                         project_description=None):
    """Assembla un riepilogo testuale strutturato dei risultati, da passare all'LLM.
    Contiene solo cifre calcolate dal motore: e' la 'verita'' su cui l'LLM deve poggiare.
    project_description: testo libero fornito dall'utente per la sezione descrittiva."""
    npv = np.asarray(res["npv"], float)
    L = []
    if project_description and str(project_description).strip():
        L.append("DESCRIZIONE FORNITA DEL PROGETTO (usala per la sezione descrittiva; "
                 "riformulala in modo chiaro ma NON aggiungere fatti non presenti):")
        L.append(str(project_description).strip())
        L.append("")
    L.append(f"VALUTAZIONE DCF MONTE CARLO — RIEPILOGO NUMERICO (valuta: {currency})")
    L.append(
        f"Impostazioni: framework {res.get('framework', cfg.framework)}; "
        f"tasso di sconto {cfg.discount_rate:.2%}; aliquota {cfg.tax_rate:.1%}; "
        f"{cfg.n_sim:,} simulazioni; campionamento "
        f"{'Latin Hypercube' if cfg.sampling == 'lhs' else 'Monte Carlo casuale'}; "
        f"copula {'t di Student (df=%g)' % cfg.copula_df if cfg.copula == 't' else 'gaussiana'}; "
        f"valore terminale {cfg.tv_method}."
    )
    L.append("")
    L.append("RISULTATI NPV:")
    L.append(f"- NPV atteso (media): {npv.mean():,.0f}")
    L.append(f"- Mediana: {np.median(npv):,.0f}")
    L.append(f"- VaR 95% (5o percentile): {np.percentile(npv, 5):,.0f}")
    L.append(f"- VaR 99% (1o percentile): {np.percentile(npv, 1):,.0f}")
    L.append(f"- Probabilita' di NPV negativo: {np.mean(npv < 0):.1%}")
    L.append(f"- Deviazione standard: {npv.std(ddof=1):,.0f}")
    if reliability:
        m = reliability["metrics"]
        L.append(
            f"- Intervalli di confidenza 95% (su {reliability['total_n']:,} simulazioni): "
            f"NPV atteso da {m['E[NPV] (media)']['ci_low']:,.0f} a {m['E[NPV] (media)']['ci_high']:,.0f}; "
            f"VaR 99% da {m['VaR 99% (P1)']['ci_low']:,.0f} a {m['VaR 99% (P1)']['ci_high']:,.0f}."
        )

    # driver (correlazione di rango)
    if core is not None:
        try:
            imp = core.driver_importance(res)
            if imp:
                L.append("")
                L.append("DRIVER DELL'NPV (correlazione di rango con l'NPV; segno = direzione; "
                         "contributo normalizzato a 100%):")
                for d in imp:
                    L.append(f"- {d['driver']}: rho={d['spearman']:+.2f}, contributo={d['contribution']:.0%}")
        except Exception:
            pass

        # tornado deterministico
        if df is not None:
            try:
                tor = core.tornado_oneway(df, cfg)
                L.append("")
                L.append(f"SENSIBILITA' (tornado; NPV base {tor['base']:,.0f}); "
                         f"NPV allo scenario basso -> alto, con ampiezza dello swing:")
                for b in tor["bars"][:6]:
                    L.append(f"- {b['label']}: da {b['low']:,.0f} a {b['high']:,.0f} (ampiezza {b['swing']:,.0f})")
            except Exception:
                pass

    # DSCR
    if res.get("has_debt") and res.get("dscr") is not None and core is not None:
        try:
            dscr = np.asarray(res["dscr"], float)
            dmin = float(np.nanmin(dscr)); dmean = float(np.nanmean(dscr))
            L.append("")
            L.append(f"COPERTURA DEL DEBITO: DSCR minimo {dmin:.2f}, medio {dmean:.2f}, "
                     f"giudizio '{core.dscr_rating(dmin)}'. (DSCR = CFADS / servizio del debito; "
                     f"sotto 1 il flusso non copre il servizio del debito.)")
        except Exception:
            pass

    # piano di finanziamento
    f = res.get("funding")
    if f:
        L.append("")
        L.append(
            f"PIANO DI FINANZIAMENTO: fabbisogno di costruzione {f['total_need']:,.0f}; "
            f"senior debt sul capex {f['debt_capex']:,.0f} (gearing {f['gearing_capex']:.0%}); "
            f"equity {f['total_equity']:,.0f}; interessi in costruzione (IDC) {f['idc_total']:,.0f}; "
            f"debito a fine costruzione {f['debt_at_cod']:,.0f}. "
            f"Il debito e' dimensionato sul caso base; i sovracosti di capex sono a carico dell'equity."
        )
    return "\n".join(L)


# ==========================================================================
# 2. PROMPT (analista investimenti, orientato alla decisione, GROUNDED)
# ==========================================================================
_SYSTEM_PROMPT_IT = """Sei un analista senior di investimenti e project finance. A partire da (1) una descrizione del progetto fornita dall'utente e (2) un riepilogo numerico dei risultati di una valutazione DCF Monte Carlo, scrivi tre sezioni per un report di comitato: la descrizione del progetto, una sintesi e un commento ai risultati.

REGOLE DI GROUNDING (fondamentali):
- La DESCRIZIONE del progetto deve basarsi ESCLUSIVAMENTE sul testo fornito dall'utente: riformulala in modo chiaro e sintetico, senza inventare fatti, cifre o dettagli non presenti. Se la descrizione fornita e' assente o troppo scarna, scrivi una riga neutra ("Descrizione del progetto non fornita in dettaglio") invece di inventare.
- La SINTESI e il COMMENTO ai risultati devono usare ESCLUSIVAMENTE i numeri del riepilogo. Non inventare cifre, non introdurre dati non forniti. Interpreta i numeri (cosa implicano per la decisione), non limitarti a ripeterli.
- NON scrivere una sezione metodologica: la nota sul modello e' fissa e viene aggiunta separatamente. Non descrivere come funziona il modello.

STRUTTURA (usa esattamente questi titoli in markdown):
## Descrizione del progetto
Cosa e' il progetto/investimento, in base al testo fornito. Chiaro e conciso.

## Sintesi
2-4 frasi: il verdetto sul valore (NPV atteso e sua incertezza), la probabilita' di perdita, e la conclusione operativa in una riga.

## Commento ai risultati
Discussione densa e orientata alla decisione: i driver principali (dalla classifica di importanza) e la sensibilita' (dal tornado), con direzione e ordine di grandezza; il downside (VaR 95% e 99%, probabilita' di NPV negativo); se presenti, la struttura di finanziamento (gearing, equity/debito, IDC) e la copertura del debito (DSCR e giudizio); se la copula e' t di Student, che il rischio e' concentrato nelle code. Chiudi con una posizione chiara (procedere / procedere con condizioni / ristrutturare / non procedere) e, in una sola frase, il "so what": "Se procediamo cosi', accettiamo una probabilita' di NPV negativo del ___ con un downside (VaR 95%) di ___."

TONO: sobrio, professionale, in italiano. E' supporto alla decisione basato sui risultati del modello, non una garanzia. Lunghezza complessiva: circa 300-450 parole."""


# ==========================================================================
# 3. GENERAZIONE (con fallback robusto)
# ==========================================================================
def generate_investment_commentary(res, cfg, df=None, reliability=None, currency="€",
                                   model="gpt-4o", api_key=None, temperature=0.3,
                                   provider="openai", project_description=None):
    """Ritorna la sintesi discorsiva (markdown) generata dall'LLM, oppure None se l'LLM
    non e' disponibile (chiave assente, libreria assente, errore di rete). NON solleva
    eccezioni: il chiamante usa il testo template quando riceve None.

    project_description: testo libero dell'utente per la sezione descrittiva del progetto.
    provider: 'openai' (default) | 'anthropic' (Claude), stessa logica di grounding."""
    digest = build_results_digest(res, cfg, df, reliability, currency, project_description)
    try:
        if provider == "anthropic":
            return _call_anthropic(digest, model, api_key, temperature)
        return _call_openai(digest, model, api_key, temperature)
    except Exception:
        return None   # fallback silenzioso -> report/app usano il template


def _call_openai(digest, model, api_key, temperature):
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT_IT},
                  {"role": "user", "content": digest}],
        max_tokens=1200,
        temperature=temperature,
    )
    txt = resp.choices[0].message.content
    return txt.strip() if txt else None


def _call_anthropic(digest, model, api_key, temperature):
    import anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model if model.startswith("claude") else "claude-sonnet-4-5",
        max_tokens=1200,
        temperature=temperature,
        system=_SYSTEM_PROMPT_IT,
        messages=[{"role": "user", "content": digest}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
    return ("\n".join(parts)).strip() or None


if __name__ == "__main__":
    # test del digest (senza chiamare l'API): mostra cosa riceve l'LLM
    import pandas as pd
    df = pd.DataFrame({"Anno": np.arange(2026, 2032),
        "Revenues min": [0, 80, 180, 260, 300, 320], "Revenues piano": [0, 100, 220, 320, 380, 400],
        "Revenues max": [0, 130, 270, 390, 460, 500],
        "Cost var min": [0, -70, -150, -210, -240, -250], "Cost var piano": [0, -55, -120, -170, -200, -210],
        "Cost var max": [0, -40, -90, -130, -150, -160],
        "Costs fixed": [-20, -25, -30, -30, -30, -30], "Amort, & Depreciation": [0, -40, -40, -40, -40, -40],
        "Capex": [-200, -100, 0, 0, 0, 0],
        "Disposal & Capex Saving min": [0]*6, "Disposal & Capex Saving": [0, 0, 0, 0, 0, 20],
        "Disposal & Capex Saving max": [0, 0, 0, 0, 0, 35],
        "Change in working cap,": [0, -10, -15, -8, -5, 5], "Debt inflow": [0]*6,
        "Debt repayment": [0]*6, "Interest rate": [0.05]*6})
    cfg = core.SimConfig(n_sim=8000, seed=3, framework="FCFE", tv_method="gordon",
                         funding_mode="derived", gearing=0.70, funding_rate=0.05, repay_years=8,
                         capex_overrun_enable=True, copula="t", copula_df=6.0)
    res = core.run_simulations(df, cfg)
    print(build_results_digest(res, cfg, df))
    print("\n---\ngenerate_investment_commentary senza chiave API ->",
          generate_investment_commentary(res, cfg, df))
