"""
dcf_app.py
==========
NPV @Risk Simulation Tool — interfaccia Streamlit.

La logica di calcolo vive in dcf_core.py (modulo puro e testato).
Questo file si occupa solo di: login, raccolta parametri, esecuzione, grafici, export.

Avvio:   streamlit run dcf_app.py
Requisiti: streamlit, pandas, numpy, scipy, numpy_financial, matplotlib, plotly, xlsxwriter, Pillow
"""

import json
import hashlib
import os
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import dcf_core as core

try:
    import damodaran_data
    _HAS_DAMODARAN = True
except Exception:
    _HAS_DAMODARAN = False

# I grafici restano i tuoi: nessuna modifica alle firme di capex.visuals
from capex.visuals import (
    plot_npv_distribution,
    plot_boxplot,
    plot_cashflows,
    plot_cumulative_npv,
    plot_payback_distribution,
    plot_probs_kri,
    plot_car_kri,
    plot_irr_trends,
    plot_ppi_distribution,
)


# ==========================================================================
# HELPER: selezione distribuzione per fattore
# ==========================================================================
DIST_LABELS = {
    "Triangolare": "triangular",
    "PERT": "pert",
    "Normale": "normal",
    "Lognormale": "lognormal",
    "Uniforme": "uniform",
    "Empirica (da storico)": "empirical",
}


def _dist_spec(factor_label, key):
    """Rende un menu di scelta della distribuzione + parametri, e ritorna lo spec dict."""
    choice = st.selectbox(f"Distribuzione — {factor_label}", list(DIST_LABELS.keys()),
                          index=0, key=f"dist_{key}")
    dist = DIST_LABELS[choice]
    spec = {"dist": dist}
    if dist == "pert":
        spec["lam"] = st.slider(f"λ PERT — {factor_label}", 1.0, 10.0, 4.0, 0.5,
                                key=f"lam_{key}",
                                help="Più alto = più peso sulla moda (stima da esperto).")
    if dist == "empirical":
        raw = st.text_area(f"Storico (valori separati da virgola) — {factor_label}",
                           value="", key=f"emp_{key}",
                           help="Applicato a tutti gli anni tramite quantile empirico.")
        try:
            vals = [float(x) for x in raw.replace(";", ",").split(",") if x.strip() != ""]
        except ValueError:
            vals = []
        spec["samples"] = vals
        if not vals:
            st.caption("⚠️ Nessun dato valido inserito: verrà usata la triangolare.")
    return spec


# ==========================================================================
# HEADER
# ==========================================================================
st.markdown("""
<h1 style='color: white; font-weight: 800; font-family: Arial, sans-serif;'>NPV @Risk Simulation Tool by ERM</h1>
<p style='color: #cccccc; font-size: 18px; font-family: Arial, sans-serif;'>Simula scenari finanziari e analizza i progetti di investimento con DCF</p>
""", unsafe_allow_html=True)


# ==========================================================================
# LOGIN
# ==========================================================================
st.sidebar.title("🔐 Login")
with open("users.json") as f:
    users = json.load(f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def check_login(username, password):
    return users.get(username) == hash_password(password)


if not st.session_state.logged_in:
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if check_login(username_input, password_input):
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.sidebar.success(f"Benvenuto {username_input}")
        else:
            st.sidebar.error("Username o password errati")
else:
    st.sidebar.success(f"Benvenuto {st.session_state.username}")


# ==========================================================================
# APP
# ==========================================================================
if st.session_state.logged_in:
    st.title("NPV @Risk Simulation Tool by ERM")

    uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx", "xls"])

    with st.sidebar:
        st.header("Parametri simulazione")
        project_name = st.text_input("Nome progetto", value="Progetto 1")
        project_description = st.text_area(
            "Descrizione del progetto (per il report)", value="", height=120,
            help="Testo libero: cos'è il progetto/investimento. Se attivi la sintesi AI, "
                 "viene usato per la sezione descrittiva (senza inventare oltre il testo). "
                 "Senza AI, viene inserito così com'è nel report.")

        # ---- FRAMEWORK: risolve l'ambiguita' del tasso di sconto ----
        framework = st.radio(
            "Framework di valutazione",
            ["FCFF (unlevered)", "FCFE (levered)"],
            index=1,
            help="FCFF: flusso operativo, si sconta al WACC. "
                 "FCFE: flusso all'equity (con interessi, quota capitale e nuovo debito), "
                 "si sconta al costo dell'equity (Ke).",
        )
        framework_code = "FCFF" if framework.startswith("FCFF") else "FCFE"
        rate_label = "WACC" if framework_code == "FCFF" else "Ke (costo equity)"

        tax_rate = st.number_input("Aliquota fiscale (es. 0.25)", value=0.25,
                                   step=0.01, format="%.4f")

        # ---- TASSO DI SCONTO: manuale oppure calcolato da costo del capitale ----
        coc = None
        rate_mode = st.radio(
            "Tasso di sconto",
            ["Inserisci manualmente", "Calcola da costo del capitale (CAPM / WACC)"],
            index=0,
        )
        if rate_mode == "Inserisci manualmente":
            discount_rate = st.number_input(f"{rate_label} (es. 0.10)", value=0.10,
                                            step=0.01, format="%.4f")
        else:
            with st.expander("💠 Costo del capitale (CAPM / WACC)", expanded=True):
                st.caption("Ke si calcola SEMPRE col CAPM. Il WACC lo usa come input: serve "
                           "solo in FCFF con debito (senza debito il WACC coincide col Ke).")

                dd_data = damodaran_data.load() if _HAS_DAMODARAN else None
                if dd_data is not None:
                    if dd_data.meta.get("is_snapshot"):
                        st.warning(
                            f"Dati Damodaran: snapshot incorporato ({dd_data.meta['date']}). "
                            "Gli ERP sono valori reali; i beta di settore sono INDICATIVI e "
                            "limitati. Esegui `python damodaran_update.py` per scaricare la "
                            "tabella settoriale ufficiale e completa."
                        )
                    else:
                        st.info(f"Dati Damodaran scaricati · {dd_data.meta['date']} · "
                                f"{dd_data.meta['source']}")

                listed = st.checkbox("Impresa quotata (beta di mercato disponibile)", value=False)

                # --- Risk-free: sempre manuale ---
                rf = st.number_input("Risk-free Rf (es. 0.03)", value=0.03,
                                     step=0.005, format="%.4f",
                                     help="Tasso di mercato maturo (es. Bund tedesco per l'area "
                                          "euro). L'ERP sotto include già il rischio-paese.")

                # --- ERP: auto da paese (se disponibile), sempre modificabile ---
                if dd_data is not None and dd_data.list_countries():
                    countries = dd_data.list_countries()
                    idx_c = countries.index("Italy") if "Italy" in countries else 0
                    country = st.selectbox("Paese (ERP totale, Damodaran)", countries, index=idx_c)
                    erp_val, erp_src, erp_date = dd_data.get_erp(country)
                    erp = st.number_input("Equity Risk Premium ERP (mercato + country)",
                                          value=float(erp_val) / 100.0 if erp_val else 0.06,
                                          step=0.005, format="%.4f")
                    st.caption(f"ERP auto da {erp_src} ({erp_date}); modificabile. Somma già il "
                               "rischio-paese: non aggiungerlo una seconda volta via Rf.")
                else:
                    erp = st.number_input("Equity Risk Premium ERP (mercato + country)",
                                          value=0.06, step=0.005, format="%.4f")

                de_target = st.number_input("D/E target (0 = nessun debito)", value=0.0,
                                            min_value=0.0, step=0.1, format="%.3f")
                kd = st.number_input("Costo del debito Kd (ante imposte)", value=0.05,
                                     step=0.005, format="%.4f")
                beta_debt = st.number_input("Beta del debito (opzionale)", value=0.0,
                                            step=0.05, format="%.3f")
                extra_prem = st.number_input("Premi aggiuntivi (size / specific)", value=0.0,
                                             step=0.005, format="%.4f")

                # --- Beta ---
                if listed:
                    beta_L_in = st.number_input("Beta levered osservato (mercato)",
                                                value=1.0, step=0.05, format="%.3f")
                    beta_U_in = None
                elif dd_data is not None and dd_data.list_regions():
                    regions = dd_data.list_regions()
                    idx_r = regions.index("Europe") if "Europe" in regions else 0
                    region = st.selectbox("Regione (Damodaran)", regions, index=idx_r)
                    sectors = dd_data.list_sectors(region)
                    sector = st.selectbox("Settore (Damodaran)", sectors) if sectors else None
                    cash_adj = st.checkbox("Beta corretto per la cassa", value=True)
                    if sector:
                        bU_val, bU_src, bU_date = dd_data.get_unlevered_beta(region, sector, cash_adj)
                    else:
                        bU_val, bU_src, bU_date = 0.75, "default", ""
                    beta_U_in = st.number_input("Beta unlevered di settore (auto, modificabile)",
                                                value=float(bU_val) if bU_val else 0.75,
                                                step=0.05, format="%.3f")
                    st.caption(f"β_U auto da {bU_src} ({bU_date}); poi rilevraggiato alla tua D/E.")
                    beta_L_in = None
                else:
                    st.caption("Beta unlevered di settore da Damodaran (colonna 'unlevered beta', "
                               "meglio corretta per la cassa). Fonte: pages.stern.nyu.edu/~adamodar.")
                    beta_U_in = st.number_input("Beta unlevered di settore (Damodaran)",
                                                value=0.75, step=0.05, format="%.3f")
                    beta_L_in = None

                coc = core.cost_of_capital(
                    framework=framework_code, rf=float(rf), erp=float(erp),
                    tax_rate=float(tax_rate), debt_equity=float(de_target),
                    listed=bool(listed), beta_levered=beta_L_in, beta_unlevered=beta_U_in,
                    kd_pretax=float(kd), beta_debt=float(beta_debt), extra_premium=float(extra_prem),
                )
                discount_rate = coc["discount_rate"]
                st.markdown(
                    f"**β levered** = {coc['beta_levered']:.3f}  ·  "
                    f"**Ke** = {coc['ke']:.2%}  ·  **WACC** = {coc['wacc']:.2%}\n\n"
                    f"→ Tasso applicato ({rate_label}): **{discount_rate:.2%}**"
                    + ("  _(= Ke, nessun debito)_" if not coc["has_debt"] else "")
                )
        n_sim = st.number_input("Numero simulazioni", min_value=100, max_value=200000,
                                value=2000, step=100)
        seed = st.number_input("Seed (0=random)", value=0)
        sampling_choice = st.radio("Campionamento", ["Latin Hypercube", "Monte Carlo casuale"],
                                   index=0, horizontal=True,
                                   help="Latin Hypercube stratifica le estrazioni: stesse "
                                        "simulazioni, stime meno rumorose (code più affidabili).")
        sampling = "lhs" if sampling_choice.startswith("Latin") else "random"

        # ---- CORRELAZIONE ----
        with st.expander("📈 Correlazione (Monte Carlo)", expanded=False):
            st.caption("Senza correlazione le code dell'NPV sono artificialmente sottili "
                       "(gli errori si 'diversificano'). Qui si impone una struttura realistica.")
            corr_rev_cost = st.slider("ρ Ricavi ↔ Costi variabili", -0.9, 0.9, 0.5, 0.05)
            corr_rev_disp = st.slider("ρ Ricavi ↔ Disposal", -0.9, 0.9, 0.0, 0.05)
            corr_cost_disp = st.slider("ρ Costi var ↔ Disposal", -0.9, 0.9, 0.0, 0.05)
            persistence = st.slider("Persistenza anno-su-anno (AR1)", 0.0, 0.95, 0.3, 0.05,
                                    help="0 = anni indipendenti; valori alti = trend persistenti.")
            st.markdown("**Struttura di dipendenza (copula)**")
            copula_choice = st.radio("Copula", ["Gaussiana", "t di Student (code spesse)"],
                                     index=0,
                                     help="La gaussiana ha dipendenza di coda nulla. La t "
                                          "cattura gli scenari in cui più fattori vanno male "
                                          "INSIEME (rischio di coda) — la lezione del 2008.")
            copula = "t" if copula_choice.startswith("t") else "gaussian"
            copula_df = st.slider("Gradi di libertà (t)", 3.0, 30.0, 8.0, 1.0,
                                  help="Più bassi = code più spesse. Alti ≈ gaussiana.") \
                if copula == "t" else 8.0

        # ---- DISTRIBUZIONI PER FATTORE ----
        with st.expander("🎲 Distribuzioni dei fattori di rischio", expanded=False):
            st.caption("Forma della distribuzione per ciascun fattore. PERT: stime da esperto "
                       "(meno peso agli estremi). Lognormale: coda destra (es. costi che possono "
                       "esplodere). Empirica: da dati storici. La correlazione resta invariata.")
            spec_rev = _dist_spec("Ricavi", "rev")
            spec_cost = _dist_spec("Costi variabili", "cost")
            spec_disp = _dist_spec("Disposal", "disp")

            st.markdown("**Costo dell'investimento (capex)**")
            capex_overrun_enable = st.checkbox(
                "Sovracosto capex stocastico", value=False,
                help="Moltiplicatore sul capex (1,20 = +20%). Il debito resta al piano base: "
                     "il sovracosto lo assorbe l'equity (FCFE più negativo).")
            if capex_overrun_enable:
                cx1, cx2, cx3 = st.columns(3)
                cx_min = cx1.number_input("min ×", value=0.95, step=0.05, format="%.2f", key="cxmn")
                cx_mode = cx2.number_input("piano ×", value=1.00, step=0.05, format="%.2f", key="cxmd")
                cx_max = cx3.number_input("max ×", value=1.30, step=0.05, format="%.2f", key="cxmx")
                spec_capex = _dist_spec("Sovracosto capex", "capex")
            else:
                cx_min, cx_mode, cx_max, spec_capex = 0.95, 1.0, 1.30, {"dist": "pert"}

        # ---- PIANO DI FINANZIAMENTO ----
        with st.expander("🏦 Piano di finanziamento (equity / senior debt)", expanded=False):
            st.caption("Se attivo, DERIVA il tiraggio di senior debt e l'iniezione di equity dal "
                       "cronoprogramma di capex, ignorando le colonne di debito dell'Excel. "
                       "Rilevante per l'FCFE. Il debito è dimensionato sul caso base; i sovracosti "
                       "di capex li assorbe l'equity.")
            funding_derived = st.checkbox("Deriva il finanziamento dal piano", value=False)
            gearing = st.slider("Gearing (quota senior debt sul fabbisogno)", 0.0, 1.0, 0.70, 0.05)
            draw_method_label = st.radio(
                "Metodo di tiraggio", ["Pari passu", "Equity prima", "Debito prima"], index=0,
                help="Pari passu: equity e debito proporzionali ogni anno. Equity prima: si "
                     "esaurisce l'equity, poi il debito (viceversa 'Debito prima').")
            idc_label = st.radio("Interessi in costruzione (IDC)",
                                 ["Capitalizzati sul debito", "Pagati per cassa", "Nessuno"], index=0)
            funding_rate = st.number_input("Tasso senior debt", value=0.05, step=0.005, format="%.4f")
            repay_years = st.slider("Tenor ammortamento (anni dopo l'entrata in esercizio)", 1, 30, 10, 1)
            repay_profile_label = st.radio("Profilo di rimborso", ["Lineare", "Annualità"], index=0)
            grace_years = st.slider("Preammortamento (anni, solo interessi)", 0, 5, 0, 1)
        funding_mode = "derived" if funding_derived else "manual"
        draw_method = {"Pari passu": "pari_passu", "Equity prima": "equity_first",
                       "Debito prima": "debt_first"}[draw_method_label]
        idc_mode = {"Capitalizzati sul debito": "capitalize", "Pagati per cassa": "cash",
                    "Nessuno": "none"}[idc_label]
        repay_profile = {"Lineare": "linear", "Annualità": "annuity"}[repay_profile_label]

        # ---- SHIFT ----
        enable_shift = st.checkbox("Abilita shift temporale", value=True)
        with st.expander("⏱️ Shift temporale (ritardo di progetto)", expanded=False):
            st.caption("Il ritardo è ora estratto UNA volta per progetto e applicato "
                       "insieme a ricavi, costi e capex (slittano coerentemente).")
            shift_0 = st.slider("P(stesso anno)", 0.0, 1.0, 0.3)
            shift_1 = st.slider("P(shift +1 anno)", 0.0, 1.0, 0.5)
            shift_2 = st.slider("P(shift +2 anni)", 0.0, 1.0, 0.2)
            shift_rev_pct = st.slider("% Ricavi da shiftare", 0, 100, 100)
            shift_cs_pct = st.slider("% Costi variabili da shiftare", 0, 100, 100)
            shift_capex_pct = st.slider("% Capex da shiftare", 0, 100, 100)

        # ---- VALORE TERMINALE ----
        with st.expander("🏁 Valore terminale", expanded=False):
            tv_choice = st.radio("Metodo", ["Nessuno", "Gordon (perpetuità)",
                                            "Multiplo di uscita"], index=0)
            tv_growth = st.number_input("g perpetuo (Gordon)", value=0.02, step=0.005,
                                        format="%.3f")
            tv_multiple = st.number_input("Multiplo su EBITDA (uscita)", value=6.0, step=0.5)
            st.caption("Gordon usa il flusso dell'ultimo anno come stato stazionario: "
                       "se lo shift è attivo, valuta di normalizzarlo o disattivare lo shift.")

        # ---- AFFIDABILITÀ ----
        with st.expander("📏 Affidabilità (intervalli di confidenza)", expanded=False):
            st.caption("Calcola l'errore standard e l'IC 95% delle metriche chiave con il "
                       "metodo della replicazione (più batch indipendenti). Aggiunge tempo "
                       "di calcolo ma dà una barra d'errore onesta ai numeri.")
            compute_ci = st.checkbox("Calcola intervalli di confidenza", value=False)
            n_batches = st.slider("Numero di batch", 5, 40, 20, 1,
                                  help="Più batch = errore standard più stabile. "
                                       "Simulazioni totali = batch × n° simulazioni.")

        # ---- SINTESI AI ----
        with st.expander("🧠 Sintesi AI (opzionale)", expanded=False):
            st.caption("La sintesi e la discussione dei risultati vengono scritte da un LLM, "
                       "a partire SOLO dai numeri calcolati (le tabelle e i grafici restano "
                       "deterministici). Serve la chiave API impostata nell'ambiente. "
                       "Se non disponibile, il report usa il testo standard.")
            ai_commentary = st.checkbox("Genera sintesi AI (report + a schermo)", value=False)
            llm_provider = st.radio("Provider", ["OpenAI", "Anthropic (Claude)"], index=0,
                                    horizontal=True)
            if llm_provider == "OpenAI":
                llm_model = st.selectbox("Modello", ["gpt-4o", "gpt-4o-mini", "gpt-4"], index=0)
            else:
                llm_model = st.selectbox("Modello", ["claude-sonnet-4-5", "claude-opus-4-1"], index=0)
        llm_provider_code = "anthropic" if llm_provider.startswith("Anthropic") else "openai"

        run_button = st.button("Esegui simulazione")

    tv_method = {"Nessuno": "none", "Gordon (perpetuità)": "gordon",
                 "Multiplo di uscita": "multiple"}[tv_choice]

    if uploaded_file is not None and run_button:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

        # ---- validazione input: avvisa in modo RUMOROSO invece di leggere 0 in silenzio ----
        ok, missing = core.validate_columns(df)
        if missing:
            st.warning("⚠️ Colonne attese non trovate (verrebbero lette come 0). "
                       "Verifica le intestazioni del file:\n\n- " + "\n- ".join(missing))

        cfg = core.SimConfig(
            framework=framework_code,
            discount_rate=float(discount_rate),
            tax_rate=float(tax_rate),
            n_sim=int(n_sim),
            seed=int(seed),
            corr_rev_cost=float(corr_rev_cost),
            corr_rev_disp=float(corr_rev_disp),
            corr_cost_disp=float(corr_cost_disp),
            persistence=float(persistence),
            enable_shift=bool(enable_shift),
            shift_probs=(shift_0, shift_1, shift_2),
            shift_rev_pct=float(shift_rev_pct),
            shift_cs_pct=float(shift_cs_pct),
            shift_capex_pct=float(shift_capex_pct),
            tv_method=tv_method,
            tv_growth=float(tv_growth),
            tv_multiple=float(tv_multiple),
            dist_revenue=spec_rev,
            dist_cost=spec_cost,
            dist_disposal=spec_disp,
            copula=copula,
            copula_df=float(copula_df),
            sampling=sampling,
            capex_overrun_enable=bool(capex_overrun_enable),
            capex_overrun={"min": float(cx_min), "mode": float(cx_mode), "max": float(cx_max)},
            dist_capex=spec_capex,
            funding_mode=funding_mode,
            gearing=float(gearing),
            draw_method=draw_method,
            idc_mode=idc_mode,
            funding_rate=float(funding_rate),
            repay_years=int(repay_years),
            repay_profile=repay_profile,
            grace_years=int(grace_years),
        )

        with st.spinner("Simulazione in corso..."):
            res = core.run_simulations(df, cfg)

        if res["tv_warning"]:
            st.warning("⚠️ " + res["tv_warning"])

        years_col = res["years_col"]
        npv_array = res["npv"]
        fcf_matrix = res["fcf"]
        fcf_pv_matrix = res["fcf_pv"]
        dscr_matrix = res["dscr"]

        cap = (f"Framework: **{res['framework']}** · sconto al "
               f"**{rate_label} = {discount_rate:.2%}** · "
               f"valore terminale: **{tv_choice}**")
        if coc is not None:
            cap += (f"\n\nCosto del capitale — β levered {coc['beta_levered']:.3f}, "
                    f"Ke {coc['ke']:.2%}, WACC {coc['wacc']:.2%}"
                    + (" (= Ke, nessun debito)" if not coc["has_debt"] else "") + ".")
        st.caption(cap)

        # ------------------------- GRAFICO ORIGINALE vs SHIFT -------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years_col, res["revenue_orig_mean"], marker="o", label="Ricavi originali")
        ax.plot(years_col, res["revenue_shift_mean"], marker="x", label="Ricavi shiftati")
        ax.plot(years_col, res["cs_orig_mean"], marker="o", label="Costi variabili originali")
        ax.plot(years_col, res["cs_shift_mean"], marker="x", label="Costi variabili shiftati")
        ax.plot(years_col, res["capex_orig_mean"], marker="o", label="Capex originali")
        ax.plot(years_col, res["capex_shift_mean"], marker="x", label="Capex shiftati")
        ax.set_xlabel("Anno")
        ax.set_ylabel("Valore flusso (media su simulazioni)")
        ax.set_title("Confronto flussi originali vs shiftati (media su tutte le simulazioni)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # ------------------------- METRICHE PRINCIPALI -------------------------
        expected_npv = float(np.mean(npv_array))
        percentile_5 = float(np.percentile(npv_array, 5))
        downside_prob = float(np.mean(npv_array < 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected NPV", f"{expected_npv:,.2f}")
        c2.metric("VaR 95% (CaR)", f"{percentile_5:,.2f}")
        c3.metric("Probabilità NPV<0", f"{downside_prob*100:.2f}%")

        # ------------------------- PIANO DI FINANZIAMENTO -------------------------
        fund = res.get("funding")
        if fund is not None:
            st.subheader("🏦 Piano di finanziamento (equity / senior debt)")
            years_lbl = [str(int(y)) for y in res["years_col"]]
            eq = res["equity_injection"]
            dd = fund["debt_inflow"]
            rp = fund["debt_repayment"]
            plan_df = pd.DataFrame({
                "Anno": years_lbl,
                "Drawdown senior debt": np.round(dd, 1),
                "Iniezione equity": np.round(eq, 1),
                "Rimborso debito": np.round(rp, 1),
            }).set_index("Anno")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Fabbisogno costruzione", f"{fund['total_need']:,.0f}")
            fc2.metric("Senior debt (su capex)", f"{fund['debt_capex']:,.0f}",
                       f"gearing {fund['gearing_capex']:.0%}")
            fc3.metric("Equity", f"{fund['total_equity']:,.0f}")
            fc4.metric("Debito a COD", f"{fund['debt_at_cod']:,.0f}",
                       f"IDC {fund['idc_total']:,.1f}")
            figf, axf = plt.subplots(figsize=(10, 3))
            x = np.arange(len(years_lbl))
            axf.bar(x - 0.2, dd, width=0.4, label="Senior debt (drawdown)", color="#4C78A8")
            axf.bar(x + 0.2, eq, width=0.4, label="Equity", color="#2E7D32")
            axf.plot(x, rp, "o-", color="#C62828", label="Rimborso debito", linewidth=1)
            axf.set_xticks(x); axf.set_xticklabels(years_lbl)
            axf.set_ylabel("Fonti / rimborsi"); axf.legend(fontsize=8); axf.grid(True, alpha=0.25)
            axf.set_title("Tiraggio delle fonti per anno di realizzazione")
            st.pyplot(figf)
            st.table(plan_df)
            st.caption("Il senior debt è dimensionato sul caso base; eventuali sovracosti di "
                       "capex sono a carico dell'equity. Con questo piano attivo, le colonne "
                       "'Debt inflow'/'Debt repayment' dell'Excel vengono ignorate.")

        # ------------------------- DSCR -------------------------
        st.subheader("Analisi del credito (DSCR)")
        if res["has_debt"]:
            dscr_mean = float(np.nanmean(dscr_matrix))
            dscr_p5 = float(np.nanpercentile(dscr_matrix, 5))
            _dscr_defined = dscr_matrix[np.isfinite(dscr_matrix)]
            dscr_default_prob = float(np.mean(_dscr_defined < 1)) if _dscr_defined.size else 0.0

            d1, d2, d3 = st.columns(3)
            d1.metric("DSCR medio", f"{dscr_mean:.2f}")
            d2.metric("DSCR P5 (stress)", f"{dscr_p5:.2f}")
            d3.metric("Probabilità DSCR < 1", f"{dscr_default_prob*100:.2f}%")

            st.write(f"Rating progetto: **{core.dscr_rating(dscr_mean)}**")
            st.caption("DSCR = CFADS / servizio del debito (cassa PRIMA del servizio del debito). "
                       "CFADS = EBITDA − imposte − capex ± circolante.")
        else:
            st.info("Nessun debito nei dati (colonne Debt inflow/repayment assenti o nulle): "
                    "DSCR non applicabile.")

        # ------------------------- PAYBACK / IRR / PPI -------------------------
        payback_array = core.compute_payback(fcf_pv_matrix)
        irr_matrix = core.compute_irr_curve(fcf_matrix, subsample=cfg.irr_subsample)
        profit_index_array = core.compute_ppi(fcf_pv_matrix, df, cfg.discount_rate)

        irr_min = np.nanmin(irr_matrix, axis=0)
        irr_p5 = np.nanpercentile(irr_matrix, 5, axis=0)
        irr_p50 = np.nanpercentile(irr_matrix, 50, axis=0)
        irr_p95 = np.nanpercentile(irr_matrix, 95, axis=0)
        irr_max = np.nanmax(irr_matrix, axis=0)

        ppi_min = np.nanmin(profit_index_array, axis=0)
        ppi_p5 = np.nanpercentile(profit_index_array, 5, axis=0)
        ppi_p50 = np.nanpercentile(profit_index_array, 50, axis=0)
        ppi_p95 = np.nanpercentile(profit_index_array, 95, axis=0)
        ppi_max = np.nanmax(profit_index_array, axis=0)

        npv_cum_matrix = np.cumsum(fcf_pv_matrix, axis=1)

        # ------------------------- GRAFICI (invariati) -------------------------
        years_labels = list(years_col)
        st.pyplot(plot_npv_distribution(npv_array, expected_npv, percentile_5, project_name))
        st.pyplot(plot_boxplot(npv_array, project_name))
        st.pyplot(plot_cashflows(fcf_matrix, fcf_matrix.shape[1], project_name))
        st.pyplot(plot_cumulative_npv(npv_cum_matrix, project_name))
        st.pyplot(plot_payback_distribution(payback_array, project_name))
        st.pyplot(plot_irr_trends(irr_p5, irr_p50, irr_p95, years_labels=years_labels,
                                  title="Andamento IRR per anno", figsize=(10, 6)))
        st.pyplot(plot_ppi_distribution(ppi_min, ppi_p5, ppi_p50, ppi_p95, ppi_max,
                                        years_labels=years_labels,
                                        title="Andamento PPI per anno", figsize=(10, 6)))

        # ---- profilo DSCR ----
        if res["has_debt"]:
            dscr_mean_curve = np.nanmean(dscr_matrix, axis=0)
            dscr_p5_curve = np.nanpercentile(dscr_matrix, 5, axis=0)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(years_col, dscr_mean_curve, label="DSCR medio", marker="o")
            ax2.plot(years_col, dscr_p5_curve, label="DSCR P5 (stress)", linestyle="--")
            ax2.axhline(1, color="red", linestyle="--", label="Default (1.0)")
            ax2.axhline(1.2, color="orange", linestyle="--", label="Soglia banca (1.2)")
            ax2.axhline(1.5, color="green", linestyle="--", label="Strong (1.5)")
            ax2.set_title("DSCR Profile")
            ax2.set_xlabel("Anno")
            ax2.set_ylabel("DSCR")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

        # ------------------------- KRI -------------------------
        st.plotly_chart(plot_car_kri(percentile_5, expected_npv, project_name))
        st.plotly_chart(plot_probs_kri(downside_prob, project_name))

        # ============================================================
        # ANALISI DI SENSIBILITÀ — cosa muove l'NPV
        # ============================================================
        st.header("🔍 Analisi di sensibilità")

        # ---- 1) Importanza dei driver (dal Monte Carlo, correlazione di rango) ----
        st.subheader("Driver dell'NPV (correlazione di rango sulle simulazioni)")
        importance = core.driver_importance(res)
        if importance:
            labels = [d["driver"] for d in importance][::-1]
            rhos = [d["spearman"] for d in importance][::-1]
            contribs = [d["contribution"] for d in importance][::-1]
            colors = ["#2ca02c" if r >= 0 else "#d62728" for r in rhos]
            figd, axd = plt.subplots(figsize=(10, max(3, 0.6 * len(labels) + 1)))
            axd.barh(labels, rhos, color=colors)
            axd.axvline(0, color="black", linewidth=0.8)
            axd.set_xlabel("Correlazione di rango (Spearman) con l'NPV")
            axd.set_xlim(-1, 1)
            for i, (r, c) in enumerate(zip(rhos, contribs)):
                axd.text(r + (0.02 if r >= 0 else -0.02), i, f"{c:.0%}",
                         va="center", ha="left" if r >= 0 else "right", fontsize=9)
            axd.set_title("Driver per importanza — segno = direzione, etichetta = contributo alla varianza")
            axd.grid(True, axis="x", alpha=0.3)
            st.pyplot(figd)
            st.caption("Verde = spinge l'NPV verso l'alto, rosso = verso il basso. "
                       "Il contributo (rho² normalizzato) somma ~100% ed è robusto con input correlati.")

        # ---- 2) Tornado deterministico ----
        st.subheader("Tornado (impatto sull'NPV, un fattore alla volta)")
        cpl1, cpl2 = st.columns(2)
        p_low = cpl1.slider("Percentile 'basso'", 0.01, 0.25, 0.10, 0.01)
        p_high = cpl2.slider("Percentile 'alto'", 0.75, 0.99, 0.90, 0.01)
        tor = core.tornado_oneway(df, cfg, p_low=p_low, p_high=p_high,
                                  include_assumptions=True)
        bars = tor["bars"]
        base_npv = tor["base"]
        if bars:
            labels = [b["label"] for b in bars][::-1]
            lows = [b["low"] for b in bars][::-1]
            highs = [b["high"] for b in bars][::-1]
            figt, axt = plt.subplots(figsize=(10, max(3, 0.6 * len(labels) + 1)))
            for i, (lo, hi) in enumerate(zip(lows, highs)):
                left, right = min(lo, hi), max(lo, hi)
                axt.barh(i, right - left, left=left, color="#4c78a8", alpha=0.85)
            axt.axvline(base_npv, color="red", linestyle="--", linewidth=1,
                        label=f"NPV base = {base_npv:,.0f}")
            axt.set_yticks(range(len(labels)))
            axt.set_yticklabels(labels)
            axt.set_xlabel("NPV")
            axt.set_title(f"Tornado — swing dell'NPV tra P{p_low*100:.0f} e P{p_high*100:.0f} "
                          f"(ipotesi scalari incluse)")
            axt.legend()
            axt.grid(True, axis="x", alpha=0.3)
            st.pyplot(figt)
            st.caption("Ogni barra mostra l'NPV quando quel singolo fattore va dal valore basso "
                       "all'alto, tenendo gli altri al 'piano'. Ordinate per ampiezza dell'impatto.")

        # ---- 3) Sensibilità a due vie ----
        st.subheader("Sensibilità a due vie")
        param_labels = {
            "Tasso di sconto": "discount_rate",
            "Aliquota": "tax_rate",
            "g terminale (Gordon)": "tv_growth",
            "Multiplo EBITDA": "tv_multiple",
            "Scala ricavi (×)": "scale_revenue",
            "Scala costi (×)": "scale_cost",
            "Scala capex (×)": "scale_capex",
        }
        cc1, cc2 = st.columns(2)
        x_lbl = cc1.selectbox("Variabile X", list(param_labels.keys()), index=0)
        y_lbl = cc2.selectbox("Variabile Y", list(param_labels.keys()), index=4)

        def _default_range(param):
            if param == "discount_rate":
                return [round(discount_rate + d, 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
            if param == "tax_rate":
                return [round(max(tax_rate + d, 0.0), 3) for d in (-0.05, 0.0, 0.05)]
            if param == "tv_growth":
                return [0.0, 0.01, 0.02, 0.03]
            if param == "tv_multiple":
                return [4.0, 6.0, 8.0, 10.0]
            return [0.8, 0.9, 1.0, 1.1, 1.2]   # scale_*

        x_param, y_param = param_labels[x_lbl], param_labels[y_lbl]
        if x_param == y_param:
            st.info("Scegli due variabili diverse per la griglia a due vie.")
        else:
            x_vals = _default_range(x_param)
            y_vals = _default_range(y_param)
            grid = core.two_way_sensitivity(df, cfg, x_param, x_vals, y_param, y_vals)
            figh, axh = plt.subplots(figsize=(1.4 * len(x_vals) + 3, 1.0 * len(y_vals) + 2))
            im = axh.imshow(grid, cmap="RdYlGn", aspect="auto", origin="lower")
            axh.set_xticks(range(len(x_vals)))
            axh.set_xticklabels([f"{v:g}" for v in x_vals])
            axh.set_yticks(range(len(y_vals)))
            axh.set_yticklabels([f"{v:g}" for v in y_vals])
            axh.set_xlabel(x_lbl)
            axh.set_ylabel(y_lbl)
            for iy in range(len(y_vals)):
                for ix in range(len(x_vals)):
                    axh.text(ix, iy, f"{grid[iy, ix]:,.0f}", ha="center", va="center",
                             fontsize=8, color="black")
            figh.colorbar(im, ax=axh, label="NPV")
            axh.set_title(f"NPV al variare di {x_lbl} × {y_lbl}")
            st.pyplot(figh)
            st.caption("NPV deterministico (fattori al 'piano') sulla griglia delle due ipotesi.")

        # ============================================================
        # AFFIDABILITÀ DELLE STIME (intervalli di confidenza)
        # ============================================================
        reliability = None
        if compute_ci:
            st.header("📏 Affidabilità delle stime")
            with st.spinner(f"Calcolo intervalli di confidenza su {int(n_batches)} batch..."):
                reliability = core.estimate_with_ci(df, cfg, n_batches=int(n_batches))
            rel_rows = []
            for k, m in reliability["metrics"].items():
                if k == "P(NPV<0)":
                    rel_rows.append({"Metrica": k, "Stima": f"{m['value']:.1%}",
                                     "± (IC 95%)": f"±{m['half_width']*100:.1f} pp",
                                     "Intervallo 95%": f"{m['ci_low']:.1%} … {m['ci_high']:.1%}"})
                else:
                    rel_rows.append({"Metrica": k, "Stima": f"{m['value']:,.1f}",
                                     "± (IC 95%)": f"±{m['half_width']:,.1f}",
                                     "Intervallo 95%": f"{m['ci_low']:,.1f} … {m['ci_high']:,.1f}"})
            st.table(pd.DataFrame(rel_rows).set_index("Metrica"))
            st.caption(
                f"Metodo della replicazione: {reliability['n_batches']} batch indipendenti, "
                f"{reliability['total_n']:,} simulazioni totali, campionamento "
                f"{'Latin Hypercube' if reliability['sampling']=='lhs' else 'Monte Carlo casuale'}. "
                "L'errore standard è la deviazione tra i batch — valido anche con LHS, dove "
                "std/√n non si applica. Con LHS la barra d'errore si stringe."
            )

        # ============================================================
        # SINTESI AI (a schermo) + REPORT WORD
        # ============================================================
        def _llm_key(provider):
            """Chiave API: prima st.secrets (Streamlit Cloud), poi variabile d'ambiente."""
            name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            try:
                if name in st.secrets:
                    return st.secrets[name]
            except Exception:
                pass
            return os.environ.get(name)

        ai_text = None
        llm_key = _llm_key(llm_provider_code) if ai_commentary else None
        if ai_commentary:
            st.header("🧠 Sintesi e discussione (AI)")
            if not llm_key:
                st.info(f"Chiave API non trovata. Imposta "
                        f"{'ANTHROPIC_API_KEY' if llm_provider_code == 'anthropic' else 'OPENAI_API_KEY'} "
                        "nei secrets dell'app (Streamlit Cloud → Manage app → Settings → Secrets). "
                        "Il report userà il testo standard.")
            else:
                with st.spinner("Generazione della sintesi AI..."):
                    try:
                        import llm_commentary
                        ai_text = llm_commentary.generate_investment_commentary(
                            res, cfg, df, reliability=reliability, model=llm_model,
                            provider=llm_provider_code, project_description=project_description,
                            api_key=llm_key)
                    except Exception:
                        ai_text = None
                if ai_text:
                    st.markdown(ai_text)
                    st.caption("Scritta da un LLM sui numeri del modello; le tabelle e i grafici "
                               "restano deterministici. Supporto alla decisione, non consulenza.")
                else:
                    st.info("Sintesi AI non disponibile (errore dell'API o del modello). "
                            "Il report userà il testo standard.")

        st.header("📄 Report per il comitato")
        st.caption("Genera un documento Word con sintesi (AI se attiva), risultati (con IC se "
                   "calcolati), distribuzione dell'NPV, driver, tornado, DSCR, piano di "
                   "finanziamento, ipotesi e nota metodologica.")
        try:
            import dcf_report
            doc_buf = BytesIO()
            dcf_report.build_report(df, cfg, res, doc_buf, project_name=project_name,
                                    reliability=reliability, ai_commentary=bool(ai_commentary),
                                    llm_model=llm_model, llm_provider=llm_provider_code,
                                    project_description=project_description,
                                    commentary_text=ai_text, api_key=llm_key)
            st.download_button(
                "⬇️ Scarica report Word",
                data=doc_buf.getvalue(),
                file_name=f"report_{project_name.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            if reliability is None:
                st.caption("Suggerimento: attiva gli intervalli di confidenza per includerli "
                           "anche nel report.")
        except Exception as e:
            st.warning(f"Report Word non disponibile: {e}")

        # ------------------------- EXPORT EXCEL -------------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            pd.DataFrame({"Simulazione": np.arange(1, len(npv_array) + 1),
                          "NPV": npv_array}).to_excel(writer, index=False, sheet_name="NPV")

            df_fcf = pd.DataFrame(fcf_matrix, columns=years_col)
            df_fcf.insert(0, "Simulazione", np.arange(1, fcf_matrix.shape[0] + 1))
            df_fcf.to_excel(writer, index=False, sheet_name="FCF_simulati")

            df_dcf = pd.DataFrame(fcf_pv_matrix, columns=years_col)
            df_dcf.insert(0, "Simulazione", np.arange(1, fcf_pv_matrix.shape[0] + 1))
            df_dcf.to_excel(writer, index=False, sheet_name="DCF_simulati")

            pd.DataFrame({"Anno": years_col,
                          "Median": np.median(fcf_matrix, axis=0),
                          "P5": np.percentile(fcf_matrix, 5, axis=0),
                          "P95": np.percentile(fcf_matrix, 95, axis=0)}
                         ).to_excel(writer, index=False, sheet_name="FCF_percentili")

            pd.DataFrame({"Anno": years_col,
                          "Median": np.median(fcf_pv_matrix, axis=0),
                          "P5": np.percentile(fcf_pv_matrix, 5, axis=0),
                          "P95": np.percentile(fcf_pv_matrix, 95, axis=0)}
                         ).to_excel(writer, index=False, sheet_name="DCF_percentili")

            pd.DataFrame({"Simulazione": np.arange(1, len(payback_array) + 1),
                          "PaybackYear": payback_array}
                         ).to_excel(writer, index=False, sheet_name="Payback_period")

            pd.DataFrame({"Anno": years_col, "IRR_min": irr_min, "IRR_p5": irr_p5,
                          "IRR_p50": irr_p50, "IRR_p95": irr_p95, "IRR_max": irr_max}
                         ).to_excel(writer, index=False, sheet_name="IRR_percentili")

            pd.DataFrame({"Anno": years_col, "PPI_min": ppi_min, "PPI_p5": ppi_p5,
                          "PPI_p50": ppi_p50, "PPI_p95": ppi_p95, "PPI_max": ppi_max}
                         ).to_excel(writer, index=False, sheet_name="PPI_percentili")

            if res["has_debt"]:
                pd.DataFrame({"Anno": years_col,
                              "DSCR_medio": np.nanmean(dscr_matrix, axis=0),
                              "DSCR_p5": np.nanpercentile(dscr_matrix, 5, axis=0),
                              "DSCR_p50": np.nanpercentile(dscr_matrix, 50, axis=0),
                              "DSCR_p95": np.nanpercentile(dscr_matrix, 95, axis=0)}
                             ).to_excel(writer, index=False, sheet_name="DSCR_percentili")

            pd.DataFrame({"Metrica": ["Framework", "Tasso di sconto", "Aliquota",
                                      "N. simulazioni", "Expected NPV", "VaR95 (CaR)",
                                      "P(NPV<0)", "Valore terminale (metodo)"],
                          "Valore": [res["framework"], f"{discount_rate:.4f}",
                                     f"{tax_rate:.4f}", int(n_sim),
                                     f"{expected_npv:,.2f}", f"{percentile_5:,.2f}",
                                     f"{downside_prob*100:.2f}%", tv_choice]}
                         ).to_excel(writer, index=False, sheet_name="Sintesi")

        st.download_button("Scarica Excel", data=output.getvalue(),
                           file_name=f"{project_name}_sim.xlsx")

else:
    st.info("🔹 Completa il login per accedere alla web-app!")
