import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import numpy_financial as npf

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

# -----------------------------
# FUNZIONE MONTE CARLO + SHIFT + DSCR
# -----------------------------
def run_simulations(
    df,
    n_sim,
    discount_rate,
    tax_rate,
    shift_probs,
    shift_rev_pct,
    shift_cs_pct,
    shift_capex_pct,
    enable_shift=True
):

    years = df.shape[0]
    years_col = df.iloc[:, 0].values

    # -----------------------------
    # INPUT BASE
    # -----------------------------
    rev_min = df.get('Revenues min', 0).fillna(0).values
    rev_mode = df.get('Revenues piano', 0).fillna(0).values
    rev_max = df.get('Revenues max', 0).fillna(0).values

    cs_min = df.get('Cost var min', 0).fillna(0).values
    cs_mode = df.get('Cost var piano', 0).fillna(0).values
    cs_max = df.get('Cost var max', 0).fillna(0).values

    costs_fixed = df.get('Costs fixed', 0).fillna(0).values
    amort = df.get('Amort, & Depreciation', 0).fillna(0).values
    capex = df.get('Capex', 0).fillna(0).values

    disposal_min = df.get('Disposal & Capex Saving min', 0).fillna(0).values
    disposal_mode = df.get('Disposal & Capex Saving', 0).fillna(0).values
    disposal_max = df.get('Disposal & Capex Saving max', 0).fillna(0).values

    change_wc = df.get('Change in working cap,', 0).fillna(0).values

    # -----------------------------
    # DEBT
    # -----------------------------
    debt_inflow = df.get('Debt inflow', pd.Series(0, index=df.index)).fillna(0).values
    debt_repayment = df.get('Debt repayment', pd.Series(0, index=df.index)).fillna(0).values
    interest_rate = df.get('Interest rate', pd.Series(0.05, index=df.index)).fillna(0.05).values

    # -----------------------------
    # OUTPUT
    # -----------------------------
    fcf_matrix = np.zeros((n_sim, years))
    fcf_pv_matrix = np.zeros((n_sim, years))
    npv_list = []

    dscr_matrix = np.zeros((n_sim, years))   # 🔥 NEW

    revenue_matrix_orig = np.zeros((n_sim, years))
    cs_matrix_orig = np.zeros((n_sim, years))
    capex_matrix_orig = np.zeros((n_sim, years))

    revenue_matrix_shifted = np.zeros((n_sim, years))
    cs_matrix_shifted = np.zeros((n_sim, years))
    capex_matrix_shifted = np.zeros((n_sim, years))

    # -----------------------------
    # SHIFT FUNCTION
    # -----------------------------
    def apply_shift(flow, probs, pct_shift):
        shifted = np.zeros_like(flow)

        for y in range(len(flow)):
            to_shift = flow[y] * pct_shift / 100
            remain = flow[y] - to_shift

            n_shift = np.random.choice([0, 1, 2], p=probs)
            target = min(y + n_shift, len(flow) - 1)

            shifted[target] += to_shift
            shifted[y] += remain

        return shifted

    # -----------------------------
    # MONTE CARLO
    # -----------------------------
    for i in range(n_sim):

        revenue_flows = np.zeros(years)
        cs_flows = np.zeros(years)
        capex_flows = capex.copy()
        disposal_flows = np.zeros(years)
        interest_flows = np.zeros(years)

        debt_stock = 0

        # -----------------------------
        # GENERAZIONE FLUSSI
        # -----------------------------
        for y in range(years):

            revenue_flows[y] = 0 if rev_min[y] == rev_mode[y] == rev_max[y] == 0 else np.random.triangular(
                *sorted([rev_min[y], rev_mode[y], rev_max[y]])
            )

            cs_flows[y] = 0 if cs_min[y] == cs_mode[y] == cs_max[y] == 0 else np.random.triangular(
                *sorted([cs_min[y], cs_mode[y], cs_max[y]])
            )

            disposal_flows[y] = 0 if disposal_min[y] == disposal_mode[y] == disposal_max[y] == 0 else np.random.triangular(
                *sorted([disposal_min[y], disposal_mode[y], disposal_max[y]])
            )

        # -----------------------------
        # SHIFT
        # -----------------------------
        if enable_shift:
            revenue_s = apply_shift(revenue_flows, shift_probs, shift_rev_pct)
            cs_s = apply_shift(cs_flows, shift_probs, shift_cs_pct)
            capex_s = apply_shift(capex_flows, shift_probs, shift_capex_pct)
        else:
            revenue_s = revenue_flows.copy()
            cs_s = cs_flows.copy()
            capex_s = capex_flows.copy()

        revenue_matrix_orig[i] = revenue_flows
        cs_matrix_orig[i] = cs_flows
        capex_matrix_orig[i] = capex_flows

        revenue_matrix_shifted[i] = revenue_s
        cs_matrix_shifted[i] = cs_s
        capex_matrix_shifted[i] = capex_s

        # -----------------------------
        # DEBT + FCF + DSCR
        # -----------------------------
        for y in range(years):

            prev_debt = debt_stock

            # interessi su stock iniziale periodo
            interest_flows[y] = -prev_debt * interest_rate[y]

            # aggiorna debito
            debt_stock = prev_debt + debt_inflow[y] - debt_repayment[y]
            debt_stock = max(debt_stock, 0)

            # -------------------------
            # EBITDA / EBIT / TAX
            # -------------------------
            ebitda = revenue_s[y] + cs_s[y] + costs_fixed[y]
            ebit = ebitda + amort[y]

            taxes = -np.maximum(ebit, 0) * tax_rate

            # -------------------------
            # FCF (correct order)
            # -------------------------
            fcf_y = (
                ebitda
                + taxes
                + interest_flows[y]
                - debt_repayment[y]
                + capex_s[y]
                + disposal_flows[y]
                + change_wc[y]
            )

            fcf_matrix[i, y] = fcf_y

            # -------------------------
            # DSCR
            # -------------------------
            debt_service = (-interest_flows[y] + debt_repayment[y])

            if debt_service > 0:
                dscr_matrix[i, y] = fcf_y / debt_service
            else:
                dscr_matrix[i, y] = np.nan

        # -----------------------------
        # DISCOUNTING
        # -----------------------------
        fcf_pv = fcf_matrix[i] / ((1 + discount_rate) ** np.arange(1, years + 1))
        fcf_pv_matrix[i] = fcf_pv
        npv_list.append(np.sum(fcf_pv))

    return (
        np.array(npv_list),
        fcf_matrix,
        fcf_pv_matrix,
        dscr_matrix,   # 🔥 NEW OUTPUT
        revenue_matrix_orig,
        cs_matrix_orig,
        capex_matrix_orig,
        revenue_matrix_shifted,
        cs_matrix_shifted,
        capex_matrix_shifted,
        years_col
    )

    
# -----------------------------
# CONFIGURAZIONE STREAMLIT
# -----------------------------
logo = Image.open("Image/logo_fibercop.PNG")
st.set_page_config(page_title="NPV @Risk Tool by ERM Fibercop", page_icon=logo , layout="wide")
st.image(logo, width=300)
st.markdown("""
<h1 style='color: white; font-weight: 800; font-family: Arial, sans-serif;'>NPV @Risk Simulation Tool by ERM</h1>
<p style='color: #cccccc; font-size: 18px; font-family: Arial, sans-serif;'>Simula scenari finanziari e analizza i progetti di investimento con DCF</p>
""", unsafe_allow_html=True)

# -----------------------------
# LOGIN
# -----------------------------
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

# -----------------------------
# PARAMETRI SIMULAZIONE + SHIFT
# -----------------------------
if st.session_state.logged_in:
    st.title("NPV @Risk Simulation Tool by ERM")

    uploaded_file = st.file_uploader("Carica file Excel", type=['xlsx','xls'])
    
    with st.sidebar:
        st.header("Parametri simulazione")
        project_name = st.text_input("Nome progetto", value="Progetto 1")
        discount_rate = st.number_input("Tasso di sconto (es. 0.10)", value=0.10, step=0.01, format="%.4f")
        tax_rate = st.number_input("Aliquota fiscale (es. 0.25)", value=0.25, step=0.01, format="%.4f")
        n_sim = st.number_input("Numero simulazioni", min_value=100, max_value=200000, value=2000, step=100)
        seed = st.number_input("Seed (0=random)", value=0)
        enable_shift = st.checkbox("Abilita shift temporale", value=True)
        
        st.markdown("### Shift probabilistici multistep")
        shift_0 = st.slider("Probabilità rimanere stesso anno", 0.0, 1.0, 0.3)
        shift_1 = st.slider("Probabilità shift 1 anno", 0.0, 1.0, 0.5)
        shift_2 = st.slider("Probabilità shift 2 anni", 0.0, 1.0, 0.2)
        shift_probs = np.array([shift_0, shift_1, shift_2])
        shift_probs = shift_probs / shift_probs.sum()  # Normalizza
        
        st.markdown("### Percentuale (%) flussi da shiftare")
        shift_rev_pct = st.slider("Ricavi (%)", 0, 100, 100)
        shift_cs_pct = st.slider("Costi variabili (%)", 0, 100, 100)
        shift_capex_pct = st.slider("Capex (%)", 0, 100, 100)


        run_button = st.button("Esegui simulazione")

    if uploaded_file is not None and run_button:
        if seed != 0:
            np.random.seed(int(seed))
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
        
        # --- baseline deterministic dal file ---
        costs_fixed_vec = df.get('Costs fixed', 0).fillna(0).to_numpy()
        capex_vec       = df.get('Capex', 0).fillna(0).to_numpy()

        # ------------------------- RUN SIMULATION -------------------------
        results = run_simulations(
            df=df,
            n_sim=n_sim,
            discount_rate=discount_rate,
            tax_rate=tax_rate,
            shift_probs=shift_probs,
            shift_rev_pct=shift_rev_pct,
            shift_cs_pct=shift_cs_pct,
            shift_capex_pct=shift_capex_pct,
            enable_shift=enable_shift)
        (
            npv_array,
            fcf_matrix,
            fcf_pv_matrix,
            dscr_matrix,
            revenue_matrix_orig,
            cs_matrix_orig,
            capex_matrix_orig,
            revenue_matrix_shifted,
            cs_matrix_shifted,
            capex_matrix_shifted,
            years_col
        ) = results
        n_sim_mean = min(1000, n_sim)  # numero di simulazioni da considerare per la media  
        revenue_mean_orig = revenue_matrix_orig.mean(axis=0)
        cs_mean_orig = cs_matrix_orig.mean(axis=0)
        capex_mean_orig = capex_matrix_orig.mean(axis=0)
        revenue_mean_shifted = revenue_matrix_shifted.mean(axis=0)
        cs_mean_shifted = cs_matrix_shifted.mean(axis=0)
        capex_mean_shifted = capex_matrix_shifted.mean(axis=0)
        
         # Grafico comparativo
        plt.figure(figsize=(12,6))
        plt.plot(years_col, revenue_mean_orig, marker='o', label="Ricavi originali")
        plt.plot(years_col, revenue_mean_shifted, marker='x', label="Ricavi shiftati")
        plt.plot(years_col, cs_mean_orig, marker='o', label="Costi variabili originali")
        plt.plot(years_col, cs_mean_shifted, marker='x', label="Costi variabili shiftati")
        plt.plot(years_col, capex_mean_orig, marker='o', label="Capex originali")
        plt.plot(years_col, capex_mean_shifted, marker='x', label="Capex shiftati")
        plt.xlabel("Anno")
        plt.ylabel("Valore flusso (media su simulazioni)")
        plt.title("Confronto flussi originali vs shiftati (media su tutte le simulazioni)")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        # ------------------------- METRICHE PRINCIPALI -------------------------
        expected_npv = np.mean(npv_array)
        percentile_5 = np.percentile(npv_array, 5)
        downside_prob = np.mean(npv_array<0)

        st.metric("Expected NPV", f"{expected_npv:,.2f}")
        st.metric("VaR 95% (CaR)", f"{percentile_5:,.2f}")
        st.metric("Probabilità NPV<0", f"{downside_prob*100:.2f}%")
        # ------------------------- DSCR KPI -------------------------
        dscr_mean = np.nanmean(dscr_matrix)
        dscr_p5 = np.nanpercentile(dscr_matrix, 5)
        dscr_default_prob = np.mean(dscr_matrix < 1)

        st.metric("DSCR medio", f"{dscr_mean:.2f}")
        st.metric("DSCR P5 (stress)", f"{dscr_p5:.2f}")
        st.metric("Probabilità DSCR < 1", f"{dscr_default_prob*100:.2f}%")

        def dscr_rating(x):
            if x < 1:
                return "🔴 Default Risk"
            elif x < 1.2:
                return "🟠 Weak"
            elif x < 1.5:
                return "🟡 Acceptable"
            else:
                return "🟢 Strong"

        rating = dscr_rating(dscr_mean)
        st.subheader("Credit Rating")
        st.write(f"Rating progetto: **{rating}**")

        # ------------------------- PAYBACK -------------------------
        payback_array = []
        N4_array = np.arange(fcf_matrix.shape[1]) + 1/6
        for i in range(fcf_matrix.shape[0]):
            npv_cum = np.cumsum(fcf_pv_matrix[i,:])
            pb = np.nan
            for j in range(len(npv_cum)):
                M19 = npv_cum[j-1] if j > 0 else 0
                N19 = npv_cum[j]
                N4 = N4_array[j]
                if N19 >= 0:
                    if j == 0:
                        pb = N19
                    else:
                        pb = -M19 / (N19 - M19) + N4 - 1
                    break
            payback_array.append(pb)
        payback_array = np.array(payback_array)

        # ------------------------- IRR -------------------------
        n_years = fcf_matrix.shape[1]
        irr_matrix = np.zeros((fcf_matrix.shape[0], n_years))
        for i in range(fcf_matrix.shape[0]):
            for j in range(n_years):
                fcf_subset = fcf_matrix[i, :j+1]
                if np.any(fcf_subset < 0) and np.any(fcf_subset > 0):
                    irr_matrix[i, j] = npf.irr(fcf_subset)
                    irr_matrix[i, j] = np.clip(irr_matrix[i, j], -1, 5)
                else:
                    irr_matrix[i, j] = 0

        # ------------------------- PPI -------------------------
        # ------------------------- PPI -------------------------
        # baseline deterministic dal file (stesso orizzonte di fcf_matrix)
        costs_fixed_vec = df.get('Costs fixed', 0).fillna(0).to_numpy()
        capex_vec       = df.get('Capex', 0).fillna(0).fillna(0).to_numpy()

        n_years = fcf_matrix.shape[1]
        disc = ((1 + discount_rate) ** np.arange(1, n_years + 1))

        # costo cumulato PV (deterministico) usato come denominatore
        cost_total = (np.abs(costs_fixed_vec[:n_years]) + np.abs(capex_vec[:n_years])) / disc
        cost_total_cum = np.cumsum(cost_total)

        profit_index_array = []
        for i in range(fcf_matrix.shape[0]):
            npv_cum = np.cumsum(fcf_pv_matrix[i, :])
            profit_index_array.append(npv_cum / cost_total_cum)
        profit_index_array = np.array(profit_index_array)
        # ------------------------- Percentili -------------------------
        ppi_min = np.nanmin(profit_index_array, axis=0)
        ppi_p5 = np.nanpercentile(profit_index_array, 5, axis=0)
        ppi_p50 = np.nanpercentile(profit_index_array, 50, axis=0)
        ppi_p95 = np.nanpercentile(profit_index_array, 95, axis=0)
        ppi_max = np.nanmax(profit_index_array, axis=0)

        irr_min = np.nanmin(irr_matrix, axis=0)
        irr_p5 = np.nanpercentile(irr_matrix, 5, axis=0)
        irr_p50 = np.nanpercentile(irr_matrix, 50, axis=0)
        irr_p95 = np.nanpercentile(irr_matrix, 95, axis=0)
        irr_max = np.nanmax(irr_matrix, axis=0)

        npv_cum_matrix = np.cumsum(fcf_pv_matrix, axis=1)
        # ------------------------- GRAFICI -------------------------
        st.pyplot(plot_npv_distribution(npv_array, expected_npv, percentile_5, project_name))
        st.pyplot(plot_boxplot(npv_array, project_name))
        st.pyplot(plot_cashflows(fcf_matrix, fcf_matrix.shape[1], project_name))
        st.pyplot(plot_cumulative_npv(npv_cum_matrix, project_name))
        st.pyplot(plot_payback_distribution(payback_array, project_name))
        st.pyplot(plot_irr_trends(irr_p5, irr_p50, irr_p95, years_labels=df['Anno'].to_list(), title="Andamento IRR per anno", figsize=(10,6)))
        st.pyplot(plot_ppi_distribution(ppi_min, ppi_p5, ppi_p50, ppi_p95, ppi_max, years_labels=df['Anno'].to_list(), title="Andamento PPI per anno", figsize=(10,6)))
        dscr_mean_curve = np.nanmean(dscr_matrix, axis=0)
        dscr_p5_curve = np.nanpercentile(dscr_matrix, 5, axis=0)
        plt.figure(figsize=(10,5))
        plt.plot(years_col, dscr_mean_curve, label="DSCR medio", marker='o')
        plt.plot(years_col, dscr_p5_curve, label="DSCR P5 (stress)", linestyle="--")
        plt.axhline(1, color="red", linestyle="--", label="Default (1.0)")
        plt.axhline(1.2, color="orange", linestyle="--", label="Bank threshold (1.2)")
        plt.axhline(1.5, color="green", linestyle="--", label="Strong (1.5)")
        plt.title("DSCR Profile")
        plt.xlabel("Anno")
        plt.ylabel("DSCR")
        plt.grid()
        plt.legend()
        st.pyplot(plt)

        # ------------------------- KRI -------------------------
        st.plotly_chart(plot_car_kri(percentile_5, expected_npv, project_name))
        st.plotly_chart(plot_probs_kri(downside_prob, project_name))

        # ------------------------- EXPORT EXCEL -------------------------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({'Simulazione': np.arange(1,len(npv_array)+1), 'NPV': npv_array}).to_excel(writer, index=False, sheet_name='NPV')
            df_fcf = pd.DataFrame(fcf_matrix, columns=years_col)
            df_fcf.insert(0, 'Simulazione', np.arange(1, fcf_matrix.shape[0]+1))
            df_fcf.to_excel(writer, index=False, sheet_name='FCF_simulati')
            df_dcf = pd.DataFrame(fcf_pv_matrix, columns=years_col)
            df_dcf.insert(0, 'Simulazione', np.arange(1, fcf_pv_matrix.shape[0]+1))
            df_dcf.to_excel(writer, index=False, sheet_name='DCF_simulati')
            median_fcf = np.median(fcf_matrix, axis=0)
            p5_fcf = np.percentile(fcf_matrix,5,axis=0)
            p95_fcf = np.percentile(fcf_matrix,95,axis=0)
            pd.DataFrame({'Anno': years_col, 'Median': median_fcf, 'P5': p5_fcf, 'P95': p95_fcf}).to_excel(writer, index=False, sheet_name='FCF_percentili')
            median_dcf = np.median(fcf_pv_matrix, axis=0)
            p5_dcf = np.percentile(fcf_pv_matrix,5,axis=0)
            p95_dcf = np.percentile(fcf_pv_matrix,95,axis=0)
            pd.DataFrame({'Anno': years_col, 'Median': median_dcf, 'P5': p5_dcf, 'P95': p95_dcf}).to_excel(writer, index=False, sheet_name='DCF_percentili')
            pd.DataFrame({'Simulazione': np.arange(1,len(payback_array)+1), 'PaybackYear': payback_array}).to_excel(writer, index=False, sheet_name='Payback_period')
            pd.DataFrame({'Anno': years_col, 'IRR_min': irr_min,'IRR_p5': irr_p5,'IRR_p50': irr_p50,'IRR_p95': irr_p95,'IRR_max': irr_max}).to_excel(writer, index=False, sheet_name='IRR_percentili')
            pd.DataFrame({'Anno': years_col, 'PPI_min': ppi_min,'PPI_p5': ppi_p5,'PPI_p50': ppi_p50,'PPI_p95': ppi_p95,'PPI_max': ppi_max}).to_excel(writer, index=False, sheet_name='PPI_percentili')

        st.download_button("Scarica Excel", data=output.getvalue(), file_name=f"{project_name}_sim.xlsx")

else:
    st.info("🔹 Completa il login per accedere alla web-app!")
