import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import numpy_financial as npf
import hashlib
import json
from PIL import Image

from capex.visuals import (
    plot_npv_distribution, plot_boxplot, plot_cashflows,
    plot_cumulative_npv, plot_payback_distribution, plot_probs_kri,
    plot_car_kri, plot_irr_trends, plot_ppi_distribution
)

# -----------------------------
# FUNZIONE SIMULAZIONE MONTE CARLO CON SHIFT MULTISTEP
# -----------------------------
def run_simulations(df, n_sim, discount_rate, tax_rate, shift_probs):
    years = df.shape[0]
    years_col = df.iloc[:, 0].values

    # Estrazione colonne con fallback a 0
    rev_min = df.get('Revenues min', pd.Series([0]*years)).fillna(0).values
    rev_mode = df.get('Revenues piano', pd.Series([0]*years)).fillna(0).values
    rev_max = df.get('Revenues max', pd.Series([0]*years)).fillna(0).values
    cs_min = df.get('Cost var min', pd.Series([0]*years)).fillna(0).values
    cs_mode = df.get('Cost var piano', pd.Series([0]*years)).fillna(0).values
    cs_max = df.get('Cost var max', pd.Series([0]*years)).fillna(0).values
    costs_fixed = df.get('Costs fixed', pd.Series([0]*years)).fillna(0).values
    amort = df.get('Amort, & Depreciation', pd.Series([0]*years)).fillna(0).values
    capex = df.get('Capex', pd.Series([0]*years)).fillna(0).values
    disposal_min = df.get('Disposal & Capex Saving min', pd.Series([0]*years)).fillna(0).values
    disposal_mode = df.get('Disposal & Capex Saving', pd.Series([0]*years)).fillna(0).values
    disposal_max = df.get('Disposal & Capex Saving max', pd.Series([0]*years)).fillna(0).values
    change_wc = df.get('Change in working cap,', pd.Series([0]*years)).fillna(0).values

    # Matrici risultati
    fcf_matrix = np.zeros((n_sim, years))
    fcf_pv_matrix = np.zeros((n_sim, years))
    npv_cum_matrix = np.zeros((n_sim, years))
    npv_list = []

    for i in range(n_sim):
        # Flussi per simulazione
        revenue_flows = np.zeros(years)
        cs_flows = np.zeros(years)
        capex_flows = np.array(capex)  # Capex già noto
        disposal_flows = np.zeros(years)

        for y in range(years):
            # Ricavi
            if rev_min[y] == rev_mode[y] == rev_max[y] == 0:
                revenue = 0
            else:
                # Assicuriamoci left <= mode <= right
                l, m, r = sorted([rev_min[y], rev_mode[y], rev_max[y]])
                revenue = np.random.triangular(l, m, r)
            revenue_flows[y] = revenue

            # Costi variabili
            if cs_min[y] == cs_mode[y] == cs_max[y] == 0:
                cs = 0
            else:
                l, m, r = sorted([cs_min[y], cs_mode[y], cs_max[y]])
                cs = np.random.triangular(l, m, r)
            cs_flows[y] = cs

            # Disposal
            if disposal_min[y] == disposal_mode[y] == disposal_max[y] == 0:
                disp = 0
            else:
                l, m, r = sorted([disposal_min[y], disposal_mode[y], disposal_max[y]])
                disp = np.random.triangular(l, m, r)
            disposal_flows[y] = disp

        # ------------------ APPLICA SHIFT MULTISTEP ------------------
        def apply_shift(flow, probs):
            shifted = np.zeros_like(flow)
            for y in range(len(flow)):
                n_shift = np.random.choice([0,1,2], p=probs)
                target = min(y + n_shift, len(flow)-1)
                shifted[target] += flow[y]
            return shifted

        revenue_flows = apply_shift(revenue_flows, shift_probs)
        cs_flows = apply_shift(cs_flows, shift_probs)
        capex_flows = apply_shift(capex_flows, shift_probs)

        # FCF
        ebitda = revenue_flows + cs_flows + costs_fixed
        ebit = ebitda + amort
        taxes = -ebit * tax_rate
        fcf = ebitda + taxes + capex_flows + disposal_flows + change_wc

        # DCF
        fcf_pv = fcf / ((1 + discount_rate) ** (np.arange(1, years+1)))

        fcf_matrix[i,:] = fcf
        fcf_pv_matrix[i,:] = fcf_pv
        npv_list.append(np.sum(fcf_pv))
        npv_cum_matrix[i,:] = np.cumsum(fcf_pv)

    return (np.array(npv_list), fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col, costs_fixed, capex, revenue_matrix_orig, cs_matrix_orig, capex_matrix_orig, revenue_matrix_shifted, cs_matrix_shifted, capex_matrix_shifted)


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

        st.markdown("### Shift probabilistici multistep")
        shift_0 = st.slider("Probabilità rimanere stesso anno", 0.0, 1.0, 0.3)
        shift_1 = st.slider("Probabilità shift 1 anno", 0.0, 1.0, 0.5)
        shift_2 = st.slider("Probabilità shift 2 anni", 0.0, 1.0, 0.2)
        shift_probs = np.array([shift_0, shift_1, shift_2])
        shift_probs = shift_probs / shift_probs.sum()  # Normalizza

        run_button = st.button("Esegui simulazione")

    if uploaded_file is not None and run_button:
        if seed != 0:
            np.random.seed(int(seed))
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
        # ------------------------- RUN SIMULATION -------------------------
        results = run_simulations(df, n_sim, discount_rate, tax_rate, shift_probs)
        (
        npv_array, fcf_matrix, fcf_pv_matrix, npv_cum_matrix, 
        years_col, costs_fixed, capex,
        revenue_matrix_orig, cs_matrix_orig, capex_matrix_orig,
        revenue_matrix_shifted, cs_matrix_shifted, capex_matrix_shifted
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
                else:
                    irr_matrix[i, j] = 0

        # ------------------------- PPI -------------------------
        profit_index_array = []
        for i in range(fcf_matrix.shape[0]):
            fcf = fcf_pv_matrix[i, :]
            npv_cum = np.cumsum(fcf)
            cost_total = np.abs(costs_fixed) + np.abs(capex)
            cost_total = cost_total / ((1 + discount_rate) ** np.arange(1, n_years + 1))
            cost_total_cum = np.cumsum(cost_total)
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

        # ------------------------- GRAFICI -------------------------
        st.pyplot(plot_npv_distribution(npv_array, expected_npv, percentile_5, project_name))
        st.pyplot(plot_boxplot(npv_array, project_name))
        st.pyplot(plot_cashflows(fcf_matrix, fcf_matrix.shape[1], project_name))
        st.pyplot(plot_cumulative_npv(npv_cum_matrix, project_name))
        st.pyplot(plot_payback_distribution(payback_array, project_name))
        st.pyplot(plot_irr_trends(irr_p5, irr_p50, irr_p95, years_labels=df['Anno'].to_list(), title="Andamento IRR per anno", figsize=(10,6)))
        st.pyplot(plot_ppi_distribution(ppi_min, ppi_p5, ppi_p50, ppi_p95, ppi_max, years_labels=df['Anno'].to_list(), title="Andamento PPI per anno", figsize=(10,6)))

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




