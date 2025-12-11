import hashlib
import json
from io import BytesIO
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf

from capex.visuals import (
    plot_npv_distribution, plot_boxplot, plot_cashflows, plot_cumulative_npv,
    plot_payback_distribution, plot_probs_kri, plot_car_kri,
    plot_irr_trends, plot_ppi_distribution
)

# -----------------------------
# FUNZIONE DI SIMULAZIONE CON SHIFT
# -----------------------------
def run_simulations(df, n_sim, discount_rate, tax_rate,
                    shift_rev={0:1.0}, shift_cs={0:1.0}, shift_capex={0:1.0}, shift_disp={0:1.0}):

    years = df.shape[0]
    years_col = df.iloc[:, 0].values

    # Colonne con fallback a 0
    rev_min = df.get('Revenues min', pd.Series([0]*years)).values
    rev_mode = df.get('Revenues piano', pd.Series([0]*years)).values
    rev_max = df.get('Revenues max', pd.Series([0]*years)).values
    cs_min = df.get('Cost var min', pd.Series([0]*years)).values
    cs_mode = df.get('Cost var piano', pd.Series([0]*years)).values
    cs_max = df.get('Cost var max', pd.Series([0]*years)).values
    costs_fixed = df.get('Costs fixed', pd.Series([0]*years)).values
    amort = df.get('Amort, & Depreciation', pd.Series([0]*years)).values
    capex = df.get('Capex', pd.Series([0]*years)).values
    disposal_min = df.get('Disposal & Capex Saving min', pd.Series([0]*years)).values
    disposal_mode = df.get('Disposal & Capex Saving', pd.Series([0]*years)).values
    disposal_max = df.get('Disposal & Capex Saving max', pd.Series([0]*years)).values
    change_wc = df.get('Change in working cap,', pd.Series([0]*years)).values

    # Matrici risultati
    fcf_matrix = np.zeros((n_sim, years))
    fcf_pv_matrix = np.zeros((n_sim, years))
    npv_cum_matrix = np.zeros((n_sim, years))
    npv_list = []

    # Prepara shift
    rev_vals, rev_probs = list(shift_rev.keys()), list(shift_rev.values())
    cs_vals, cs_probs = list(shift_cs.keys()), list(shift_cs.values())
    capex_vals, capex_probs = list(shift_capex.keys()), list(shift_capex.values())
    disp_vals, disp_probs = list(shift_disp.keys()), list(shift_disp.values())

    # Monte Carlo
    for i in range(n_sim):
        for y in range(years):
            # ------ SHIFT TEMPORALI ------
            rev_shift = np.random.choice(rev_vals, p=rev_probs)
            cs_shift = np.random.choice(cs_vals, p=cs_probs)
            capex_shift = np.random.choice(capex_vals, p=capex_probs)
            disp_shift = np.random.choice(disp_vals, p=disp_probs)

            idx_rev = np.clip(y - rev_shift, 0, years-1)
            idx_cs = np.clip(y - cs_shift, 0, years-1)
            idx_capex = np.clip(y - capex_shift, 0, years-1)
            idx_disp = np.clip(y - disp_shift, 0, years-1)

            # Ricavi
            if rev_min[idx_rev] == rev_mode[idx_rev] == rev_max[idx_rev] == 0:
                revenue = 0
            else:
                revenue = np.random.triangular(rev_min[idx_rev], rev_mode[idx_rev], rev_max[idx_rev])

            # Costi variabili
            if cs_min[idx_cs] == cs_mode[idx_cs] == cs_max[idx_cs] == 0:
                cs = 0
            else:
                cs = np.random.triangular(cs_min[idx_cs], cs_mode[idx_cs], cs_max[idx_cs])

            # CAPEX e disposal
            capex_y = capex[idx_capex]
            if disposal_min[idx_disp] == disposal_mode[idx_disp] == disposal_max[idx_disp] == 0:
                disposal_y = 0
            else:
                disposal_y = np.random.triangular(disposal_min[idx_disp], disposal_mode[idx_disp], disposal_max[idx_disp])

            # EBITDA e FCF
            ebitda = revenue + cs + costs_fixed[y]
            ebit = ebitda + amort[y]
            taxes = -ebit * tax_rate
            fcf = ebitda + taxes + capex_y + disposal_y + change_wc[y]

            # PV
            fcf_pv = fcf / ((1 + discount_rate) ** (y+1))

            fcf_matrix[i, y] = fcf
            fcf_pv_matrix[i, y] = fcf_pv

    # Calcolo NPV cumulato
    for i in range(n_sim):
        npv = np.sum(fcf_pv_matrix[i, :])
        npv_cum = np.cumsum(fcf_pv_matrix[i, :])
        npv_list.append(npv)
        npv_cum_matrix[i, :] = npv_cum

    return np.array(npv_list), fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col, costs_fixed, capex

# -----------------------------
# STREAMLIT APP
# -----------------------------

# Carica logo
logo = Image.open("Image/logo_fibercop.PNG")
st.set_page_config(page_title="NPV @Risk Tool by ERM Fibercop", page_icon=logo, layout="wide")

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(logo, width=300)
st.markdown("""
<h1 style='color: white; font-weight: 800; font-family: Arial, sans-serif;'>NPV @Risk Simulation Tool by ERM</h1>
<p style='color: #cccccc; font-size: 18px; font-family: Arial, sans-serif;'>
Simula scenari finanziari e analizza i progetti di investimento con DCF
</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# LOGIN SICURO
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
# CONTENUTO APP
# -----------------------------
if st.session_state.logged_in:

    st.title("NPV @Risk Simulation Tool by ERM")
    uploaded_file = st.file_uploader("Carica file Excel", type=['xlsx','xls'])

    # ---------- SIDEBAR PARAMETRI ----------
    st.sidebar.markdown("### ⚙️ Parametri progetto")
    project_name = st.sidebar.text_input("Nome progetto", value="Progetto 1")
    discount_rate = st.sidebar.number_input("Tasso di sconto", value=0.10, step=0.01, format="%.4f")
    tax_rate = st.sidebar.number_input("Aliquota fiscale", value=0.25, step=0.01, format="%.4f")
    n_sim = st.sidebar.number_input("Numero simulazioni", min_value=100, max_value=200000, value=2000, step=100)
    seed = st.sidebar.number_input("Seed (0=random)", value=0)

    # ---------- SIDEBAR SHIFT ----------
    st.sidebar.markdown("### ⏳ Shift Monte Carlo")

    def convert_shift_to_dict(values_str, probs_str):
        vals = [int(x.strip()) for x in values_str.split(",")]
        probs = [float(x.strip()) for x in probs_str.split(",")]
        if len(vals) != len(probs):
            st.error("🚨 Numero valori e probabilità non coincide")
            st.stop()
        if abs(sum(probs) - 1.0) > 0.0001:
            st.error("🚨 Le probabilità devono sommare a 1.0")
            st.stop()
        return {vals[i]: probs[i] for i in range(len(vals))}

    cs_shift_values = st.sidebar.text_input("Valori shift costi variabili", value="0,1")
    cs_shift_probs = st.sidebar.text_input("Probabilità shift costi variabili", value="0.7,0.3")
    capex_shift_values = st.sidebar.text_input("Valori shift CAPEX", value="0,-1")
    capex_shift_probs = st.sidebar.text_input("Probabilità shift CAPEX", value="0.9,0.1")
    rev_shift_values = st.sidebar.text_input("Valori shift ricavi", value="0")
    rev_shift_probs = st.sidebar.text_input("Probabilità shift ricavi", value="1.0")
    disp_shift_values = st.sidebar.text_input("Valori shift disposal", value="0")
    disp_shift_probs = st.sidebar.text_input("Probabilità shift disposal", value="1.0")

    cs_shift_dict = convert_shift_to_dict(cs_shift_values, cs_shift_probs)
    capex_shift_dict = convert_shift_to_dict(capex_shift_values, capex_shift_probs)
    rev_shift_dict = convert_shift_to_dict(rev_shift_values, rev_shift_probs)
    disp_shift_dict = convert_shift_to_dict(disp_shift_values, disp_shift_probs)

    run_button = st.sidebar.button("Esegui simulazione")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)

    if run_button and uploaded_file is not None:

        if seed != 0:
            np.random.seed(int(seed))

        # Run simulations con shift
        npv_array, fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col, costs_fixed, capex = run_simulations(
            df, int(n_sim), float(discount_rate), float(tax_rate),
            shift_rev=rev_shift_dict,
            shift_cs=cs_shift_dict,
            shift_capex=capex_shift_dict,
            shift_disp=disp_shift_dict
        )

        # ------------------------- CALCOLI -------------------------
        payback_array = []
        N4_array = np.arange(fcf_matrix.shape[1]) + 1/6
        for i in range(fcf_matrix.shape[0]):
            npv_cum = np.cumsum(fcf_pv_matrix[i,:])
            pb = np.nan
            for j in range(len(npv_cum)):
                M19 = npv_cum[j-1] if j>0 else 0
                N19 = npv_cum[j]
                N4 = N4_array[j]
                if N19 >= 0:
                    if j==0:
                        pb = N19
                    else:
                        pb = -M19/(N19-M19) + N4 - 1
                    break
            payback_array.append(pb)
        payback_array = np.array(payback_array)

        # IRR e PPI
        n_years = fcf_matrix.shape[1]
        irr_matrix = np.zeros((fcf_matrix.shape[0], n_years))
        for i in range(fcf_matrix.shape[0]):
            for j in range(n_years):
                fcf_subset = fcf_matrix[i,:j+1]
                irr_matrix[i,j] = npf.irr(fcf_subset) if (np.any(fcf_subset<0) and np.any(fcf_subset>0)) else 0

        profit_index_array = []
        for i in range(fcf_matrix.shape[0]):
            fcf = fcf_pv_matrix[i,:]
            npv_cum = np.cumsum(fcf)
            cost_total = np.abs(costs_fixed) + np.abs(capex)
            cost_total = cost_total / ((1+discount_rate)**np.arange(1,n_years+1))
            cost_total_cum = np.cumsum(cost_total)
            profit_index_array.append(npv_cum / cost_total_cum)
        profit_index_array = np.array(profit_index_array)

        # Metriche principali
        expected_npv = np.mean(npv_array)
        percentile_5 = np.percentile(npv_array,5)
        downside_prob = np.mean(npv_array<0)

        st.metric("Expected NPV", f"{expected_npv:,.2f}")
        st.metric("VaR 95% (CaR)", f"{percentile_5:,.2f}")
        st.metric("Probabilità NPV<0", f"{downside_prob*100:.2f}%")

        # Grafici
        st.pyplot(plot_npv_distribution(npv_array, expected_npv, percentile_5, project_name))
        st.pyplot(plot_boxplot(npv_array, project_name))
        st.pyplot(plot_cashflows(fcf_matrix, fcf_matrix.shape[1], project_name))
        st.pyplot(plot_cumulative_npv(npv_cum_matrix, project_name))
        st.pyplot(plot_payback_distribution(payback_array, project_name))
        st.pyplot(plot_irr_trends(
            np.nanmin(irr_matrix, axis=0),
            np.nanpercentile(irr_matrix,5,axis=0),
            np.nanpercentile(irr_matrix,50,axis=0),
            np.nanpercentile(irr_matrix,95,axis=0),
            np.nanmax(irr_matrix, axis=0),
            years_labels=df['Anno'].to_list(),
            title="Andamento IRR per anno",
            figsize=(10,6)
        ))
        st.pyplot(plot_ppi_distribution(
            np.nanmin(profit_index_array,axis=0),
            np.nanpercentile(profit_index_array,5,axis=0),
            np.nanpercentile(profit_index_array,50,axis=0),
            np.nanpercentile(profit_index_array,95,axis=0),
            np.nanmax(profit_index_array,axis=0),
            years_labels=df['Anno'].to_list(),
            title="Andamento PPI per anno",
            figsize=(10,6)
        ))

        # KRI Gauges
        st.plotly_chart(plot_car_kri(percentile_5, expected_npv, project_name))
        st.plotly_chart(plot_probs_kri(downside_prob, project_name))

        # Export Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({'Simulazione': np.arange(1,len(npv_array)+1),'NPV':npv_array}).to_excel(writer,index=False,sheet_name='NPV')
            pd.DataFrame(fcf_matrix, columns=years_col).to_excel(writer,index=False,sheet_name='FCF_simulati')
            pd.DataFrame(fcf_pv_matrix, columns=years_col).to_excel(writer,index=False,sheet_name='DCF_simulati')
        st.download_button("Scarica Excel", data=output.getvalue(), file_name=f"{project_name}_sim.xlsx")

else:
    st.info("🔹 Completa il login per accedere alla web-app!")
