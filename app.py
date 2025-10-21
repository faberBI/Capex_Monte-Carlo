import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import numpy_financial as npf

from capex.visuals import (plot_npv_distribution, plot_boxplot, plot_cashflows , plot_cumulative_npv, plot_payback_distribution, plot_probs_kri, plot_car_kri, plot_irr_trends)

from capex.montecarlo import (triangular_sample, run_simulations)

import hashlib
from oauth2client.service_account import ServiceAccountCredentials
import json

from PIL import Image
import streamlit as st

# Carica il logo
logo = Image.open("Image/logo_fibercop.PNG")

st.set_page_config(page_title="NPV @Risk Tool by ERM Fibercop", page_icon=logo , layout="wide")
st.markdown("""
<div style='text-align: center;'>
""", unsafe_allow_html=True)

st.image(logo, width=300)  # logo centrato grazie al div

st.markdown("""
<h1 style='color: white; font-weight: 800; font-family: Arial, sans-serif;'>
NPV @Risk Tool
</h1>
<p style='color: #cccccc; font-size: 18px; font-family: Arial, sans-serif;'>
Simula scenari finanziari e analizza il rischio con la potenza della Monte Carlo
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
# CONTENUTO DELL'APP
# -----------------------------
if st.session_state.logged_in:
    
# ------------------------- Streamlit UI -------------------------
    st.title("NPV @Risk Monte Carlo Simulator")
    
    uploaded_file = st.file_uploader("Carica file Excel", type=['xlsx','xls'])
    
    with st.sidebar:
        project_name = st.text_input("Nome progetto", value="Progetto 1")
        discount_rate = st.number_input("Tasso di sconto (es. 0.10)", value=0.10, step=0.01, format="%.4f")
        tax_rate = st.number_input("Aliquota fiscale (es. 0.25)", value=0.25, step=0.01, format="%.4f")
        n_sim = st.number_input("Numero simulazioni", min_value=100, max_value=200000, value=2000, step=100)
        seed = st.number_input("Seed (0=random)", value=0)
        run_button = st.button("Esegui simulazione")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
    
    if run_button and uploaded_file is not None:
        if seed !=0:
            np.random.seed(int(seed))
        npv_array, fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col = run_simulations(df, int(n_sim), float(discount_rate), float(tax_rate))
    
        payback_array = []
        N4_array = np.arange(fcf_matrix.shape[1]) + 1/6  # frazione iniziale anno
        
        for i in range(fcf_matrix.shape[0]):
            npv_cum = np.cumsum(fcf_pv_matrix[i,:])
            pb = np.nan  # default se NPV cumulato resta negativo
        
            for j in range(len(npv_cum)):
                M19 = npv_cum[j-1] if j > 0 else 0
                N19 = npv_cum[j]
                N4 = N4_array[j]
        
                if N19 >= 0:
                    if j == 0:
                        pb = N19  # già positivo nel primo anno
                    else:
                        pb = -M19 / (N19 - M19) + N4 - 1
                    break  # payback trovato
                
            payback_array.append(pb)
        
        payback_array = np.array(payback_array)
        
        
        # ------------------------- IRR per anno -------------------------
        n_years = fcf_matrix.shape[1]
        irr_matrix = np.zeros((fcf_matrix.shape[0], n_years))
        
        for i in range(fcf_matrix.shape[0]):
            for j in range(n_years):
                fcf_subset = fcf_matrix[i, :j+1]  # flussi fino all'anno j
                # IRR calcolabile solo se ci sono flussi negativi e positivi
                if np.any(fcf_subset < 0) and np.any(fcf_subset > 0):
                    irr_matrix[i, j] = npf.irr(fcf_subset)
                else:
                    irr_matrix[i, j] = 0
        
        # Percentili IRR per anno
        irr_min = np.nanmin(irr_matrix, axis=0)
        irr_p5 = np.nanpercentile(irr_matrix, 5, axis=0)
        irr_p50 = np.nanpercentile(irr_matrix, 50, axis=0)
        irr_p95 = np.nanpercentile(irr_matrix, 95, axis=0)
        irr_max = np.nanmax(irr_matrix, axis=0)
    
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
        st.pyplot(plot_irr_trends(irr_min, irr_p5, irr_p50, irr_p95, irr_max, years_labels=df['Anno'].to_list(), title="Andamento IRR per anno", figsize=(10,6)))
        
    
        # ------------------------- KRI Gauges -------------------------
        st.plotly_chart(plot_car_kri(percentile_5, expected_npv, project_name))
        fig_prob = plot_probs_kri(downside_prob, project_name)
        st.plotly_chart(fig_prob)
    
    # Export Excel multi-sheet
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # NPV simulati
        pd.DataFrame({'Simulazione': np.arange(1,len(npv_array)+1), 'NPV': npv_array}).to_excel(writer, index=False, sheet_name='NPV')
        
        # FCF simulati
        df_fcf = pd.DataFrame(fcf_matrix, columns=years_col)
        df_fcf.insert(0, 'Simulazione', np.arange(1, fcf_matrix.shape[0]+1))
        df_fcf.to_excel(writer, index=False, sheet_name='FCF_simulati')
        
        # DCF simulati (FCF scontati)
        df_dcf = pd.DataFrame(fcf_pv_matrix, columns=years_col)
        df_dcf.insert(0, 'Simulazione', np.arange(1, fcf_pv_matrix.shape[0]+1))
        df_dcf.to_excel(writer, index=False, sheet_name='DCF_simulati')
        
        # Percentili FCF
        median_fcf = np.median(fcf_matrix, axis=0)
        p5_fcf = np.percentile(fcf_matrix,5,axis=0)
        p95_fcf = np.percentile(fcf_matrix,95,axis=0)
        pd.DataFrame({'Anno': years_col, 'Median': median_fcf, 'P5': p5_fcf, 'P95': p95_fcf}).to_excel(writer, index=False, sheet_name='FCF_percentili')
        
        # Percentili DCF
        median_dcf = np.median(fcf_pv_matrix, axis=0)
        p5_dcf = np.percentile(fcf_pv_matrix,5,axis=0)
        p95_dcf = np.percentile(fcf_pv_matrix,95,axis=0)
        pd.DataFrame({'Anno': years_col, 'Median': median_dcf, 'P5': p5_dcf, 'P95': p95_dcf}).to_excel(writer, index=False, sheet_name='DCF_percentili')
        
    
        # Payback period
        pd.DataFrame({'Simulazione': np.arange(1,len(payback_array)+1), 'PaybackYear': payback_array}).to_excel(writer, index=False, sheet_name='Payback_period')
        
        
        # Percentili IRR
        pd.DataFrame({
        'Anno': years_col,
        'IRR_min': irr_min,
        'IRR_p5': irr_p5,
        'IRR_p50': irr_p50,
        'IRR_p95': irr_p95,
        'IRR_max': irr_max,
        }).to_excel(writer, index=False, sheet_name='IRR_percentili')
    
    st.download_button("Scarica Excel", data=output.getvalue(), file_name=f"{project_name}_sim.xlsx")
else:
    st.info("🔹 Completa il login per accedere alla web-app!")














