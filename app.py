import streamlit as st
from capex.wacc import calculate_wacc
from capex.montecarlo import run_montecarlo
from capex.visuals import plot_npv_distribution, plot_boxplot, plot_cashflows, plot_risk_return_matrix
import pandas as pd
import numpy as np

# init session state
if "projects" not in st.session_state:
    st.session_state.projects = []

def add_project():
    st.session_state.projects.append({
        "name": f"Progetto {len(st.session_state.projects)+1}",
        "equity": 0.5, "debt": 0.5,
        "ke": 0.10, "kd": 0.05, "tax": 0.30,
        "capex": 200.0, "years": 10,
        "revenues": {
            "price": {"dist": "Normale", "p1": 100.0, "p2": 10.0},
            "quantity": {"dist": "Normale", "p1": 1000.0, "p2": 100.0}
        },
        "costs": {"var_pct": 0.08, "fixed": -50.0},
        "price_growth": [0.01]*10,
        "quantity_growth": [0.05]*10,
        "fixed_cost_inflation": [0.02]*10
    })

# --- UI
st.title("📊 CAPEX Risk Framework con WACC & Trend annuali")
st.button("➕ Aggiungi progetto", on_click=add_project)
n_sim = st.slider("Numero simulazioni Monte Carlo", 1000, 1000_000, 10_000)

results = []

for i, proj in enumerate(st.session_state.projects):
    with st.expander(f"⚙️ Parametri {proj['name']}", expanded=True):
        proj["equity"] = st.slider("Peso Equity", 0.0, 1.0, proj["equity"], key=f"equity_{i}")
        proj["debt"] = 1 - proj["equity"]
        proj["ke"] = st.number_input("Costo Equity (ke)", value=proj["ke"], key=f"ke_{i}")
        proj["kd"] = st.number_input("Costo Debito (kd)", value=proj["kd"], key=f"kd_{i}")
        proj["tax"] = st.number_input("Tax Rate", value=proj["tax"], key=f"tax_{i}")
        proj["capex"] = st.number_input("CAPEX iniziale", value=proj["capex"], key=f"capex_{i}")
        proj["years"] = st.slider("Orizzonte temporale (anni)", 1, 20, proj["years"], key=f"years_{i}")

        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        st.write(f"**WACC calcolato:** {wacc:.2%}")

    # --- Simulazione
    sim_result = run_montecarlo(proj, n_sim, wacc)
    results.append({"name": proj["name"], **sim_result})

    # --- Visuals
    st.subheader(f"📊 Risultati {proj['name']}")
    st.write(f"Expected NPV: {sim_result['expected_npv']:.2f}")
    st.write(f"CaR (95%): {sim_result['car']:.2f}")
    st.write(f"Probabilità NPV < 0: {sim_result['downside_prob']*100:.1f}%")
    st.write(f"Conditional VaR (95%): {sim_result['cvar']:.2f}")

    st.pyplot(plot_npv_distribution(sim_result["npv_array"], sim_result["expected_npv"], 
                                    np.percentile(sim_result["npv_array"], 5), proj["name"]))
    st.pyplot(plot_boxplot(sim_result["npv_array"], proj["name"]))
    st.pyplot(plot_cashflows(sim_result["yearly_cash_flows"], proj["years"], proj["name"]))

if results:
    st.subheader("📌 Matrice rischio-rendimento")
    st.pyplot(plot_risk_return_matrix(results))

