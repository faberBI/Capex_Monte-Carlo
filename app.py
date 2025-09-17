import streamlit as st
import numpy as np
import io
import pandas as pd
import time
from openai import OpenAI, OpenAIError

from capex.wacc import calculate_wacc
from capex.montecarlo import run_montecarlo
from capex.visuals import (
    plot_npv_distribution,
    plot_boxplot,
    plot_cashflows,
    plot_risk_return_matrix,
    get_dynamic_thresholds,
    plot_car_kri
)
api_key = "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"

# ------------------ Helper per sample distribuzioni ------------------
def sample(dist_obj):
    if dist_obj["dist"] == "Normale":
        return np.random.normal(dist_obj["p1"], dist_obj["p2"])
    elif dist_obj["dist"] == "Triangolare":
        return np.random.triangular(dist_obj["p1"], dist_obj["p2"], dist_obj.get("p3", dist_obj["p1"] + dist_obj["p2"]))
    elif dist_obj["dist"] == "Lognormale":
        return np.random.lognormal(dist_obj["p1"], dist_obj["p2"])
    elif dist_obj["dist"] == "Uniforme":
        return np.random.uniform(dist_obj["p1"], dist_obj["p2"])
    else:
        return dist_obj["p1"]

# ------------------ Session state ------------------
if "projects" not in st.session_state:
    st.session_state.projects = []
if "results" not in st.session_state:
    st.session_state.results = None

# ------------------ Funzione per aggiungere progetto ------------------
def add_project():
    st.session_state.projects.append({
        "name": f"Progetto {len(st.session_state.projects)+1}",
        "equity": 0.5,
        "debt": 0.5,
        "ke": 0.10,
        "kd": 0.05,
        "tax": 0.30,
        "capex": 200.0,
        "years": 10,
        "capex_rec": [0.0]*10,  # lista default
        "revenues": {
            "price": {"dist": "Normale", "p1": 100.0, "p2": 10.0},
            "quantity": {"dist": "Normale", "p1": 1000.0, "p2": 100.0}
        },
        "costs": {"var_pct": 0.08, "fixed": -50.0},
        "price_growth": [0.01]*10,
        "quantity_growth": [0.05]*10,
        "fixed_cost_inflation": [0.02]*10
    })

# ------------------ UI ------------------
st.title("📊 CAPEX Risk Framework con WACC & CAPEX Ricorrente")

st.button("➕ Aggiungi progetto", on_click=add_project)
n_sim = st.slider("Numero simulazioni Monte Carlo", 1000, 1000_000, 10_000)

# ------------------ Loop progetti ------------------
for i, proj in enumerate(st.session_state.projects):
    with st.expander(f"⚙️ Parametri {proj['name']}", expanded=True):
        # Parametri finanziari
        proj["name"] = st.text_input("Nome progetto", value=proj["name"], key=f"name_{i}")
        proj["equity"] = st.slider("Peso Equity", 0.0, 1.0, proj["equity"], key=f"equity_{i}")
        proj["debt"] = 1 - proj["equity"]
        proj["ke"] = st.number_input("Costo Equity (ke)", value=proj["ke"], key=f"ke_{i}")
        proj["kd"] = st.number_input("Costo Debito (kd)", value=proj["kd"], key=f"kd_{i}")
        proj["tax"] = st.number_input("Tax Rate", value=proj["tax"], key=f"tax_{i}")
        proj["capex"] = st.number_input("CAPEX iniziale", value=proj["capex"], key=f"capex_{i}")
        proj["years"] = st.slider("Orizzonte temporale (anni)", 1, 20, proj["years"], key=f"years_{i}")

        
        # ------------------ CAPEX Ricorrente per anno ------------------
        st.subheader("🏗️ CAPEX Ricorrente (anno per anno)")

# Riallinea capex_rec alla lunghezza degli anni
        if len(proj["capex_rec"]) < proj["years"]:
            proj["capex_rec"] += [0.0] * (proj["years"] - len(proj["capex_rec"]))
        elif len(proj["capex_rec"]) > proj["years"]:
            proj["capex_rec"] = proj["capex_rec"][:proj["years"]]

        df_capex = pd.DataFrame({
        "Anno": list(range(1, proj["years"]+1)),
        "CAPEX Ricorrente": proj["capex_rec"]
        })

        df_capex_edit = st.data_editor(
        df_capex,
        key=f"capex_rec_{i}",
        num_rows="dynamic"
        )
        proj["capex_rec"] = df_capex_edit["CAPEX Ricorrente"].tolist()


        # ------------------ Ricavi ------------------
        st.subheader("📈 Ricavi")
        for key, label in [("price", "Prezzo"), ("quantity", "Quantità")]:
            dist = st.selectbox(
                f"Distribuzione {label}",
                ["Normale","Triangolare","Lognormale","Uniforme"],
                index=["Normale","Triangolare","Lognormale","Uniforme"].index(proj["revenues"][key]["dist"]),
                key=f"{key}_dist_{i}"
            )
            proj["revenues"][key]["dist"] = dist
            proj["revenues"][key]["p1"] = st.number_input(f"{label} - Param 1", value=proj["revenues"][key]["p1"], key=f"{key}_p1_{i}")
            proj["revenues"][key]["p2"] = st.number_input(f"{label} - Param 2", value=proj["revenues"][key]["p2"], key=f"{key}_p2_{i}")
            if dist == "Triangolare":
                proj["revenues"][key]["p3"] = st.number_input(
                    f"{label} - Param 3 (max)",
                    value=proj["revenues"][key].get("p3", proj["revenues"][key]["p1"]+proj["revenues"][key]["p2"]),
                    key=f"{key}_p3_{i}"
                )

        # ------------------ Costi ------------------
        st.subheader("💸 Costi")
        proj["costs"]["var_pct"] = st.number_input("% Costi Variabili sui ricavi", value=proj["costs"]["var_pct"], min_value=0.0, max_value=1.0, step=0.01, key=f"var_pct_{i}")
        proj["costs"]["fixed"] = st.number_input("Costi Fissi annui", value=proj["costs"]["fixed"], step=1.0, key=f"fixed_{i}")

        # ------------------ Trend annuali ------------------
        st.subheader("📊 Trend annuali")
        proj.setdefault("price_growth", [0.0]*proj["years"])
        proj.setdefault("quantity_growth", [0.0]*proj["years"])
        proj.setdefault("fixed_cost_inflation", [0.0]*proj["years"])
        for t in range(proj["years"]):
            proj["price_growth"][t] = st.slider(
                f"Crescita prezzo anno {t+1} (%)",
                min_value=-0.5, max_value=0.5,
                value=proj["price_growth"][t],
                step=0.05, key=f"pg_{i}_{t}"
            )
            proj["quantity_growth"][t] = st.slider(
                f"Crescita quantità anno {t+1} (%)",
                min_value=-0.5, max_value=0.5,
                value=proj["quantity_growth"][t],
                step=0.05, key=f"qg_{i}_{t}"
            )
            proj["fixed_cost_inflation"][t] = st.slider(
                f"Crescita costi fissi anno {t+1} (%)",
                min_value=-0.5, max_value=0.5,
                value=proj["fixed_cost_inflation"][t],
                step=0.05, key=f"fi_{i}_{t}"
            )

        # WACC
        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        st.write(f"**WACC calcolato:** {wacc:.2%}")

# ------------------ Avvio simulazioni ------------------
if st.button("▶️ Avvia simulazioni"):
    results = []
    for proj in st.session_state.projects:
        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        sim_result = run_montecarlo(proj, n_sim, wacc)
        results.append({"name": proj["name"], **sim_result})

        st.subheader(f"📊 Risultati {proj['name']}")
        st.write(f"Expected NPV: {sim_result['expected_npv']:.2f}")
        st.write(f"CaR (95%): {sim_result['car']:.2f}")
        st.write(f"Probabilità NPV < 0: {sim_result['downside_prob']*100:.1f}%")
        st.write(f"Conditional VaR (95%): {sim_result['cvar']:.2f}")

        st.pyplot(plot_npv_distribution(sim_result["npv_array"], sim_result["expected_npv"], 
                                        np.percentile(sim_result["npv_array"], 5), proj["name"]))
        st.pyplot(plot_boxplot(sim_result["npv_array"], proj["name"]))
        st.pyplot(plot_cashflows(sim_result["yearly_cash_flows"], proj["years"], proj["name"]))
        # 🔥 Tachimetro rischio (KRI) basato su CaR
        st.plotly_chart(
        plot_car_kri(sim_result["car"], sim_result["expected_npv"], proj["name"]),
        use_container_width=True
        )
        # Testo sintetico KRI    
        car_pct = sim_result["car"] / sim_result["expected_npv"] if sim_result["expected_npv"] != 0 else 1.0

        if car_pct > 0.5:
            kri_text = "🔴 Rischio Alto"
        elif car_pct > 0.25:
            kri_text = "🟡 Rischio Medio"
        else:
            kri_text = "🟢 Rischio Basso"

        st.markdown(f"**KRI sintetico:** {kri_text} ({car_pct*100:.1f}% del valore atteso)")
                    
    st.session_state.results = results

# ------------------ Matrice rischio-rendimento ------------------
if st.session_state.results:
    results = st.session_state.results
    st.subheader("📌 Matrice rischio-rendimento")
    st.pyplot(plot_risk_return_matrix(results))

    # Crea client OpenAI
    client = OpenAI(api_key=api_key)

    summary_df = pd.DataFrame([
        {
            "name": r["name"],
            "expected_npv": r["expected_npv"],
            "car": r["car"],
            "downside_prob": r["downside_prob"],
            "cvar": r["cvar"],
            "avg_yearly_cashflow": np.mean(r["yearly_cash_flows"])
        }
        for r in results
    ])

    prompt = f"""
Ecco i risultati sintetici dei progetti (Monte Carlo CAPEX Risk):

{summary_df.to_string(index=False)}

Fornisci un commento sintetico e professionale, evidenziando:
- Miglior trade-off rischio/rendimento.
- Robustezza dei NPV stimati.
- Eventuali rischi particolari emersi dalle simulazioni.
"""

    if st.button("💬 Genera commento GPT"):
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Sei un analista finanziario esperto in valutazioni CAPEX."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.subheader("📑 Commento di GPT")
                st.write(response.choices[0].message.content)
                break
            except OpenAIError:
                st.warning("Errore OpenAI (es. rate limit), riprovo tra 5 secondi...")
                time.sleep(5)

# ------------------ Export risultati in Excel ------------------
if st.session_state.results:
    results = st.session_state.results
    st.subheader("💾 Esporta risultati")
    
    df_summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "npv_array"} for r in results
    ])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, index=False, sheet_name='Risultati')
        for r in results:
            npv_df = pd.DataFrame(r["npv_array"], columns=["NPV"])
            sheet_name = r["name"][:31]
            npv_df.to_excel(writer, index=False, sheet_name=sheet_name)
            
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Scarica risultati in Excel",
        data=excel_data,
        file_name="capex_risultati.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )








