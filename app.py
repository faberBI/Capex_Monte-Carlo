import streamlit as st
import numpy as np
import pandas as pd
import io
import time
from openai import OpenAI, OpenAIError

from capex.wacc import calculate_wacc
from capex.montecarlo import run_montecarlo
from capex.visuals import (
    plot_npv_distribution,
    plot_boxplot,
    plot_cashflows,
    plot_risk_return_matrix,
    plot_car_kri,
    plot_discounted_pbp
)

api_key = st.secrets["OPENAI_API_KEY"]

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
        "capex_rec": None,  # ora fissi anno per anno
        "fixed_costs": None,  # ora fissi anno per anno
        "revenues_list": [
            {
                "name": "Ricavo 1",
                "price": [{"dist": "Normale", "p1": 100.0, "p2": 10.0} for _ in range(10)],
                "quantity": [{"dist": "Normale", "p1": 1000.0, "p2": 100.0} for _ in range(10)]
            }
        ],
        "costs": {"var_pct": 0.08},
        "other_costs": [],
        "price_growth": [0.01]*10,
        "quantity_growth": [0.05]*10,
        "depreciation": [20.0]*10
    })

# ------------------ UI ------------------
st.title("📊 CAPEX @Risk Framework by ERM")
st.button("➕ Aggiungi progetto", on_click=add_project)
n_sim = st.slider("Numero simulazioni Monte Carlo", 1000, 1_000_000, 10_000)

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

        # ------------------ CAPEX Ricorrente (fisso anno per anno) ------------------
        st.subheader("🏗️ CAPEX Ricorrente (anno per anno, valori fissi)")
        if proj.get("capex_rec") is None:
            if st.button(f"➕ Aggiungi CAPEX Ricorrente", key=f"add_capex_rec_{i}"):
                proj["capex_rec"] = [0.0 for _ in range(proj["years"])]

        if proj.get("capex_rec") is not None:
            for y in range(proj["years"]):
                proj["capex_rec"][y] = st.number_input(f"CAPEX anno {y+1}", value=proj["capex_rec"][y], key=f"capex_rec_{i}_{y}")

        # ------------------ Ricavi multipli con distribuzione ------------------
        st.subheader("📈 Ricavi")
        for j, rev in enumerate(proj["revenues_list"]):
            st.markdown(f"**{rev['name']}**")
            for key in ["price","quantity"]:
                for y in range(proj["years"]):
                    st.markdown(f"Anno {y+1} - {key}")
                    dist_type = st.selectbox(
                        "Distribuzione",
                        ["Normale","Triangolare","Lognormale","Uniforme"],
                        index=["Normale","Triangolare","Lognormale","Uniforme"].index(rev[key][y]["dist"]),
                        key=f"{key}_dist_{i}_{j}_{y}"
                    )
                    rev[key][y]["dist"] = dist_type
                    if dist_type == "Normale":
                        rev[key][y]["p1"] = st.number_input("Media (p1)", value=rev[key][y].get("p1",0.0), key=f"{key}_p1_{i}_{j}_{y}")
                        rev[key][y]["p2"] = st.number_input("Deviazione standard (p2)", value=rev[key][y].get("p2",0.0), key=f"{key}_p2_{i}_{j}_{y}")
                    elif dist_type == "Triangolare":
                        rev[key][y]["p1"] = st.number_input("Minimo (p1)", value=rev[key][y].get("p1",0.0), key=f"{key}_p1_{i}_{j}_{y}")
                        rev[key][y]["p2"] = st.number_input("Modal (p2)", value=rev[key][y].get("p2",0.0), key=f"{key}_p2_{i}_{j}_{y}")
                        rev[key][y]["p3"] = st.number_input("Massimo (p3)", value=rev[key][y].get("p3",0.0), key=f"{key}_p3_{i}_{j}_{y}")
                    elif dist_type == "Lognormale":
                        rev[key][y]["p1"] = st.number_input("Media log (p1)", value=rev[key][y].get("p1",0.0), key=f"{key}_p1_{i}_{j}_{y}")
                        rev[key][y]["p2"] = st.number_input("Deviazione log (p2)", value=rev[key][y].get("p2",0.0), key=f"{key}_p2_{i}_{j}_{y}")
                    elif dist_type == "Uniforme":
                        rev[key][y]["p1"] = st.number_input("Minimo (p1)", value=rev[key][y].get("p1",0.0), key=f"{key}_p1_{i}_{j}_{y}")
                        rev[key][y]["p2"] = st.number_input("Massimo (p2)", value=rev[key][y].get("p2",0.0), key=f"{key}_p2_{i}_{j}_{y}")
        if st.button(f"➕ Aggiungi voce di ricavo al progetto {proj['name']}", key=f"add_revenue_{i}"):
            proj["revenues_list"].append({
                "name": f"Ricavo {len(proj['revenues_list'])+1}",
                "price": [{"dist":"Normale","p1":100.0,"p2":10.0} for _ in range(proj["years"])],
                "quantity": [{"dist":"Normale","p1":1000.0,"p2":100.0} for _ in range(proj["years"])]
            })

        # ------------------ Costi Variabili ------------------
        st.subheader("💸 Costi Variabili")
        proj["costs"]["var_pct"] = st.number_input("% Costi Variabili sui ricavi", value=proj["costs"]["var_pct"], min_value=0.0, max_value=1.0, step=0.01, key=f"var_pct_{i}")

        # ------------------ Costi Fissi anno per anno ------------------
        st.subheader("💸 Costi Fissi annui")
        if proj.get("fixed_costs") is None or len(proj["fixed_costs"]) != proj["years"]:
            proj["fixed_costs"] = [0.0]*proj["years"]
        for y in range(proj["years"]):
            proj["fixed_costs"][y] = st.number_input(f"Costi fissi anno {y+1}", value=proj["fixed_costs"][y], key=f"fixed_{i}_{y}")

        # ------------------ Costi aggiuntivi stocastici ------------------
        st.subheader("📉 Costi aggiuntivi")
        proj.setdefault("other_costs", [])
        for j, cost in enumerate(proj["other_costs"]):
            st.markdown(f"**{cost['name']}**")
            for year_idx in range(proj["years"]):
                st.markdown(f"Anno {year_idx+1}")
                year_cost = cost["values"][year_idx]
                dist_options = ["Normale", "Triangolare", "Lognormale", "Uniforme"]
                selected_dist = st.selectbox(
                    f"Distribuzione anno {year_idx+1} - {cost['name']}",
                    options=dist_options,
                    index=dist_options.index(year_cost.get("dist", "Normale")),
                    key=f"oc_dist_{i}_{j}_{year_idx}"
                )
                year_cost["dist"] = selected_dist
                if selected_dist == "Normale":
                    year_cost["p1"] = st.number_input(f"Media (p1) anno {year_idx+1}", value=year_cost.get("p1",0.0), key=f"oc_n_p1_{i}_{j}_{year_idx}")
                    year_cost["p2"] = st.number_input(f"Std Dev (p2) anno {year_idx+1}", value=year_cost.get("p2",0.0), key=f"oc_n_p2_{i}_{j}_{year_idx}")
                elif selected_dist == "Triangolare":
                    year_cost["p1"] = st.number_input(f"Minimo (p1) anno {year_idx+1}", value=year_cost.get("p1",0.0), key=f"oc_t_p1_{i}_{j}_{year_idx}")
                    year_cost["p2"] = st.number_input(f"Modalità (p2) anno {year_idx+1}", value=year_cost.get("p2",0.0), key=f"oc_t_p2_{i}_{j}_{year_idx}")
                    year_cost["p3"] = st.number_input(f"Massimo (p3) anno {year_idx+1}", value=year_cost.get("p3",0.0), key=f"oc_t_p3_{i}_{j}_{year_idx}")
                elif selected_dist == "Lognormale":
                    year_cost["p1"] = st.number_input(f"Mu (p1) anno {year_idx+1}", value=year_cost.get("p1",0.0), key=f"oc_l_p1_{i}_{j}_{year_idx}")
                    year_cost["p2"] = st.number_input(f"Sigma (p2) anno {year_idx+1}", value=year_cost.get("p2",0.0), key=f"oc_l_p2_{i}_{j}_{year_idx}")
                elif selected_dist == "Uniforme":
                    year_cost["p1"] = st.number_input(f"Min (p1) anno {year_idx+1}", value=year_cost.get("p1",0.0), key=f"oc_u_p1_{i}_{j}_{year_idx}")
                    year_cost["p2"] = st.number_input(f"Max (p2) anno {year_idx+1}", value=year_cost.get("p2",0.0), key=f"oc_u_p2_{i}_{j}_{year_idx}")
        if st.button(f"➕ Aggiungi costo stocastico al progetto {proj['name']}", key=f"add_oc_{i}"):
            proj["other_costs"].append({
                "name": f"Costo {len(proj['other_costs'])+1}",
                "values": [{"dist":"Normale","p1":0.0,"p2":0.0} for _ in range(proj["years"])]
            })

        # ------------------ Ammortamento ------------------
        st.subheader("🏗️ Ammortamento (Depreciation)")
        if "depreciation" not in proj or len(proj["depreciation"]) != proj["years"]:
            proj["depreciation"] = [proj["capex"]/proj["years"]]*proj["years"]
        df_dep = pd.DataFrame({"Anno": range(1, proj["years"]+1), "Ammortamento": proj["depreciation"]})
        df_dep_edit = st.data_editor(df_dep, key=f"dep_{i}", num_rows="dynamic")
        proj["depreciation"] = df_dep_edit["Ammortamento"].tolist()

        # ------------------ Trend annuali ------------------
        st.subheader("📊 Trend annuali")
        proj.setdefault("price_growth", [0.0]*proj["years"])
        proj.setdefault("quantity_growth", [0.0]*proj["years"])
        for t in range(proj["years"]):
            proj["price_growth"][t] = st.slider(f"Crescita prezzo anno {t+1} (%)", -0.5, 0.5, proj["price_growth"][t], 0.05, key=f"pg_{i}_{t}")
            proj["quantity_growth"][t] = st.slider(f"Crescita quantità anno {t+1} (%)", -0.5, 0.5, proj["quantity_growth"][t], 0.05, key=f"qg_{i}_{t}")

        # WACC
        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        st.write(f"**WACC calcolato:** {wacc:.2%}")

# ------------------ Funzione Monte Carlo aggiornata ------------------
def sample(dist_obj, year_idx=None):
    """Campionamento stocastico solo per other_costs o ricavi, non per CAPEX o costi fissi."""
    # Se è una lista, seleziona anno
    if isinstance(dist_obj, list):
        if year_idx is None:
            raise ValueError("year_idx deve essere specificato per liste di distribuzioni anno per anno")
        dist_obj = dist_obj[year_idx]
    dist_type = dist_obj.get("dist", "Normale")
    p1 = dist_obj.get("p1", 0.0)
    p2 = dist_obj.get("p2", 0.0)
    p3 = dist_obj.get("p3", p1+p2)
    if dist_type == "Normale":
        return np.random.normal(p1, p2)
    elif dist_type == "Triangolare":
        return np.random.triangular(p1, p2, p3)
    elif dist_type == "Lognormale":
        return np.random.lognormal(p1, p2)
    elif dist_type == "Uniforme":
        return np.random.uniform(p1, p2)
    else:
        raise ValueError(f"Distribuzione non supportata: {dist_type}")

# ------------------ Avvio simulazioni ------------------
if st.button("▶️ Avvia simulazioni"):
    results = []
    for proj in st.session_state.projects:
        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        sim_result = run_montecarlo(proj, n_sim, wacc)
        results.append({"name": proj["name"], **sim_result})

        # Visualizzazione
        st.subheader(f"📊 Risultati {proj['name']}")
        st.write(f"Expected NPV: {sim_result['expected_npv']:.2f}")
        st.write(f"CaR (95%): {sim_result['car']:.2f}")
        st.write(f"Probabilità NPV < 0: {sim_result['downside_prob']*100:.1f}%")
        st.write(f"Conditional VaR (95%): {sim_result['cvar']:.2f}")

        # ------------------ Grafici classici ------------------
        st.pyplot(plot_npv_distribution(sim_result["npv_array"], sim_result["expected_npv"], 
                                        np.percentile(sim_result["npv_array"], 5), proj["name"]))
        st.pyplot(plot_boxplot(sim_result["npv_array"], proj["name"]))
        st.pyplot(plot_cashflows(sim_result["yearly_cash_flows"], proj["years"], proj["name"]))
        st.plotly_chart(plot_car_kri(sim_result["car"], sim_result["expected_npv"], proj["name"]), use_container_width=True)

        # ------------------ KRI corretto ------------------
        kri_pct = (sim_result["expected_npv"] - sim_result["car"]) / sim_result["expected_npv"] if sim_result["expected_npv"] != 0 else 1.0
        kri_text = "🔴 Rischio Alto" if kri_pct > 0.5 else ("🟡 Rischio Medio" if kri_pct > 0.25 else "🟢 Rischio Basso")
        st.markdown(f"**KRI sintetico:** {kri_text} ({kri_pct*100:.1f}% del valore atteso)")

        # ------------------ NPV cumulato e Payback ------------------
        npv_cum_matrix = np.cumsum(sim_result['yearly_cash_flows'], axis=1)  # cumulativo anno per anno
        sim_result['npv_cum_matrix'] = npv_cum_matrix

        # PBP mediana (anno in cui NPV mediano diventa positivo)
        median_cum_npv = np.median(npv_cum_matrix, axis=0)
        pbp_median = np.argmax(median_cum_npv > 0) + 1  # +1 perché gli anni partono da 1
        st.write(f"**PBP attualizzato medio (mediana NPV):** {pbp_median} anni")

        # Grafico NPV cumulato a serie storica
        st.pyplot(plot_cumulative_npv(npv_cum_matrix, proj['name']))

    st.session_state.results = results

# ------------------ Matrice rischio-rendimento e GPT ------------------
if st.session_state.results:
    results = st.session_state.results
    st.subheader("📌 Matrice rischio-rendimento")
    st.pyplot(plot_risk_return_matrix(results))

    # Preparazione dati per GPT
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

# ------------------ Export risultati ------------------
if st.session_state.results:
    results = st.session_state.results
    st.subheader("💾 Esporta risultati")
    
    df_summary = pd.DataFrame([{k:v for k,v in r.items() if k!="npv_array"} for r in results])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet riepilogo
        df_summary.to_excel(writer, index=False, sheet_name='Risultati')
        # Sheet NPV per progetto
        for r in results:
            npv_df = pd.DataFrame(r["npv_array"], columns=["NPV"])
            sheet_name = r["name"][:31]  # Excel max 31 char
            npv_df.to_excel(writer, index=False, sheet_name=sheet_name)
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Scarica risultati in Excel",
        data=excel_data,
        file_name="capex_risultati.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )









