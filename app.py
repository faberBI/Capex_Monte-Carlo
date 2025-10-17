import streamlit as st
import numpy as np
import pandas as pd
import io
import time
from openai import OpenAI, OpenAIError

from capex.wacc import calculate_wacc
from capex.montecarlo import run_montecarlo, calculate_yearly_financials
from capex.visuals import (
    plot_npv_distribution,
    plot_boxplot,
    plot_cashflows,
    plot_risk_return_matrix,
    plot_car_kri,
    plot_cumulative_npv,
    plot_probs_kri
)

# ------------------ API Key OpenAI ------------------
api_key = st.secrets["OPENAI_API_KEY"]



# ------------------ Calcolo dettagliato per anno ------------------
def calculate_yearly_financials(proj, wacc=0.0):
    years = proj["years"]
    revenues_total, ebitda_list, ebit_list, taxes_list, fcf_list = [], [], [], [], []

    for year in range(years):
        total_revenue = 0.0
        for rev in proj["revenues_list"]:
            if not rev["price"][year]["is_stochastic"]:
                # Deterministico: usa p1 * q1
                price_val = rev["price"][year].get("p1", 0.0)
                quantity_val = rev["quantity"][year].get("p1", 1.0)
            else:
                # Stocastico: campiona
                price_val = sample(rev["price"][year], year)
                quantity_val = sample(rev["quantity"][year], year)
            revenue_year = price_val * quantity_val
            total_revenue += revenue_year

        revenues_total.append(total_revenue)

        # Costi
        fixed_cost = proj.get("fixed_costs", [0]*years)[year]
        var_cost = total_revenue * proj["costs"].get("var_pct", 0.0)
        other_costs_total = sum(sample(c.get("values", None), year) for c in proj.get("other_costs", []))

        ebitda = total_revenue - var_cost - fixed_cost - other_costs_total
        ebitda_list.append(ebitda)

        depreciation = proj.get("depreciation", [0]*years)[year]
        depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
        ammortamenti_tot = depreciation + depreciation_0

        ebit = ebitda - ammortamenti_tot
        ebit_list.append(ebit)

        taxes = -ebit * proj["tax"] if ebit >=0 else -(-ebit * proj["tax"])
        taxes_list.append(taxes)

        capex_rec = proj.get("capex_rec", [0]*years)[year]
        fcf = ebitda + taxes - capex_rec
        fcf_list.append(fcf)

    # Creazione DataFrame
    df = pd.DataFrame({
        "Anno": list(range(1, years+1)),
        "Ricavi": revenues_total,
        "EBITDA": ebitda_list,
        "EBIT": ebit_list,
        "Tasse": taxes_list,
        "FCF": fcf_list
    })

    # NPV medio
    npv_medio = sum(fcf / ((1 + wacc) ** (year+1)) for year, fcf in enumerate(fcf_list))

    return df, npv_medio





# ------------------ Funzione di campionamento ------------------
def sample(dist_obj, year_idx=None):
    """Campionamento stocastico per ricavi o other_costs."""
    if isinstance(dist_obj, list):
        if year_idx is None:
            raise ValueError("year_idx deve essere specificato per liste anno per anno")
        dist_obj = dist_obj[year_idx]

    dist_type = dist_obj.get("dist", "Normale")
    p1 = dist_obj.get("p1", 0.0) or 0.0
    p2 = dist_obj.get("p2", 0.0) or 0.0
    p3 = dist_obj.get("p3", p1 + p2) or (p1 + p2)

    if dist_type == "Normale":
        return np.random.normal(p1, max(p2, 1e-6))
    elif dist_type == "Triangolare":
        p2 = max(min(p2, p3), p1)
        return np.random.triangular(p1, p2, p3)
    elif dist_type == "Lognormale":
        return np.random.lognormal(p1, max(p2, 1e-6))
    elif dist_type == "Uniforme":
        if p2 < p1:
            p2 = p1
        return np.random.uniform(p1, p2)
    elif dist_type == "Deterministico":
        return dist_obj.get("value", p1)
    else:
        raise ValueError(f"Distribuzione non supportata: {dist_type}")


# ------------------ Session state ------------------
if "projects" not in st.session_state:
    st.session_state.projects = []
if "results" not in st.session_state:
    st.session_state.results = None

# ------------------ Aggiungi progetto ------------------
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
        "capex_rec": None,
        "fixed_costs": None,
        "revenues_list": [
            {
                "name": "Ricavo 1",
                "price": [{"dist": "Normale", "p1": 100.0, "p2": 10.0, "value": 100.0, "is_stochastic": True} for _ in range(10)],
                "quantity": [{"dist": "Normale", "p1": 1000.0, "p2": 100.0, "value": 1.0, "is_stochastic": True} for _ in range(10)]
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
n_sim = st.slider("Numero simulazioni Monte Carlo", 5000, 100_000, 10_000, step=5000)

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

        # ------------------ CAPEX Ricorrente ------------------
        st.subheader("🏗️ CAPEX Ricorrente (anno per anno)")
        if proj.get("capex_rec") is None:
            if st.button(f"➕ Aggiungi CAPEX Ricorrente", key=f"add_capex_rec_{i}"):
                proj["capex_rec"] = [0.0 for _ in range(proj["years"])]
        if proj.get("capex_rec") is not None:
            for y in range(proj["years"]):
                proj["capex_rec"][y] = st.number_input(f"CAPEX anno {y+1}", value=proj["capex_rec"][y], key=f"capex_rec_{i}_{y}")

        # ------------------ Ricavi ------------------
        st.subheader("📈 Ricavi")
        for j, rev in enumerate(proj["revenues_list"]):
            st.markdown(f"**{rev['name']}**")
            # Assicura che price e quantity abbiano la lunghezza corretta
            for key in ["price", "quantity"]:
                while len(rev[key]) < proj["years"]:
                    rev[key].append({"is_stochastic": True, "dist": "Normale", "p1": 0.0, "p2": 0.0})
        
            for y in range(proj["years"]):
                st.markdown(f"Anno {y+1} - {rev['name']}")
                is_stochastic = st.checkbox(
                    "Stocastico",
                    value=rev["price"][y]["is_stochastic"] or rev["quantity"][y]["is_stochastic"],
                    key=f"stochastic_{i}_{j}_{y}"
                )
                rev["price"][y]["is_stochastic"] = is_stochastic
                rev["quantity"][y]["is_stochastic"] = is_stochastic
            
                if is_stochastic:
                    rev["price"][y]["dist"] = st.selectbox(
                        "Distribuzione Price",
                        ["Normale", "Triangolare", "Lognormale", "Uniforme"],
                        index=["Normale", "Triangolare", "Lognormale", "Uniforme"].index(rev["price"][y].get("dist", "Normale")),
                        key=f"price_dist_{i}_{j}_{y}"
                    )
                    rev["price"][y]["p1"] = st.number_input("Price p1", value=rev["price"][y].get("p1",0.0), key=f"price_p1_{i}_{j}_{y}")
                    rev["price"][y]["p2"] = st.number_input("Price p2", value=rev["price"][y].get("p2",0.0), key=f"price_p2_{i}_{j}_{y}")
                    rev["quantity"][y]["dist"] = st.selectbox(
                        "Distribuzione Quantity",
                        ["Normale", "Triangolare", "Lognormale", "Uniforme"],
                        index=["Normale", "Triangolare", "Lognormale", "Uniforme"].index(rev["quantity"][y].get("dist", "Normale")),
                        key=f"quantity_dist_{i}_{j}_{y}"
                    )
                    rev["quantity"][y]["p1"] = st.number_input("Quantity p1", value=rev["quantity"][y].get("p1",0.0), key=f"quantity_p1_{i}_{j}_{y}")
                    rev["quantity"][y]["p2"] = st.number_input("Quantity p2", value=rev["quantity"][y].get("p2",0.0), key=f"quantity_p2_{i}_{j}_{y}")
                else:
                    # Deterministico: inserisci prezzo e quantità
                    rev["price"][y]["p1"] = st.number_input(f"Prezzo anno {y+1}", value=rev["price"][y].get("p1",0.0), key=f"price_det_{i}_{j}_{y}")
                    rev["quantity"][y]["p1"] = st.number_input(f"Quantità anno {y+1}", value=rev["quantity"][y].get("p1",1.0), key=f"quantity_det_{i}_{j}_{y}")


        # ------------------ Costi Variabili ------------------
        st.subheader("💸 Costi Variabili")
        proj["costs"]["var_pct"] = st.number_input("% Costi Variabili sui ricavi", value=proj["costs"]["var_pct"], min_value=0.0, max_value=1.0, step=0.01, key=f"var_pct_{i}")

        # ------------------ Costi Fissi ------------------
        st.subheader("💸 Costi Fissi annui")
        if proj.get("fixed_costs") is None or len(proj["fixed_costs"]) != proj["years"]:
            proj["fixed_costs"] = [0.0]*proj["years"]
        for y in range(proj["years"]):
            proj["fixed_costs"][y] = st.number_input(f"Costi fissi anno {y+1}", value=proj["fixed_costs"][y], key=f"fixed_{i}_{y}")

        # ------------------ Ammortamenti ------------------
        st.subheader("🏗️ Ammortamento")
        if "depreciation_0" not in proj:
            proj["depreciation_0"] = proj["capex"]/proj["years"]
        proj["depreciation_0"] = st.number_input(f"Ammortamento anno 0", value=proj["depreciation_0"], key=f"dep0_{i}")
        if "depreciation" not in proj or len(proj["depreciation"]) != proj["years"]:
            proj["depreciation"] = [proj["capex"]/proj["years"]]*proj["years"]
        df_dep = pd.DataFrame({"Anno": range(1, proj["years"]+1), "Ammortamento": proj["depreciation"]})
        df_dep_edit = st.data_editor(df_dep, key=f"dep_{i}", num_rows="dynamic")
        proj["depreciation"] = df_dep_edit["Ammortamento"].tolist()

        # ------------------ WACC ------------------
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

        st.pyplot(plot_npv_distribution(sim_result["npv_array"], sim_result["expected_npv"], np.percentile(sim_result["npv_array"], 5), proj["name"]))
        st.pyplot(plot_boxplot(sim_result["npv_array"], proj["name"]))
        st.pyplot(plot_cashflows(sim_result["yearly_cash_flows"], proj["years"], proj["name"]))
        st.plotly_chart(plot_car_kri(sim_result["car"], sim_result["expected_npv"], proj["name"]), use_container_width=True)

    st.session_state.results = results

# ------------------ Dettaglio finanziario per anno ------------------
if st.session_state.results:
    for r in st.session_state.results:
        proj = next(p for p in st.session_state.projects if p["name"] == r["name"])
        wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
        try:
            df_financials, npv_medio = calculate_yearly_financials(proj, wacc=wacc)
            st.subheader(f"📊 Dettaglio finanziario per anno - {proj['name']}")
            st.dataframe(df_financials.style.format("{:.2f}"))
        except Exception as e:
            st.error(f"Errore nel calcolo dei dettagli finanziari per {proj['name']}")
            st.write(str(e))




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

# ------------------ Export risultati completo ------------------
if st.session_state.results:
    results = st.session_state.results
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # --- Sheet riepilogo generale ---
        df_summary = pd.DataFrame([
            {
                "name": r["name"],
                "expected_npv": r["expected_npv"],
                "car": r["car"],
                "cvar": r["cvar"],
                "downside_prob": r["downside_prob"],
                "discounted_pbp": r["discounted_pbp"]
            } for r in results
        ])
        df_summary.to_excel(writer, index=False, sheet_name='Risultati')

        # --- Sheet per ciascun progetto ---
        for r in results:
            project_name = r['name'][:31]  # Excel max 31 char

            # 1️⃣ NPV simulazioni
            npv_df = pd.DataFrame(r["npv_array"], columns=["NPV"])
            npv_df.to_excel(writer, index=False, sheet_name=f"{project_name}_NPV")

            # 2️⃣ Cash flow annuali
            cf_df = pd.DataFrame(r["yearly_cash_flows"], columns=[f"Anno {i+1}" for i in range(r["yearly_cash_flows"].shape[1])])
            cf_df.to_excel(writer, index=False, sheet_name=f"{project_name}_CashFlow")

            # 3️⃣ Percentili cash flow annuali
            cf_pct_df = pd.DataFrame(r["yearly_cashflow_percentiles"])
            cf_pct_df.to_excel(writer, index=False, sheet_name=f"{project_name}_CF_Percentili")

            # 4️⃣ NPV cumulato
            npv_cum_df = pd.DataFrame(r["npv_cum_matrix"], columns=[f"Anno {i+1}" for i in range(r["npv_cum_matrix"].shape[1])])
            npv_cum_df.to_excel(writer, index=False, sheet_name=f"{project_name}_NPV_Cum")

            # 5️⃣ Percentili NPV cumulato
            npv_cum_pct_df = pd.DataFrame(r["yearly_npv_cum_percentiles"])
            npv_cum_pct_df.to_excel(writer, index=False, sheet_name=f"{project_name}_NPV_CumPct")

            # 6️⃣ Payback period simulazioni
            pbp_df = pd.DataFrame(r["pbp_array"], columns=["PBP"])
            pbp_df.to_excel(writer, index=False, sheet_name=f"{project_name}_PBP")

            # 7️⃣ Percentili Payback period
            pbp_pct_df = pd.DataFrame(r["pbp_percentiles"], index=[0])
            pbp_pct_df.to_excel(writer, index=False, sheet_name=f"{project_name}_PBP_Pct")

    excel_data = output.getvalue()

    st.download_button(
        label="📥 Scarica risultati completi in Excel",
        data=excel_data,
        file_name="capex_risultati_completi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

























































