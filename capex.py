import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Helper per estrarre un campione ------------------
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

# ------------------ Setup session state ------------------
if "projects" not in st.session_state:
    st.session_state.projects = []

# ------------------ Funzione per aggiungere un nuovo progetto ------------------
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
        "revenues": {
            "price": {"dist": "Normale", "p1": 100.0, "p2": 10.0},
            "quantity": {"dist": "Normale", "p1": 1000.0, "p2": 100.0}
        },
        "costs": {
            "var_pct": 0.08,  # costi variabili % sui ricavi
            "fixed": -50.0    # costi fissi
        },
        "price_growth": [0.01]*10,        # trend prezzo anno per anno
        "quantity_growth": [0.05]*10,     # trend quantità anno per anno
        "fixed_cost_inflation": [0.02]*10 # trend costi fissi anno per anno
    })

# ------------------ Interfaccia principale ------------------
st.title("📊 CAPEX Risk Framework con WACC & Trend annuali")

st.button("➕ Aggiungi progetto", on_click=add_project)
n_sim = st.slider("Numero simulazioni Monte Carlo", 1000, 1000_000, 10_000)

results = []

# ------------------ Loop sui progetti ------------------
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

        # Calcolo WACC
        wacc = proj["equity"]*proj["ke"] + proj["debt"]*proj["kd"]*(1-proj["tax"])
        st.write(f"**WACC calcolato:** {wacc:.2%}")

        # ------------------ Parametri Ricavi ------------------
        st.subheader("📈 Ricavi")
        for key, label in [("price","Prezzo"),("quantity","Quantità")]:
            dist = st.selectbox(f"Distribuzione {label}", ["Normale","Triangolare","Lognormale","Uniforme"],
                                index=["Normale","Triangolare","Lognormale","Uniforme"].index(proj["revenues"][key]["dist"]),
                                key=f"{key}_dist_{i}")
            proj["revenues"][key]["dist"] = dist
            proj["revenues"][key]["p1"] = st.number_input(f"{label} - Param 1", value=proj["revenues"][key]["p1"], key=f"{key}_p1_{i}")
            proj["revenues"][key]["p2"] = st.number_input(f"{label} - Param 2", value=proj["revenues"][key]["p2"], key=f"{key}_p2_{i}")
            if dist == "Triangolare":
                proj["revenues"][key]["p3"] = st.number_input(f"{label} - Param 3 (max)", value=proj["revenues"][key].get("p3", proj["revenues"][key]["p1"]+proj["revenues"][key]["p2"]), key=f"{key}_p3_{i}")

        # ------------------ Parametri Costi ------------------
        st.subheader("💸 Costi")
        proj["costs"]["var_pct"] = st.number_input("% Costi Variabili sui ricavi", value=proj["costs"]["var_pct"], min_value=0.0, max_value=1.0, step=0.01, key=f"var_pct_{i}")
        proj["costs"]["fixed"] = st.number_input("Costi Fissi annui", value=proj["costs"]["fixed"], step=1.0, key=f"fixed_{i}")

        # ------------------ Trend annuali ------------------
        st.subheader("📊 Trend annuali (può essere negativo)")
        proj.setdefault("price_growth", [0.0]*proj["years"])
        proj.setdefault("quantity_growth", [0.0]*proj["years"])
        proj.setdefault("fixed_cost_inflation", [0.0]*proj["years"])
        for t in range(proj["years"]):
            proj["price_growth"][t] = st.number_input(f"Crescita prezzo anno {t+1}", value=proj["price_growth"][t], step=0.001, format="%.3f", key=f"pg_{i}_{t}")
            proj["quantity_growth"][t] = st.number_input(f"Crescita quantità anno {t+1}", value=proj["quantity_growth"][t], step=0.001, format="%.3f", key=f"qg_{i}_{t}")
            proj["fixed_cost_inflation"][t] = st.number_input(f"Crescita costi fissi anno {t+1}", value=proj["fixed_cost_inflation"][t], step=0.001, format="%.3f", key=f"fi_{i}_{t}")

    # ------------------ Simulazione Monte Carlo ------------------
    npv_list = []
    yearly_cash_flows = np.zeros(proj["years"])  # media dei cash flow annui

    for _ in range(n_sim):
        cash_flows = []
        for t in range(1, proj["years"]+1):
            # Ricavi con trend annuo
            price = sample(proj["revenues"]["price"]) * (1 + proj["price_growth"][t-1])
            quantity = sample(proj["revenues"]["quantity"]) * (1 + proj["quantity_growth"][t-1])
            revenue = price * quantity

            # Costi variabili e fissi con trend annuo
            var_cost_pct = proj["costs"]["var_pct"]
            fixed_cost = proj["costs"]["fixed"] * (1 + proj["fixed_cost_inflation"][t-1])

            cf = revenue - (revenue * var_cost_pct) - fixed_cost
            cash_flows.append(cf)

        yearly_cash_flows += np.array(cash_flows) / n_sim  # media annua
        discounted = [cf / ((1+wacc)**t) for t, cf in enumerate(cash_flows, start=1)]
        npv = sum(discounted) - proj["capex"]
        npv_list.append(npv)

    npv_array = np.array(npv_list)
    expected_npv = np.mean(npv_array)
    percentile_5 = np.percentile(npv_array, 5)
    car = expected_npv - percentile_5
    downside_prob = np.mean(npv_array < 0)
    cvar = np.mean(npv_array[npv_array <= percentile_5])

    results.append({
        "name": proj["name"],
        "expected_npv": expected_npv,
        "car": car,
        "downside_prob": downside_prob,
        "cvar": cvar
    })

    # ------------------ Grafici ------------------
    st.subheader(f"📊 Risultati {proj['name']}")
    st.write(f"Expected NPV: {expected_npv:.2f}")
    st.write(f"CaR (95%): {car:.2f}")
    st.write(f"Probabilità NPV < 0: {downside_prob*100:.1f}%")
    st.write(f"Conditional VaR (95%): {cvar:.2f}")

    # Distribuzione NPV
    fig, ax = plt.subplots()
    ax.hist(npv_array, bins=50, alpha=0.7)
    ax.axvline(expected_npv, color="g", linestyle="--", label="Expected NPV")
    ax.axvline(percentile_5, color="r", linestyle="--", label="VaR 95%")
    ax.set_title(f"Distribuzione NPV - {proj['name']}")
    ax.legend()
    st.pyplot(fig)

    # Boxplot NPV
    fig2, ax2 = plt.subplots()
    ax2.boxplot(npv_array)
    ax2.set_title(f"Boxplot NPV - {proj['name']}")
    st.pyplot(fig2)

    # Cash flow annuo medio
    fig3, ax3 = plt.subplots()
    ax3.bar(range(1, proj["years"]+1), yearly_cash_flows)
    ax3.set_xlabel("Anno")
    ax3.set_ylabel("Cash Flow Medio")
    ax3.set_title(f"Cash Flow annuo medio - {proj['name']}")
    st.pyplot(fig3)

# ------------------ Matrice rischio-rendimento ------------------
if results:
    st.subheader("📌 Matrice rischio-rendimento")
    fig, ax = plt.subplots()
    for r in results:
        ax.scatter(r["expected_npv"], r["car"], s=100, label=r["name"])
        ax.text(r["expected_npv"], r["car"], r["name"])
    ax.set_xlabel("Expected NPV")
    ax.set_ylabel("CaR (95%)")
    ax.set_title("Matrice rischio-rendimento")
    ax.legend()
    st.pyplot(fig)
