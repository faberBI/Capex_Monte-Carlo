import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_npv_distribution(npv_array, expected_npv, percentile_5, name):
    fig, ax = plt.subplots()
    ax.hist(npv_array, bins=50, alpha=0.7)
    ax.axvline(expected_npv, color="g", linestyle="--", label="Expected NPV")
    ax.axvline(percentile_5, color="r", linestyle="--", label="VaR 95%")
    ax.set_title(f"Distribuzione NPV - {name}")
    ax.legend()
    return fig

def plot_boxplot(npv_array, name):
    fig, ax = plt.subplots()
    ax.boxplot(npv_array)
    ax.set_title(f"Boxplot NPV - {name}")
    return fig

def plot_cashflows(yearly_cash_flows, years, name):
    fig, ax = plt.subplots()
    ax.bar(range(1, years+1), yearly_cash_flows)
    ax.set_xlabel("Anno")
    ax.set_ylabel("Cash Flow Medio")
    ax.set_title(f"Cash Flow annuo medio - {name}")
    return fig

def plot_risk_return_matrix(results):
    fig, ax = plt.subplots()
    for r in results:
        ax.scatter(r["expected_npv"], r["car"], s=100, label=r["name"])
        ax.text(r["expected_npv"], r["car"], r["name"])
    ax.set_xlabel("Expected NPV")
    ax.set_ylabel("CaR (95%)")
    ax.set_title("Matrice rischio-rendimento")
    ax.legend()
    return fig

def get_dynamic_thresholds(npv_array):
    """
    Calcola le soglie per rischio (basso, medio, alto) dai percentili del NPV simulato.
    Converte npv_array in NumPy array se necessario.
    """
    npv_array = np.array(npv_array)  # ✅ Assicurati che sia NumPy array
    p33 = np.percentile(npv_array, 33)
    p66 = np.percentile(npv_array, 66)
    return p33, p66


def plot_car_kri(car_value, cap_invested, project_name):
    """
    Gauge dinamico per KRI basato sul CaR rispetto al capitale investito.
    """
    # -----------------------------
    # Definizione soglie in % del capitale investito
    # -----------------------------
    soglia_alta = 0.5 * cap_invested     # >50% capitale a rischio = alto
    soglia_media = 0.25 * cap_invested   # 25-50% = medio
    # <25% = basso

    # -----------------------------
    # Determina livello rischio e colore
    # -----------------------------
    if car_value > soglia_alta:
        risk_level = "Alto"
        color = "red"
    elif car_value > soglia_media:
        risk_level = "Medio"
        color = "yellow"
    else:
        risk_level = "Basso"
        color = "green"

    # -----------------------------
    # Creo gauge
    # -----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=car_value,
        number={'prefix': "$", 'font': {'color': color, 'size': 28}},
        delta={'reference': cap_invested, 'increasing': {'color': 'red'}, 'position': "top"},
        title={'text': f"KRI - {project_name} ({risk_level})", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, cap_invested], 'tickprefix': "$", 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},  # barra che indica il CaR
            'steps': [
                {'range': [0, soglia_media], 'color': "green"},
                {'range': [soglia_media, soglia_alta], 'color': "yellow"},
                {'range': [soglia_alta, cap_invested], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': car_value
            }
        }
    ))

    fig.update_layout(paper_bgcolor='white', font={'color': "darkblue", 'family': "Arial"})
    
    return fig





