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

def plot_car_kri(car_value, expected_npv, project_name):
    """
    Gauge professionale: Capital at Risk (CaR) vs Expected NPV (%).
    """
    soglia_alta = 0.5
    soglia_media = 0.25

    car_pct = car_value / expected_npv if expected_npv != 0 else 1.0

    if car_pct > soglia_alta:
        risk_level = "Alto"
        color = "#ef4444"
    elif car_pct > soglia_media:
        risk_level = "Medio"
        color = "#facc15"
    else:
        risk_level = "Basso"
        color = "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=car_pct*100,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        delta={'reference': 100, 'increasing': {'color': 'red'}, 'position': "top"},
        title={'text': f"{project_name} - CaR ({risk_level})", 'font': {'size': 22}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': ["0%", "20%", "40%", "60%", "80%", "100%"]
            },
            'bar': {'color': "black", 'thickness': 0.05},
            'steps': [
                {'range': [0, soglia_media*100], 'color': "#d1fae5"},
                {'range': [soglia_media*100, soglia_alta*100], 'color': "#fef08a"},
                {'range': [soglia_alta*100, 100], 'color': "#fecaca"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 6},
                'thickness': 0.8,
                'value': car_pct*100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='white',
        font={'color': "darkblue", 'family': "Arial"},
        height=450,
    )

    return fig
















