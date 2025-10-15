import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# ------------------ Distribuzione NPV ------------------
def plot_npv_distribution(npv_array, expected_npv, percentile_5, name):
    npv_array = np.array(npv_array)
    fig, ax = plt.subplots()
    ax.hist(npv_array, bins=50, alpha=0.7, color="#3b82f6", edgecolor="black")
    ax.axvline(expected_npv, color="g", linestyle="--", label="Expected NPV")
    ax.axvline(percentile_5, color="r", linestyle="--", label="VaR 95%")
    ax.set_title(f"Distribuzione NPV - {name}")
    ax.set_xlabel("NPV")
    ax.set_ylabel("Frequenza")
    ax.legend()
    return fig

# ------------------ Boxplot NPV ------------------
def plot_boxplot(npv_array, name):
    npv_array = np.array(npv_array)
    fig, ax = plt.subplots()
    ax.boxplot(npv_array, patch_artist=True,
               boxprops=dict(facecolor="#3b82f6", color="black"),
               medianprops=dict(color="red"))
    ax.set_title(f"Boxplot NPV - {name}")
    ax.set_ylabel("NPV")
    return fig

# ------------------ Cashflows ------------------
def plot_cashflows(yearly_cash_flows, years, name):
    """
    yearly_cash_flows può essere:
    - 1D: valori medi per anno
    - 2D: simulazioni x anni
    Viene sempre plottata la media con banda di confidenza 5%-95%.
    """
    yearly_cash_flows = np.array(yearly_cash_flows)
    
    if yearly_cash_flows.ndim == 1:
        mean_cf = yearly_cash_flows
        low_cf = yearly_cash_flows
        high_cf = yearly_cash_flows
    else:
        mean_cf = np.mean(yearly_cash_flows, axis=0)
        low_cf = np.percentile(yearly_cash_flows, 5, axis=0)
        high_cf = np.percentile(yearly_cash_flows, 95, axis=0)
    
    fig, ax = plt.subplots()
    ax.bar(range(1, years+1), mean_cf, color="#3b82f6", alpha=0.7, label="Cash Flow Medio")
    ax.fill_between(range(1, years+1), low_cf, high_cf, color="#93c5fd", alpha=0.4, label="5%-95% intervallo")
    ax.set_xlabel("Anno")
    ax.set_ylabel("Cash Flow")
    ax.set_title(f"Cash Flow annuo medio - {name}")
    ax.legend()
    return fig

# ------------------ Matrice rischio-rendimento ------------------
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

# ------------------ Soglie rischio ------------------
def get_dynamic_thresholds(npv_array):
    npv_array = np.array(npv_array)
    p33 = np.percentile(npv_array, 33)
    p66 = np.percentile(npv_array, 66)
    return p33, p66

# ------------------ KRI Gauge ------------------
def plot_car_kri(car_value, expected_npv, project_name):
    """
    Gauge professionale: KRI = rischio di perdita rispetto all'NPV atteso (%).
    Utilizza la formula corretta: KRI = (Expected NPV - CaR) / Expected NPV
    """
    # Calcolo KRI corretto
    kri_pct = (expected_npv - car_value) / expected_npv if expected_npv != 0 else 1.0

    # Soglie rischio
    soglia_alta = 0.5
    soglia_media = 0.25

    if kri_pct > soglia_alta:
        risk_level = "Alto"
        color = "#ef4444"  # rosso
    elif kri_pct > soglia_media:
        risk_level = "Medio"
        color = "#facc15"  # giallo
    else:
        risk_level = "Basso"
        color = "#22c55e"  # verde

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=kri_pct*100,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        delta={'reference': 100, 'increasing': {'color': 'red'}, 'position': "top"},
        title={
            'text': f"KRI - {project_name}: {risk_level}",
            'font': {'size': 22, 'color': "darkblue"}
        },
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
                'value': kri_pct*100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='white',
        font={'color': "darkblue", 'family': "Arial"},
        height=450,
    )

    return fig

def plot_discounted_pbp(pbp_array, proj_name):
    """
    Grafico istogramma del Payback Period attualizzato per le simulazioni Monte Carlo.

    Args:
        pbp_array (np.array): array PBP per simulazione
        proj_name (str): nome progetto
    """
    plt.figure(figsize=(8,5))
    plt.hist(pbp_array[~np.isnan(pbp_array)], bins=range(1, int(np.nanmax(pbp_array))+2), 
             color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"📊 Distribuzione Payback Period attualizzato - {proj_name}")
    plt.xlabel("Payback Period (anni)")
    plt.ylabel("Frequenza")
    plt.grid(axis='y', alpha=0.3)
    plt.show()



