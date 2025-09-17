import matplotlib.pyplot as plt

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
    """
    p33 = np.percentile(npv_array, 33)
    p66 = np.percentile(npv_array, 66)
    return p33, p66

def plot_risk_gauge_dynamic(car_value, npv_array, project_name):
    """
    Gauge dinamico: colora il rischio in base al CaR confrontato con la distribuzione NPV.
    """
    p33, p66 = get_dynamic_thresholds(npv_array)

    if car_value <= p33:
        risk_level = "Alto"
        color = "red"
    elif car_value <= p66:
        risk_level = "Medio"
        color = "yellow"
    else:
        risk_level = "Basso"
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=car_value,
        title={'text': f"Rischio {project_name} ({risk_level})"},
        gauge={
            'axis': {'range': [np.min(npv_array), np.max(npv_array)]},
            'bar': {'color': color},
            'steps': [
                {'range': [np.min(npv_array), p33], 'color': "red"},
                {'range': [p33, p66], 'color': "yellow"},
                {'range': [p66, np.max(npv_array)], 'color': "green"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'value': car_value
            }
        }
    ))
    return fig


