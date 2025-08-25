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
