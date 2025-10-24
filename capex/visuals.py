import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import numpy_financial as npf


# ------------------------- Funzioni di plotting -------------------------
def plot_npv_distribution(npv_array, expected_npv, percentile_5, name):
    fig, ax = plt.subplots()
    ax.hist(npv_array, bins=50, alpha=0.7, color="#3b82f6", edgecolor="black")
    ax.axvline(expected_npv, color="g", linestyle="--", label="Expected NPV")
    ax.axvline(percentile_5, color="r", linestyle="--", label="VaR 95%")
    ax.set_title(f"Distribuzione NPV - {name}")
    ax.set_xlabel("NPV")
    ax.set_ylabel("Frequenza")
    ax.legend()
    return fig

def plot_boxplot(npv_array, name):
    fig, ax = plt.subplots()
    ax.boxplot(npv_array, patch_artist=True,
               boxprops=dict(facecolor="#3b82f6", color="black"),
               medianprops=dict(color="red"))
    ax.set_title(f"Boxplot NPV - {name}")
    ax.set_ylabel("NPV")
    return fig

def plot_cashflows(yearly_cash_flows, years, name):
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

def plot_cumulative_npv(npv_cum_matrix, proj_name):
    median = np.median(npv_cum_matrix, axis=0)
    p5 = np.percentile(npv_cum_matrix, 5, axis=0)
    p95 = np.percentile(npv_cum_matrix, 95, axis=0)
    years = np.arange(1, npv_cum_matrix.shape[1]+1)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(years, median, label="Mediana NPV", color="blue", lw=2)
    ax.fill_between(years, p5, p95, color="blue", alpha=0.2, label="5°-95° percentile")
    try:
        pbp_median = np.argmax(median > 0) + 1
        ax.axvline(pbp_median, color="green", ls="--", lw=2, label=f"NPV positivo anno {pbp_median}")
    except:
        pass
    ax.set_title(f"📈 NPV cumulato per anno - {proj_name}")
    ax.set_xlabel("Anno")
    ax.set_ylabel("NPV cumulato")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig

def plot_payback_distribution(payback_array, name):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(payback_array[~np.isnan(payback_array)], bins=range(1, int(np.nanmax(payback_array))+2),
            color="#f97316", edgecolor='black', alpha=0.7)
    ax.set_xlabel("Anno payback")
    ax.set_ylabel("Frequenza")
    ax.set_title(f"Distribuzione payback period - {name}")
    return fig
    
    
def plot_probs_kri(downside_prob, project_name):
    """
    Gauge professionale: KRI basato sulla probabilità di NPV < 0.
    
    Soglie:
    - Verde: <5%
    - Giallo: <7%
    - Rosso: >7%
    """
    # Definizione soglie
    soglia_verde = 0.05
    soglia_gialla = 0.07

    # Determina livello di rischio e colore
    if downside_prob <=soglia_verde:
        risk_level = "Basso"
        color = "#22c55e"  # verde
    elif downside_prob <= soglia_gialla:
        risk_level = "Medio"
        color = "#facc15"  # giallo
    else:
        risk_level = "Alto"
        color = "#ef4444"  # rosso

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=downside_prob*100,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        title={
            'text': f"KRI - {project_name}: {risk_level}",
            'font': {'size': 22, 'color': "darkblue"}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 20, 40, 60, 80, 100]},
            'bar': {'color': "black", 'thickness': 0.05},
            'steps': [
                {'range': [0, soglia_verde*100], 'color': "#d1fae5"},
                {'range': [soglia_verde*100, soglia_gialla*100], 'color': "#fef08a"},
                {'range': [soglia_gialla*100, 100], 'color': "#fecaca"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 6},
                'thickness': 0.8,
                'value': downside_prob*100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='white',
        font={'color': "darkblue", 'family': "Arial"},
        height=450,
    )

    return fig
   
def plot_car_kri(car_value, expected_npv, project_name):
    """
    Gauge professionale: KRI = rischio di perdita rispetto all'NPV atteso (%).
    Utilizza la formula corretta: KRI = (Expected NPV - CaR) / Expected NPV
    """
    # Calcolo KRI corretto
    kri_pct = (expected_npv - car_value) / expected_npv if expected_npv != 0 else 1.0
    kri_pct = np.clip(kri_pct, 0, 1)

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

def plot_irr_trends(irr_min, irr_p5, irr_p50, irr_p95, irr_max, years_labels=None,
                    title="Andamento IRR per anno", figsize=(10,6)):
    """
    Grafico andamento IRR per anno con percentili, IRR in %, anni personalizzati.
    
    Parametri:
    - irr_min, irr_p5, irr_p50, irr_p95, irr_max: array dei percentili per anno
    - years_labels: lista o array degli anni reali (es. [2025, 2026, ...])
    - title: titolo del grafico
    - figsize: dimensione della figura
    
    Ritorna:
    - fig: figura matplotlib pronta da visualizzare
    """
    
    # Converti IRR in percentuale
    irr_min = np.array(irr_min) * 100
    irr_p5 = np.array(irr_p5) * 100
    irr_p50 = np.array(irr_p50) * 100
    irr_p95 = np.array(irr_p95) * 100
    irr_max = np.array(irr_max) * 100
    
    n_years = len(irr_min)
    
    if years_labels is None:
        years_labels = np.arange(1, n_years + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Area tra 5° e 95°
    ax.fill_between(years_labels, irr_p5, irr_p95, color='lightblue', alpha=0.5, label='5-95 percentile')
    
    # Linea mediana
    ax.plot(years_labels, irr_p50, color='blue', linewidth=2, label='Mediana (50° percentile)')
    
    # Linee estreme (min e max)
    ax.plot(years_labels, irr_min, '--', color='gray', linewidth=1, label='Min')
    ax.plot(years_labels, irr_max, '--', color='gray', linewidth=1, label='Max')
    
    ax.set_xlabel("Anno")
    ax.set_ylabel("IRR (%)")
    ax.set_title(title)
    ax.set_xticks(years_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def plot_ppi_distribution(ppi_min, ppi_p5, ppi_p50, ppi_p95, ppi_max, years_labels=None,
                    title="Andamento PPI per anno", figsize=(10,6)):
    """
    Grafico andamento PPI per anno con percentili, IRR in %, anni personalizzati.
    
    Parametri:
    - ppi_min, ppi_p5, ppi_p50, ppi_p95, ppi_max: array dei percentili per anno
    - years_labels: lista o array degli anni reali (es. [2025, 2026, ...])
    - title: titolo del grafico
    - figsize: dimensione della figura
    
    Ritorna:
    - fig: figura matplotlib pronta da visualizzare
    """
    
    # Converti IRR in percentuale
  
    n_years = len(ppi_min)
    
    if years_labels is None:
        years_labels = np.arange(1, n_years + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Area tra 5° e 95°
    ax.fill_between(years_labels, ppi_p5, ppi_p95, color='lightblue', alpha=0.5, label='5-95 percentile')
    
    # Linea mediana
    ax.plot(years_labels, ppi_p50, color='blue', linewidth=2, label='Mediana (50° percentile)')
    
    # Linee estreme (min e max)
    ax.plot(years_labels, ppi_min, '--', color='gray', linewidth=1, label='Min')
    ax.plot(years_labels, ppi_max, '--', color='gray', linewidth=1, label='Max')
    
    ax.set_xlabel("Anno")
    ax.set_ylabel("Profitability Index")
    ax.set_title(title)
    ax.set_xticks(years_labels)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig



