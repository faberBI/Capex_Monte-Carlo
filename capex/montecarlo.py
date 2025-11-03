import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import numpy_financial as npf


def triangular_sample(min_v, mode_v, max_v, size):
    min_v = np.where(np.isnan(min_v), mode_v, min_v)
    max_v = np.where(np.isnan(max_v), mode_v, max_v)
    mode_v = np.clip(mode_v, min_v, max_v)
    min_v = np.minimum(min_v, mode_v)
    max_v = np.maximum(max_v, mode_v)
    
    # gestisci min==max
    constant_mask = min_v == max_v
    result = np.random.triangular(min_v, mode_v, max_v, size)
    if np.any(constant_mask):
        result[constant_mask] = min_v[constant_mask]
    return result

def run_simulations(df, n_sim, discount_rate, tax_rate):
    years = df.shape[0]
    npv_list = []
    fcf_matrix = np.zeros((n_sim, years))
    fcf_pv_matrix = np.zeros((n_sim, years))
    npv_cum_matrix = np.zeros((n_sim, years))
    years_col = df.iloc[:, 0].values

    # Estraggo colonne
    rev_min = df.get('Revenues min', pd.Series(0)).values
    rev_mode = df.get('Revenues piano', pd.Series(0)).values
    rev_max = df.get('Revenues max', pd.Series(0)).values
    cs_min = df.get('Cost var min', pd.Series(0)).values
    cs_mode = df.get('Cost var piano', pd.Series(0)).values
    cs_max = df.get('Cost var max', pd.Series(0)).values
    costs_fixed = df.get('Costs fixed', pd.Series(0)).values
    amort = df.get('Amort. & Depreciation', pd.Series(0)).values
    capex = df.get('Capex', pd.Series(0)).values
    disposal = df.get('Disposal & Capex Saving', pd.Series(0)).values
    change_wc = df.get('Change in working cap.', pd.Series(0)).values

    for i in range(n_sim):
        fcf = np.zeros(years)

        for y in range(years):
            # Ricavi
            if rev_mode[y] == 0:
                revenue = 0
            else:
                revenue = np.random.triangular(rev_min[y], rev_mode[y], rev_max[y])

            # Costi variabili
            if cs_mode[y] == 0:
                cs = 0
            else:
                cs = np.random.triangular(cs_min[y], cs_mode[y], cs_max[y])

            # EBITDA
            ebitda = revenue + cs + costs_fixed[y]

            # EBIT
            ebit = ebitda + amort[y]

            # Tasse
            taxes = -ebit * tax_rate

            # FCF per anno
            fcf[y] = ebitda + taxes + capex[y] + disposal[y] + change_wc[y]

        # Sconto DCF
        discounts = (1 + discount_rate) ** np.arange(1, years + 1)
        fcf_pv = fcf / discounts

        # NPV e cumulati
        npv = np.sum(fcf_pv)
        npv_cum = np.cumsum(fcf_pv)

        # Salvataggio
        npv_list.append(npv)
        fcf_matrix[i, :] = fcf
        fcf_pv_matrix[i, :] = fcf_pv
        npv_cum_matrix[i, :] = npv_cum

    return np.array(npv_list), fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col, costs_fixed, capex




















