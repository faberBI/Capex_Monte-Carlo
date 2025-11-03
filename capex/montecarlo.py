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

    # Ricavi
    rev_min = df.get('Revenues min', pd.Series(0)).values
    rev_mode = df.get('Revenues piano', pd.Series(0)).values
    rev_max = df.get('Revenues max', pd.Series(0)).values
         
    # Costi variabili
    cs_min = df.get('Cost var min', pd.Series(0)).values
    cs_mode = df.get('Cost var piano', pd.Series(0)).values
    cs_max = df.get('Cost var max', pd.Series(0)).values
    
    # Costi fissi
    costs_fixed = df.get('Costs fixed', pd.Series(0)).values
    
    # Ammortamento
    amort = df.get('Amort. & Depreciation', pd.Series(0)).values
    
    # Capex
    capex = df.get('Capex', pd.Series(0)).values
    
    #Disposal
    disposal = df.get('Disposal & Capex Saving', pd.Series(0)).values
    
    #Change in working cap.
    change_wc = df.get('Change in working cap.', pd.Series(0)).values

    for i in range(n_sim):
        
        # Generazione valori stocastici triangolari
        rev_samp = triangular_sample(rev_min, rev_mode, rev_max, years)
                
        if cs_mode.sum() == 0:
            cs_samp = np.zeros(years)
        else:
            cs_samp = triangular_sample(cs_min, cs_mode, cs_max, years)

        # 1. EBITDA contribution
        revenue = rev_samp
        costi = (cs_samp + costs_fixed)
        
        ebitda = revenue + costi

        # 2. EBIT contribution
        ebit = ebitda + amort  # ammortamenti negativi

        taxes = -ebit * tax_rate

        # 4. FCF: se EBITDA == 0 allora FCF = |tasse|, altrimenti formula standard
        fcf = ebitda + taxes + capex + disposal + change_wc  # segni già corretti

        # 5. FCF present value
        discounts = (1 + discount_rate) ** np.arange(1, years + 1)
        fcf_pv = fcf / discounts

        # 6. NPV e cumulati
        npv = np.sum(fcf_pv)
        npv_cum = np.cumsum(fcf_pv)

        # Salvataggio risultati
        npv_list.append(npv)
        fcf_matrix[i, :] = fcf
        fcf_pv_matrix[i, :] = fcf_pv
        npv_cum_matrix[i, :] = npv_cum

    return np.array(npv_list), fcf_matrix, fcf_pv_matrix, npv_cum_matrix, years_col, costs_fixed, capex
















