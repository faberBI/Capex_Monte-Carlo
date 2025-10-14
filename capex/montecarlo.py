import numpy as np
from .revenues import sample
from .costs import compute_costs
from capex.wacc import calculate_wacc

import numpy as np
from capex.wacc import calculate_wacc

def run_montecarlo(proj, n_sim, wacc):
    """
    Esegue simulazioni Monte Carlo per un progetto CAPEX con ricavi, CAPEX ricorrente e costi aggiuntivi stocastici.
    
    Args:
        proj (dict): progetto con parametri, ricavi, costi e CAPEX ricorrente.
        n_sim (int): numero simulazioni.
        wacc (float): tasso di sconto WACC.
    
    Returns:
        dict: risultati simulazioni, tra cui NPV medio, CaR, CVaR, probabilità NPV<0 e cashflows annuali medi.
    """
    npv_array = np.zeros(n_sim)
    yearly_cash_flows_sim = []

    years = proj["years"]

    for sim in range(n_sim):
        yearly_cashflows = []

        for year in range(years):
            # CAPEX iniziale al primo anno
            capex = proj["capex"] if year == 0 else 0.0

            # CAPEX Ricorrente
            capex += sample(proj["capex_rec"], year)

            # Ricavi
            total_revenue = 0.0
            for rev in proj["revenues_list"]:
                price = sample(rev["price"], year)
                quantity = sample(rev["quantity"], year)
                total_revenue += price * quantity

            # Crescita prezzi e quantità
            total_revenue *= (1 + proj["price_growth"][year]) * (1 + proj["quantity_growth"][year])

            # Costi variabili e fissi
            var_cost = proj["costs"]["var_pct"] * total_revenue
            fixed_cost = proj["costs"]["fixed"] * (1 + proj["fixed_cost_inflation"][year])

            # Altri costi stocastici
            other_costs = 0.0
            for cost in proj.get("other_costs", []):
                other_costs += sample(cost, year)

            # Ammortamento
            depreciation = proj["depreciation"][year]

            # Cashflow operativo
            ebit = total_revenue - var_cost - fixed_cost - other_costs - depreciation
            tax = ebit * proj["tax"] if ebit > 0 else 0
            nopat = ebit - tax

            # Cashflow netto dell'anno
            cf_year = nopat + depreciation - capex
            yearly_cashflows.append(cf_year)

        # Calcolo NPV attualizzato
        discount_factors = np.array([(1 + wacc) ** year for year in range(years)])
        npv = sum([cf / df for cf, df in zip(yearly_cashflows, discount_factors)])
        npv_array[sim] = npv
        yearly_cash_flows_sim.append(yearly_cashflows)

    # Risultati
    expected_npv = np.mean(npv_array)
    car = expected_npv - np.percentile(npv_array, 5)  # CaR al 95%
    cvar = np.mean(npv_array[npv_array <= np.percentile(npv_array, 5)])  # Conditional VaR
    downside_prob = np.mean(npv_array < 0)  # Probabilità NPV < 0

    return {
        "npv_array": npv_array,
        "expected_npv": expected_npv,
        "car": car,
        "cvar": cvar,
        "downside_prob": downside_prob,
        "yearly_cash_flows": np.mean(yearly_cash_flows_sim, axis=0)
    }



