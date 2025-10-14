import numpy as np
from .revenues import sample
from .costs import compute_costs
from capex.wacc import calculate_wacc

from .app import sample 

def run_montecarlo(proj, n_sim, wacc):
    """
    Esegue simulazioni Monte Carlo per un progetto.
    
    proj: dizionario con parametri del progetto
    n_sim: numero di simulazioni
    wacc: costo medio ponderato del capitale
    """
    years = proj["years"]
    npv_array = []
    yearly_cashflows_matrix = []

    for sim in range(n_sim):
        yearly_cashflows = []

        for year in range(years):
            # ----------------- CAPEX -----------------
            capex_init = proj.get("capex", 0.0) if year == 0 else 0.0
            capex_rec = sample(proj.get("capex_rec"), year) if proj.get("capex_rec") else 0.0
            total_capex = capex_init + capex_rec

            # ----------------- Ricavi -----------------
            total_revenue = 0.0
            for rev in proj.get("revenues_list", []):
                price = sample(rev.get("price"), year)
                quantity = sample(rev.get("quantity"), year)
                total_revenue += price * quantity

            # ----------------- Costi -----------------
            var_pct = proj.get("costs", {}).get("var_pct", 0.0)
            fixed_cost = proj.get("costs", {}).get("fixed", 0.0)
            total_costs = fixed_cost + total_revenue * var_pct

            # Altri costi stocastici
            for cost in proj.get("other_costs", []):
                cost_val = sample(cost.get("values"), year)
                total_costs += cost_val

            # ----------------- Ammortamento -----------------
            depreciation = proj.get("depreciation", [0.0]*years)
            depreciation_val = depreciation[year]

            # ----------------- Cash flow -----------------
            cf_before_tax = total_revenue - total_costs - total_capex
            tax = proj.get("tax", 0.0)
            cf_after_tax = cf_before_tax * (1 - tax) + depreciation_val  # aggiungo ammortamento back

            yearly_cashflows.append(cf_after_tax)

        # ----------------- NPV -----------------
        npv = sum([cf / (1 + wacc)**(t+1) for t, cf in enumerate(yearly_cashflows)])
        npv_array.append(npv)
        yearly_cashflows_matrix.append(yearly_cashflows)

    npv_array = np.array(npv_array)
    expected_npv = np.mean(npv_array)
    car = np.percentile(npv_array, 5)
    downside_prob = np.mean(npv_array < 0)
    cvar = np.mean(npv_array[npv_array <= car])

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": np.array(yearly_cashflows_matrix),
        "expected_npv": expected_npv,
        "car": car,
        "downside_prob": downside_prob,
        "cvar": cvar
    }






