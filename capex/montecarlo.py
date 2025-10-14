import numpy as np
from .revenues import sample
from .costs import compute_costs
from capex.wacc import calculate_wacc

import numpy as np
from capex.wacc import calculate_wacc

def run_montecarlo(proj, n_sim, wacc):
    """
    Esegue simulazioni Monte Carlo per un progetto.

    proj: dizionario progetto
    n_sim: numero simulazioni
    wacc: tasso di sconto
    """
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_cashflows_all = np.zeros((n_sim, years))

    for sim in range(n_sim):
        yearly_cashflows = []
        for year in range(years):
            # --- CAPEX ---
            capex_initial = proj["capex"] if year == 0 else 0
            capex_rec = sample(proj["capex_rec"], year) if proj.get("capex_rec") else 0
            capex_total = capex_initial + capex_rec

            # --- Ricavi ---
            revenue_total = 0
            for rev in proj["revenues_list"]:
                price = sample(rev["price"], year)
                quantity = sample(rev["quantity"], year)
                revenue_total += price * quantity

            # --- Costi ---
            var_cost = proj["costs"]["var_pct"] * revenue_total
            fixed_cost = proj["costs"]["fixed"] * (1 + proj["fixed_cost_inflation"][year])
            other_cost_total = 0
            for cost in proj.get("other_costs", []):
                # usa .get("values") perché altri costi sono strutturati così
                other_cost_total += sample(cost["values"], year)

            depreciation = proj["depreciation"][year]

            # --- Cash flow ---
            ebit = revenue_total - var_cost - fixed_cost - other_cost_total - depreciation
            tax = proj["tax"] * ebit if ebit > 0 else 0
            net_cashflow = ebit - tax + depreciation - capex_total
            yearly_cashflows.append(net_cashflow)

        # --- NPV ---
        discounted_cf = [cf / ((1 + wacc) ** (y+1)) for y, cf in enumerate(yearly_cashflows)]
        npv = sum(discounted_cf)
        npv_array[sim] = npv
        yearly_cashflows_all[sim, :] = yearly_cashflows

    expected_npv = np.mean(npv_array)
    car = np.percentile(npv_array, 5)  # 5% percentile
    downside_prob = np.mean(npv_array < 0)
    cvar = np.mean(npv_array[npv_array <= car])

    return {
        "npv_array": npv_array,
        "expected_npv": expected_npv,
        "car": car,
        "cvar": cvar,
        "downside_prob": downside_prob,
        "yearly_cash_flows": np.mean(yearly_cash_flows_sim, axis=0)
    }




