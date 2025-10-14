import numpy as np
from .revenues import sample
from .costs import compute_costs

import numpy as np
from capex.wacc import calculate_wacc

def run_montecarlo(project, n_sim, wacc):
    """
    Esegue simulazioni Monte Carlo per un progetto con CAPEX ricorrente,
    ricavi e costi stocastici distribuiti anno per anno.
    
    project: dict con dati del progetto
    n_sim: numero simulazioni
    wacc: costo medio ponderato del capitale
    """
    
    years = project["years"]
    npv_array = np.zeros(n_sim)
    yearly_cashflows_list = []

    for sim in range(n_sim):
        cashflows = []

        for t in range(years):
            # CAPEX totale (iniziale + ricorrente)
            capex = 0
            if t == 0:
                capex += project["capex"]
            capex += sample(project["capex_rec"], year_idx=t)

            # Ricavi
            revenue = 0
            for rev in project["revenues_list"]:
                price = sample(rev["price"], year_idx=t) * (1 + project["price_growth"][t])
                quantity = sample(rev["quantity"], year_idx=t) * (1 + project["quantity_growth"][t])
                revenue += price * quantity

            # Costi variabili
            var_cost = revenue * project["costs"]["var_pct"]

            # Costi fissi
            fixed_cost = project["costs"]["fixed"] * (1 + project["fixed_cost_inflation"][t])

            # Costi aggiuntivi
            other_costs = 0
            for oc in project.get("other_costs", []):
                other_costs += sample(oc, year_idx=t)

            # Ammortamento
            depreciation = project["depreciation"][t]

            # Cashflow netto
            ebit = revenue - var_cost - fixed_cost - other_costs - depreciation
            tax = ebit * project["tax"] if ebit > 0 else 0
            net_cashflow = ebit - tax + depreciation - capex

            cashflows.append(net_cashflow)

        # Calcolo NPV
        discounted_cf = [cf / ((1 + wacc) ** (t+1)) for t, cf in enumerate(cashflows)]
        npv = sum(discounted_cf)
        npv_array[sim] = npv
        yearly_cashflows_list.append(cashflows)

    expected_npv = np.mean(npv_array)
    car = np.percentile(npv_array, 5)  # 5% worst-case
    downside_prob = np.mean(npv_array < 0)
    cvar = np.mean(npv_array[npv_array <= car]) if np.any(npv_array <= car) else car

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": yearly_cashflows_list,
        "expected_npv": expected_npv,
        "car": car,
        "downside_prob": downside_prob,
        "cvar": cvar
    }

