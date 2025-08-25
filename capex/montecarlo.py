import numpy as np
from .revenues import sample
from .costs import compute_costs

def run_montecarlo(proj, n_sim, wacc):
    npv_list = []
    yearly_cash_flows = np.zeros(proj["years"])

    for _ in range(n_sim):
        cash_flows = []
        for t in range(1, proj["years"]+1):
            price = sample(proj["revenues"]["price"]) * (1 + proj["price_growth"][t-1])
            quantity = sample(proj["revenues"]["quantity"]) * (1 + proj["quantity_growth"][t-1])
            revenue = price * quantity

            total_cost = compute_costs(revenue, proj["costs"]["var_pct"], proj["costs"]["fixed"], proj["fixed_cost_inflation"][t-1])
            cf = revenue - total_cost
            cash_flows.append(cf)

        yearly_cash_flows += np.array(cash_flows) / n_sim
        discounted = [cf / ((1+wacc)**t) for t, cf in enumerate(cash_flows, start=1)]
        npv = sum(discounted) - proj["capex"]
        npv_list.append(npv)

    npv_array = np.array(npv_list)
    expected_npv = np.mean(npv_array)
    percentile_5 = np.percentile(npv_array, 5)
    car = expected_npv - percentile_5
    downside_prob = np.mean(npv_array < 0)
    cvar = np.mean(npv_array[npv_array <= percentile_5])

    return {
        "npv_array": npv_array,
        "expected_npv": expected_npv,
        "car": car,
        "downside_prob": downside_prob,
        "cvar": cvar,
        "yearly_cash_flows": yearly_cash_flows
    }
