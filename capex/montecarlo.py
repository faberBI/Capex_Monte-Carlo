import numpy as np
from .revenues import sample
from .costs import compute_costs

def run_montecarlo(proj, n_sim, wacc):
    npv_list = []
    yearly_cash_flows = np.zeros(proj["years"])

    capex_initial = proj["capex"]
    if "depreciation" not in proj or len(proj["depreciation"]) != proj["years"]:
        proj["depreciation"] = [capex_initial / proj["years"]] * proj["years"]
    capex_rec = proj.get("capex_rec", [0]*proj["years"])

    for _ in range(n_sim):
        cash_flows = []

        for t in range(proj["years"]):
            # --- Ricavi multipli ---
            revenue = 0
            for rev in proj.get("revenues_list", []):
                price = sample(rev["price"]) * (1 + proj["price_growth"][t])
                quantity = sample(rev["quantity"]) * (1 + proj["quantity_growth"][t])
                revenue += price * quantity

            # --- Costi variabili/fissi ---
            fixed_sample = proj["costs"]["fixed"]
            total_cost = compute_costs(
                revenue,
                proj["costs"]["var_pct"],
                fixed_sample,
                proj["fixed_cost_inflation"][t]
            )

            # --- Costi aggiuntivi stocastici ---
            extra_costs = sum(sample(oc) for oc in proj.get("other_costs", []))

            # --- Ammortamento ---
            depreciation = proj["depreciation"][t]

            # --- EBIT ---
            ebit = revenue - total_cost - extra_costs - depreciation

            # --- Tasse ---
            taxes = max(0, ebit) * proj["tax"]

            # --- Free Cash Flow ---
            fcf = ebit - taxes + depreciation - capex_rec[t]
            cash_flows.append(fcf)

        yearly_cash_flows += np.array(cash_flows) / n_sim
        discounted = [cf / ((1 + wacc) ** (t+1)) for t, cf in enumerate(cash_flows)]
        npv = sum(discounted) - capex_initial
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
