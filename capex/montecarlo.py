import numpy as np
from .revenues import sample
from .costs import compute_costs

def run_montecarlo(proj, n_sim, wacc):
    """
    Simulazione Monte Carlo per un progetto CAPEX con ricavi e costi stocastici,
    CAPEX ricorrente e ammortamento personalizzato.
    
    Args:
        proj (dict): Parametri del progetto.
        n_sim (int): Numero di simulazioni Monte Carlo.
        wacc (float): Tasso di sconto WACC.

    Returns:
        dict: Risultati della simulazione con NPV, CaR, CVaR, downside probability e flussi medi annuali.
    """
    npv_list = []
    yearly_cash_flows = np.zeros(proj["years"])

    # CAPEX iniziale
    capex_initial = proj["capex"]

    # Ammortamento personalizzato
    if "depreciation" not in proj or len(proj["depreciation"]) != proj["years"]:
        proj["depreciation"] = [capex_initial / proj["years"]] * proj["years"]

    # CAPEX ricorrente
    if isinstance(proj.get("capex_rec", 0), (int, float)):
        capex_rec = [proj.get("capex_rec", 0)] * proj["years"]
    else:
        capex_rec = proj.get("capex_rec", [0] * proj["years"])

    for _ in range(n_sim):
        cash_flows = []

        for t in range(1, proj["years"] + 1):
            # --- Ricavi ---
            price = sample(proj["revenues"]["price"]) * (1 + proj["price_growth"][t - 1])
            quantity = sample(proj["revenues"]["quantity"]) * (1 + proj["quantity_growth"][t - 1])
            revenue = price * quantity

            # --- Costi variabili/fissi ---
            fixed_sample = sample(proj.get(
                "costs_fixed",
                {"dist": "Normale", "p1": proj["costs"]["fixed"], "p2": 0}
            ))
            total_cost = compute_costs(
                revenue,
                proj["costs"]["var_pct"],
                fixed_sample,
                proj["fixed_cost_inflation"][t - 1]
            )

            # --- Costi aggiuntivi stocastici ---
            extra_costs = sum(sample(oc) for oc in proj.get("other_costs", []))

            # --- Ammortamento personalizzato ---
            depreciation = proj["depreciation"][t - 1]

            # --- EBIT ---
            ebit = revenue - total_cost - extra_costs - depreciation

            # --- Tasse ---
            taxes = max(0, ebit) * proj["tax"]

            # --- Free Cash Flow ---
            fcf = ebit - taxes + depreciation - capex_rec[t - 1]
            cash_flows.append(fcf)

        # Media dei flussi annuali sulle simulazioni
        yearly_cash_flows += np.array(cash_flows) / n_sim

        # NPV scontato
        discounted = [cf / ((1 + wacc) ** t) for t, cf in enumerate(cash_flows, start=1)]
        npv = sum(discounted) - capex_initial
        npv_list.append(npv)

    # --- Indicatori di rischio ---
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
