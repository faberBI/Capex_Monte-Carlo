import numpy as np
from .costs import compute_costs
from capex.wacc import calculate_wacc


# ------------------ Helper per sample distribuzioni ------------------
def sample(dist_obj, year_idx=None):
    """
    Campiona un valore da una distribuzione.
    
    dist_obj può essere:
    - un dizionario singolo: {"dist": "Normale", "p1": ..., "p2": ..., "p3": ...}
    - una lista di dizionari per anno, in tal caso serve year_idx
    
    year_idx: indice dell'anno (0-based) se dist_obj è lista
    """
    # Se è una lista di distribuzioni per anno, seleziona quella giusta
    if isinstance(dist_obj, list):
        if year_idx is None:
            raise ValueError("year_idx deve essere specificato per liste di distribuzioni anno per anno")
        dist_obj = dist_obj[year_idx]

    # Parametri della distribuzione
    dist_type = dist_obj.get("dist", "Normale")
    p1 = dist_obj.get("p1", 0.0)
    p2 = dist_obj.get("p2", 0.0)
    p3 = dist_obj.get("p3", p1 + p2)  # solo per triangolare

    # Campionamento
    if dist_type == "Normale":
        return np.random.normal(p1, p2)
    elif dist_type == "Triangolare":
        return np.random.triangular(p1, p2, p3)
    elif dist_type == "Lognormale":
        return np.random.lognormal(p1, p2)
    elif dist_type == "Uniforme":
        return np.random.uniform(p1, p2)
    else:
        raise ValueError(f"Distribuzione non supportata: {dist_type}")

import numpy as np

def run_montecarlo(proj, n_sim, wacc):
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_cash_flows = np.zeros((n_sim, years))
    pbp_array = np.zeros(n_sim)

    for sim in range(n_sim):
        cash_flows = []
        discounted_cf_cum = 0
        pbp_found = False

        for year in range(years):
            capex_init = proj["capex"] if year == 0 else 0
            capex_rec_list = proj.get("capex_rec") or [0] * years
            capex_rec = capex_rec_list[year]
            fixed_costs_list = proj.get("fixed_costs") or [0] * years
            fixed_cost = fixed_costs_list[year]
            depreciation_list = proj.get("depreciation") or [0] * years
            depreciation = depreciation_list[year]

            total_revenue = sum(
                sample(rev["price"], year) * sample(rev["quantity"], year)
                for rev in proj["revenues_list"]
            )

            var_cost = total_revenue * proj["costs"]["var_pct"]
            other_costs_total = sum(sample(cost.get("values", None), year) for cost in proj.get("other_costs", []))
            cf = total_revenue - var_cost - fixed_cost - other_costs_total - capex_init - capex_rec
            cash_flows.append(cf)

            discounted_cf_cum += cf / ((1 + wacc) ** (year + 1))
            if not pbp_found and discounted_cf_cum >= 0:
                pbp_array[sim] = year + 1
                pbp_found = True

        if not pbp_found:
            pbp_array[sim] = np.nan

        discounted_cf = [cf / ((1 + wacc) ** (year + 1)) for year, cf in enumerate(cash_flows)]
        npv_array[sim] = sum(discounted_cf)
        yearly_cash_flows[sim, :] = cash_flows

    avg_discounted_pbp = np.nanmean(pbp_array)

    # --- Percentili annuali ---
    percentiles = [5, 25, 50, 75, 95]
    yearly_percentiles = {
        f"p{p}": np.percentile(yearly_cash_flows, p, axis=0).tolist()
        for p in percentiles
    }

    # --- Percentili del Payback ---
    pbp_percentiles = {
        f"p{p}": np.nanpercentile(pbp_array, p) for p in percentiles
    }

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": yearly_cash_flows,
        "expected_npv": np.mean(npv_array),
        "car": np.percentile(npv_array, 5),
        "cvar": np.mean(npv_array[npv_array <= np.percentile(npv_array, 5)])
                 if np.any(npv_array <= np.percentile(npv_array, 5)) else np.percentile(npv_array, 5),
        "downside_prob": np.mean(npv_array < 0),
        "discounted_pbp": avg_discounted_pbp,
        "pbp_array": pbp_array,
        "yearly_percentiles": yearly_percentiles,  # ✅ aggiunto
        "pbp_percentiles": pbp_percentiles         # ✅ aggiunto
    }













