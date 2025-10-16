import numpy as np
from .costs import compute_costs
from capex.wacc import calculate_wacc


import numpy as np

def run_montecarlo(proj, n_sim, wacc):
    """
    Esegue simulazioni Monte Carlo per un progetto CAPEX.
    
    Args:
        proj (dict): progetto con tutte le informazioni (capex, revenues, costi, ammortamento, ecc.)
        n_sim (int): numero di simulazioni
        wacc (float): tasso di sconto WACC

    Returns:
        dict: risultati della simulazione con npv_array, yearly_cash_flows, percentili, npv cumulato e PBP
    """
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
            depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0

            # Ricavi totali stocastici
            total_revenue = sum(
                sample(rev["price"], year) * sample(rev["quantity"], year)
                for rev in proj["revenues_list"]
            )

            # Costi variabili e aggiuntivi
            var_cost = total_revenue * proj["costs"]["var_pct"]
            other_costs_total = sum(sample(cost.get("values", None), year) for cost in proj.get("other_costs", []))

            # Cash flow anno corrente
            cf = total_revenue - var_cost - fixed_cost - other_costs_total - capex_init - capex_rec - depreciation - depreciation_0
            cash_flows.append(cf)

            # Payback period attualizzato
            discounted_cf_cum += cf / ((1 + wacc) ** (year + 1))
            if not pbp_found and discounted_cf_cum >= 0:
                pbp_array[sim] = year + 1
                pbp_found = True

        if not pbp_found:
            pbp_array[sim] = np.nan

        # NPV totale
        discounted_cf = [cf / ((1 + wacc) ** (year + 1)) for year, cf in enumerate(cash_flows)]
        npv_array[sim] = sum(discounted_cf)
        yearly_cash_flows[sim, :] = cash_flows

    avg_discounted_pbp = np.nanmean(pbp_array)

    # --- Percentili annuali cash flow ---
    percentiles = [5, 25, 50, 75, 95]
    yearly_cashflow_percentiles = {
        f"p{p}": np.percentile(yearly_cash_flows, p, axis=0).tolist()
        for p in percentiles
    }

    # --- NPV cumulato e percentili ---
    npv_cum_matrix = np.cumsum(yearly_cash_flows, axis=1)
    yearly_npv_cum_percentiles = {
        f"p{p}": np.percentile(npv_cum_matrix, p, axis=0).tolist()
        for p in percentiles
    }

    # --- Percentili Payback ---
    pbp_percentiles = {
        f"p{p}": np.nanpercentile(pbp_array, p) for p in percentiles
    }

    # --- Calcolo CVaR ---
    car_5pct = np.percentile(npv_array, 5)
    cvar_5pct = np.mean(npv_array[npv_array <= car_5pct]) if np.any(npv_array <= car_5pct) else car_5pct

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": yearly_cash_flows,
        "npv_cum_matrix": npv_cum_matrix,
        "expected_npv": np.mean(npv_array),
        "car": car_5pct,
        "cvar": cvar_5pct,
        "downside_prob": np.mean(npv_array < 0),
        "discounted_pbp": avg_discounted_pbp,
        "pbp_array": pbp_array,
        "yearly_cashflow_percentiles": yearly_cashflow_percentiles,
        "yearly_npv_cum_percentiles": yearly_npv_cum_percentiles,
        "pbp_percentiles": pbp_percentiles
    }


# ------------------ Funzione sample per stocasticità ------------------
def sample(dist_obj, year_idx=None):
    """Campionamento stocastico solo per other_costs o ricavi, non per CAPEX o costi fissi."""
    if isinstance(dist_obj, list):
        if year_idx is None:
            raise ValueError("year_idx deve essere specificato per liste di distribuzioni anno per anno")
        dist_obj = dist_obj[year_idx]

    dist_type = dist_obj.get("dist", "Normale")
    p1 = dist_obj.get("p1", 0.0)
    p2 = dist_obj.get("p2", 0.0)
    p3 = dist_obj.get("p3", p1+p2)

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










