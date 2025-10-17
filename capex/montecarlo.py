## buono

import numpy as np
from .costs import compute_costs
from capex.wacc import calculate_wacc
import pandas as pd


def run_montecarlo(proj, n_sim, wacc):
    """
    Simulazioni Monte Carlo per un progetto CAPEX.

    EBITDA = Ricavi - Costi
    EBIT = EBITDA - Ammortamenti
    Tasse = -(EBIT * tax) se EBIT>0, |EBIT*tax| se EBIT<0
    FCF = EBITDA + Tasse - CAPEX

    Args:
        proj (dict): progetto con info (capex, ricavi, costi, ammortamenti)
        n_sim (int): numero simulazioni
        wacc (float): tasso di sconto WACC

    Returns:
        dict: risultati simulazione
    """
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_dcf = np.zeros((n_sim, years))
    pbp_array = np.zeros(n_sim)

    def calculate_fractional_pbp(discounted_cum_cf):
        """Interpolazione lineare per PBP frazionario"""
        negative_idx = np.where(discounted_cum_cf < 0)[0]
        if len(negative_idx) == 0:
            return 1.0
        last_neg_idx = negative_idx[-1]
        if last_neg_idx == len(discounted_cum_cf) - 1:
            return np.nan
        cf_before = discounted_cum_cf[last_neg_idx]
        cf_after = discounted_cum_cf[last_neg_idx + 1]
        fraction = -cf_before / (cf_after - cf_before)
        return last_neg_idx + 1 + fraction

    for sim in range(n_sim):
        dcf_per_year = []

        for year in range(years):
            # CAPEX ricorrente
            capex_rec = proj.get("capex_rec", [0]*years)[year]

            # Costi fissi e ammortamenti
            fixed_cost = proj.get("fixed_costs", [0]*years)[year]
            depreciation = proj.get("depreciation", [0]*years)[year]
            depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
            ammortamenti_tot = depreciation + depreciation_0

            # --- Ricavi ---
            total_revenue = 0.0
            for rev in proj["revenues_list"]:
                # Price
                if rev["price"][year]["is_stochastic"]:
                    price_val = sample(rev["price"][year], year)
                else:
                    price_val = rev["price"][year].get("value", 0.0)

                # Quantity
                if rev["quantity"][year]["is_stochastic"]:
                    quantity_val = sample(rev["quantity"][year], year)
                else:
                    quantity_val = rev["quantity"][year].get("value", 1.0)

                # Deterministico totale
                if not rev["price"][year]["is_stochastic"] and not rev["quantity"][year]["is_stochastic"]:
                    revenue_year = price_val  # già totale
                else:
                    revenue_year = price_val * quantity_val

                total_revenue += revenue_year

            # Costi variabili e aggiuntivi
            var_cost = total_revenue * proj["costs"]["var_pct"]
            other_costs_total = sum(
                sample(cost.get("values", None), year) for cost in proj.get("other_costs", [])
            )

            # --- EBITDA ---
            ebitda = total_revenue - var_cost - fixed_cost - other_costs_total

            # --- EBIT ---
            ebit = ebitda - ammortamenti_tot

            # --- Tasse ---
            taxes = -ebit * proj["tax"]
            if ebit < 0:
                taxes = -taxes

            # --- FCF ---
            #capex_all = capex_init + capex_rec
            capex_all =  capex_rec
            
            if capex_all== 0 and ebitda<1:
                taxes = taxes*-1
                fcf = ebitda + taxes - capex_all
            else:
                # fcf = ebitda + taxes - capex_init - capex_rec
                fcf = ebitda + taxes - capex_rec

            # --- DCF attualizzato ---
            dcf = fcf / ((1 + wacc) ** (year + 1))
            dcf_per_year.append(dcf)

        dcf_per_year = np.array(dcf_per_year)
        yearly_dcf[sim, :] = dcf_per_year
        npv_array[sim] = dcf_per_year.sum()
        pbp_array[sim] = calculate_fractional_pbp(np.cumsum(dcf_per_year))

    avg_discounted_pbp = np.nanmean(pbp_array)

    # Percentili annuali DCF
    percentiles = [5, 25, 50, 75, 95]
    yearly_dcf_percentiles = {
        f"p{p}": np.percentile(yearly_dcf, p, axis=0).tolist() for p in percentiles
    }

    # Percentili cumulati NPV
    npv_cum_matrix = np.cumsum(yearly_dcf, axis=1)
    yearly_npv_cum_percentiles = {
        f"p{p}": np.percentile(npv_cum_matrix, p, axis=0).tolist() for p in percentiles
    }

    pbp_percentiles = {f"p{p}": np.nanpercentile(pbp_array, p) for p in percentiles}

    car_5pct = np.percentile(npv_array, 5)
    cvar_5pct = np.mean(npv_array[npv_array <= car_5pct]) if np.any(npv_array <= car_5pct) else car_5pct

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": yearly_dcf,
        "npv_cum_matrix": npv_cum_matrix,
        "expected_npv": np.mean(npv_array),
        "car": car_5pct,
        "cvar": cvar_5pct,
        "downside_prob": np.mean(npv_array < 0),
        "discounted_pbp": avg_discounted_pbp,
        "pbp_array": pbp_array,
        "yearly_cashflow_percentiles": yearly_dcf_percentiles,
        "yearly_npv_cum_percentiles": yearly_npv_cum_percentiles,
        "pbp_percentiles": pbp_percentiles
    }



# ------------------ Funzione sample per stocasticità ------------------
def sample(dist_obj, year_idx=None):
    """Campionamento stocastico per ricavi o other_costs."""
    if isinstance(dist_obj, list):
        if year_idx is None:
            raise ValueError("year_idx deve essere specificato per liste anno per anno")
        dist_obj = dist_obj[year_idx]

    dist_type = dist_obj.get("dist", "Normale")
    p1 = dist_obj.get("p1", 0.0) or 0.0
    p2 = dist_obj.get("p2", 0.0) or 0.0
    p3 = dist_obj.get("p3", p1 + p2) or (p1 + p2)

    if dist_type == "Normale":
        return np.random.normal(p1, max(p2, 1e-6))
    elif dist_type == "Triangolare":
        # Corregge p2 se fuori range
        p2 = max(min(p2, p3), p1)
        return np.random.triangular(p1, p2, p3)
    elif dist_type == "Lognormale":
        return np.random.lognormal(p1, max(p2, 1e-6))
    elif dist_type == "Uniforme":
        if p2 < p1:
            p2 = p1
        return np.random.uniform(p1, p2)
    elif dist_type == "Deterministico":
        # Per il nuovo toggle deterministico
        return dist_obj.get("value", p1)  # prende value se presente
    else:
        raise ValueError(f"Distribuzione non supportata: {dist_type}")


def calculate_yearly_financials(proj):
    import numpy as np
    import pandas as pd

    years = proj["years"]
    wacc = proj.get("wacc", 0.1)

    df = pd.DataFrame({"Anno": range(1, years + 1)})

    revenues_total = []
    var_costs_total = []
    fixed_costs_total = []
    depreciation_total = []
    ebitda_list = []
    ebit_list = []
    taxes_list = []
    fcf_list = []

    for year in range(years):
        total_revenue = 0.0
        for rev in proj.get("revenues_list", []):
            price_obj = rev["price"][year] if year < len(rev["price"]) else {"is_stochastic": True, "p1":0,"p2":0}
            quantity_obj = rev["quantity"][year] if year < len(rev["quantity"]) else {"is_stochastic": True, "p1":0,"p2":0}

            # Prezzo
            if price_obj.get("is_stochastic", True):
                price_val = sample(price_obj, year)
            else:
                price_val = price_obj.get("value", 0.0)

            # Quantità
            if quantity_obj.get("is_stochastic", True):
                quantity_val = sample(quantity_obj, year)
            else:
                quantity_val = quantity_obj.get("value", 0.0)

            total_revenue += price_val * quantity_val

        revenues_total.append(total_revenue)

        # Costi variabili
        var_pct = proj.get("costs", {}).get("var_pct", 0.0)
        var_cost = total_revenue * var_pct
        var_costs_total.append(var_cost)

        # Costi fissi
        fixed_cost = proj.get("fixed_costs", [0.0]*years)[year]
        fixed_costs_total.append(fixed_cost)

        # Ammortamenti
        depreciation = proj.get("depreciation", [0.0]*years)[year]
        depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
        ammortamenti_tot = depreciation + depreciation_0
        depreciation_total.append(ammortamenti_tot)

        # EBITDA
        ebitda = total_revenue - var_cost - fixed_cost
        ebitda_list.append(ebitda)

        # EBIT
        ebit = ebitda - ammortamenti_tot
        ebit_list.append(ebit)

        # Tasse
        taxes = -ebit * proj.get("tax", 0.3)
        if ebit < 0:
            taxes = -taxes
        taxes_list.append(taxes)

        # FCF
        capex_rec = proj.get("capex_rec", [0.0]*years)[year]
        fcf = ebitda + taxes - capex_rec
        fcf_list.append(fcf)

    df["Ricavi"] = revenues_total
    df["Costi variabili"] = var_costs_total
    df["Costi fissi"] = fixed_costs_total
    df["Ammortamenti"] = depreciation_total
    df["EBITDA"] = ebitda_list
    df["EBIT"] = ebit_list
    df["Tasse"] = taxes_list
    df["FCF"] = fcf_list

    # NPV medio
    fcf_array = np.array(fcf_list)
    npv_medio = np.sum(fcf_array / ((1 + wacc) ** np.arange(1, years + 1)))

    return df, npv_medio













