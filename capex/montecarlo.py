## buono

import numpy as np
from .costs import compute_costs
from capex.wacc import calculate_wacc
import pandas as pd

def run_montecarlo(proj, n_sim, wacc):
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_dcf = np.zeros((n_sim, years))
    pbp_array = np.zeros(n_sim)

    def calculate_fractional_pbp(discounted_cum_cf):
        negative_idx = np.where(discounted_cum_cf < 0)[0]
        if len(negative_idx) == 0:
            return 1.0
        last_neg_idx = negative_idx[-1]
        if last_neg_idx == len(discounted_cum_cf) - 1:
            return np.nan
        cf_before = discounted_cum_cf[last_neg_idx]
        cf_after = discounted_cum_cf[last_neg_idx + 1]
        return last_neg_idx + 1 + (-cf_before / (cf_after - cf_before))

    for sim in range(n_sim):
        dcf_per_year = []

        for year in range(years):
            total_revenue = 0.0
            for rev in proj["revenues_list"]:
                if not rev["price"][year]["is_stochastic"]:
                    # Deterministico: usa p1 * q1
                    price_val = rev["price"][year].get("p1", 0.0)
                    quantity_val = rev["quantity"][year].get("p1", 1.0)
                else:
                    # Stocastico: campiona
                    price_val = sample(rev["price"][year], year)
                    quantity_val = sample(rev["quantity"][year], year)
                revenue_year = price_val * quantity_val
                total_revenue += revenue_year

            # Costi
            fixed_cost = proj.get("fixed_costs", [0]*years)[year]
            var_cost = total_revenue * proj["costs"].get("var_pct", 0.0)
            other_costs_total = sum(sample(c.get("values", None), year) for c in proj.get("other_costs", []))

            # EBITDA / EBIT / Tasse
            ebitda = total_revenue - var_cost - fixed_cost - other_costs_total
            depreciation = proj.get("depreciation", [0]*years)[year]
            depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
            capex_rec = proj.get("capex_rec", [0]*years)[year]
            ammortamenti_tot = depreciation + depreciation_0
            ebit = ebitda - ammortamenti_tot
            
            # --- Tasse ---
            taxes = -ebit * proj["tax"]
            if ebit < 0:
                taxes = -taxes  # beneficio fiscale
            
            # --- FCF ---
            #capex_all = capex_init + capex_rec
            capex_all =  capex_rec
            
            if capex_all== 0 and ebitda<1:
                taxes = taxes*-1
                fcf = ebitda + taxes - capex_all
            else:
                # fcf = ebitda + taxes - capex_init - capex_rec
                fcf = ebitda + taxes - capex_rec

            # FCF scontati
            dcf = fcf / ((1 + wacc) ** (year + 1))
            dcf_per_year.append(dcf)

        dcf_per_year = np.array(dcf_per_year)
        yearly_dcf[sim, :] = dcf_per_year
        npv_array[sim] = dcf_per_year.sum()
        pbp_array[sim] = calculate_fractional_pbp(np.cumsum(dcf_per_year))

    # Percentili, CAR, CVaR
    percentiles = [5, 25, 50, 75, 95]
    yearly_dcf_percentiles = {f"p{p}": np.percentile(yearly_dcf, p, axis=0).tolist() for p in percentiles}
    npv_cum_matrix = np.cumsum(yearly_dcf, axis=1)
    yearly_npv_cum_percentiles = {f"p{p}": np.percentile(npv_cum_matrix, p, axis=0).tolist() for p in percentiles}
    pbp_percentiles = {f"p{p}": np.nanpercentile(pbp_array, p) for p in percentiles}
    car_5pct = np.percentile(npv_array, 5)
    cvar_5pct = np.mean(npv_array[npv_array <= car_5pct]) if np.any(npv_array <= car_5pct) else car_5pct
    avg_discounted_pbp = np.nanmean(pbp_array)

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



# ------------------ Calcolo dettagliato per anno ------------------
def calculate_yearly_financials(proj, wacc=0.0):
    years = proj["years"]
    revenues_total, ebitda_list, ebit_list, taxes_list, fcf_list = [], [], [], [], []

    for year in range(years):
        total_revenue = 0.0
        for rev in proj["revenues_list"]:
            price_obj = rev["price"][year]
            quantity_obj = rev["quantity"][year]

            if not price_obj.get("is_stochastic", False) and not quantity_obj.get("is_stochastic", False):
                revenue_year = price_obj.get("value", 0.0)
            else:
                price_val = sample(price_obj, year) if price_obj.get("is_stochastic", False) else price_obj.get("value", 0.0)
                quantity_val = sample(quantity_obj, year) if quantity_obj.get("is_stochastic", False) else quantity_obj.get("value", 0.0)
                revenue_year = price_val * quantity_val

            total_revenue += revenue_year

        revenues_total.append(total_revenue)

        # Costi
        fixed_cost = proj.get("fixed_costs", [0]*years)[year]
        var_cost = total_revenue * proj["costs"].get("var_pct", 0.0)
        other_costs_total = sum(sample(c.get("values", None), year) for c in proj.get("other_costs", []))

        ebitda = total_revenue - var_cost - fixed_cost - other_costs_total
        ebitda_list.append(ebitda)

        depreciation = proj.get("depreciation", [0]*years)[year]
        depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
        ammortamenti_tot = depreciation + depreciation_0

        ebit = ebitda - ammortamenti_tot
        ebit_list.append(ebit)

        taxes = -ebit * proj["tax"] if ebit >= 0 else -(-ebit * proj["tax"])
        taxes_list.append(taxes)

        capex_rec = proj.get("capex_rec", [0]*years)[year]
        fcf = ebitda + taxes - capex_rec
        fcf_list.append(fcf)

    df = pd.DataFrame({
        "Anno": list(range(1, years + 1)),
        "Ricavi": revenues_total,
        "EBITDA": ebitda_list,
        "EBIT": ebit_list,
        "Tasse": taxes_list,
        "FCF": fcf_list
    })

    npv_medio = sum(fcf / ((1 + wacc) ** (year + 1)) for year, fcf in enumerate(fcf_list))
    return df, npv_medio
    










