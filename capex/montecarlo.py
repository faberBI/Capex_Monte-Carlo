import numpy as np
from .costs import compute_costs
from capex.wacc import calculate_wacc
import pandas as pd

def run_montecarlo(proj, n_sim, wacc):
    """
    Simulazioni Monte Carlo per un progetto CAPEX con logica:
    EBITDA = Ricavi - Costi
    EBIT = EBITDA - Ammortamenti
    Tasse = -(EBIT * tax) se EBIT>0, |EBIT*tax| se EBIT<0
    FCF = EBITDA + Tasse - CAPEX

    Args:
        proj (dict): progetto con info (capex, ricavi, costi, ammortamenti)
        n_sim (int): numero simulazioni
        wacc (float): tasso di sconto WACC

    Returns:
        dict: risultati simulazione con npv_array, yearly_cash_flows, percentili, npv cumulato e PBP
    """
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_cash_flows = np.zeros((n_sim, years))
    pbp_array = np.zeros(n_sim)

    def calculate_fractional_pbp(discounted_cum_cf):
        """Interpolazione lineare per PBP frazionario"""
        negative_idx = np.where(discounted_cum_cf < 0)[0]
        if len(negative_idx) == 0:
            return 1.0  # PBP < primo anno
        last_neg_idx = negative_idx[-1]
        if last_neg_idx == len(discounted_cum_cf) - 1:
            return np.nan  # mai positivo
        cf_before = discounted_cum_cf[last_neg_idx]
        cf_after = discounted_cum_cf[last_neg_idx + 1]
        fraction = -cf_before / (cf_after - cf_before)
        return last_neg_idx + 1 + fraction

    for sim in range(n_sim):
        cash_flows = []
        discounted_cf_cum = 0

        for year in range(years):
            # CAPEX
            capex_init = proj["capex"] if year == 0 else 0
            capex_rec_list = proj.get("capex_rec") or [0] * years
            capex_rec = capex_rec_list[year]

            # Costi fissi e ammortamenti
            fixed_costs_list = proj.get("fixed_costs") or [0] * years
            fixed_cost = fixed_costs_list[year]
            depreciation_list = proj.get("depreciation") or [0] * years
            depreciation = depreciation_list[year]
            depreciation_0 = proj.get("depreciation_0", 0) if year == 0 else 0
            ammortamenti_tot = depreciation + depreciation_0

            # Ricavi stocastici
            total_revenue = sum(
                sample(rev["price"], year) * sample(rev["quantity"], year)
                for rev in proj["revenues_list"]
            )

            # Costi variabili e aggiuntivi
            var_cost = total_revenue * proj["costs"]["var_pct"]
            other_costs_total = sum(sample(cost.get("values", None), year) for cost in proj.get("other_costs", []))

            # --- Calcolo EBITDA ---
            ebitda = total_revenue - var_cost - fixed_cost - other_costs_total

            # --- Calcolo EBIT ---
            ebit = ebitda - ammortamenti_tot

            # --- Tasse ---
            taxes = -ebit * proj["tax"]  # uscita positiva se EBIT>0, beneficio se EBIT<0
            if ebit < 0:
                taxes = -taxes  # beneficio fiscale positivo

            # --- Free Cash Flow ---
            fcf = ebitda + taxes - capex_init - capex_rec
            cash_flows.append(fcf)

        # --- PBP frazionario ---
        discounted_cum_cf = np.cumsum([cf / ((1 + wacc) ** (y + 1)) for y, cf in enumerate(cash_flows)])
        pbp_array[sim] = calculate_fractional_pbp(discounted_cum_cf)

        # --- NPV totale ---
        discounted_cf = [cf / ((1 + wacc) ** (y + 1)) for y, cf in enumerate(cash_flows)]
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
    pbp_percentiles = {f"p{p}": np.nanpercentile(pbp_array, p) for p in percentiles}

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


def calculate_yearly_financials(proj):
    """
    Calcola i valori medi anno per anno per un progetto:
    Ricavi, Costi, Ammortamenti, EBITDA, EBIT, Tasse, NOPAT, FCF, DCF.
    
    Args:
        proj (dict): progetto con tutti i dati
    
    Returns:
        pd.DataFrame: tabella anno per anno con tutte le metriche
        float: NPV medio
    """
    years = proj["years"]

    # Inizializzo array
    ricavi = np.zeros(years)
    var_costs = np.zeros(years)
    fixed_costs = np.array(proj.get("fixed_costs", [0]*years))
    other_costs = np.zeros(years)
    capex_init = np.zeros(years)
    capex_rec = np.array(proj.get("capex_rec", [0]*years))
    depreciation = np.array(proj.get("depreciation", [0]*years))
    depreciation_0 = proj.get("depreciation_0", 0)
    ammortamenti_tot = np.array([depreciation_0] + list(depreciation[1:]))

    # --- Calcolo medie anno per anno ---
    for year in range(years):
        # Ricavi medi
        total_rev = 0
        for rev in proj["revenues_list"]:
            price = rev["price"][year]["p1"]
            quantity = rev["quantity"][year]["p1"]
            total_rev += price * quantity
        ricavi[year] = total_rev

        # Costi variabili
        var_costs[year] = total_rev * proj["costs"]["var_pct"]

        # Costi aggiuntivi medi
        other_costs[year] = sum(cost["values"][year]["p1"] for cost in proj.get("other_costs", []))

        # CAPEX iniziale solo anno 0
        capex_init[year] = proj["capex"] if year == 0 else 0

    # --- EBITDA, EBIT, Tasse, NOPAT, FCF, DCF ---
    ebitda = ricavi - var_costs - fixed_costs - other_costs
    ebit = ebitda - ammortamenti_tot

    # Tasse: segno coerente con EBIT
    taxes = np.where(ebit >= 0, -ebit * proj["tax"], -ebit * proj["tax"])

    nopat = ebit + taxes  # perché taxes hanno già il segno corretto
    fcf = ebitda + taxes - capex_init - capex_rec

    # DCF
    wacc = calculate_wacc(proj["equity"], proj["debt"], proj["ke"], proj["kd"], proj["tax"])
    dcf = fcf / ((1 + wacc) ** (np.arange(1, years+1)))

    # NPV medio
    npv_medio = np.sum(dcf)

    # --- Creazione DataFrame ---
    df_financials = pd.DataFrame({
        "Anno": np.arange(1, years+1),
        "Ricavi": ricavi,
        "Costi variabili": var_costs,
        "Costi fissi": fixed_costs,
        "Costi aggiuntivi": other_costs,
        "CAPEX iniziale": capex_init,
        "CAPEX ricorrente": capex_rec,
        "Ammortamenti": ammortamenti_tot,
        "EBITDA": ebitda,
        "EBIT": ebit,
        "Tasse": taxes,
        "NOPAT": nopat,
        "FCF": fcf,
        "DCF": dcf
    })

    return df_financials, npv_medio
















