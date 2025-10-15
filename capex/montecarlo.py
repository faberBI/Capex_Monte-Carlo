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

def run_montecarlo(proj, n_sim, wacc):
    """
    Esegue la simulazione Monte Carlo per un progetto.
    
    Args:
        proj (dict): progetto con ricavi, costi, CAPEX ricorrente, ecc.
        n_sim (int): numero di simulazioni
        wacc (float): WACC del progetto
    
    Returns:
        dict: risultati simulazione contenente npv_array, yearly_cash_flows, car, cvar, ecc.
    """
    years = proj["years"]
    npv_array = np.zeros(n_sim)
    yearly_cash_flows = np.zeros((n_sim, years))

    for sim in range(n_sim):
        cash_flows = []
        for year in range(years):
            # CAPEX iniziale solo al primo anno
            capex_init = proj["capex"] if year == 0 else 0
            # CAPEX ricorrente fisso
            capex_rec = proj.get("capex_rec", [0]*years)[year]
            # Costi fissi anno per anno
            fixed_cost = proj.get("fixed_costs", [0]*years)[year]

            # Ricavi totali (stocastici)
            total_revenue = 0
            for rev in proj["revenues_list"]:
                price = sample(rev["price"], year)
                quantity = sample(rev["quantity"], year)
                total_revenue += price * quantity

            # Costi variabili
            var_cost = total_revenue * proj["costs"]["var_pct"]

            # Costi aggiuntivi stocastici
            other_costs_total = sum(sample(cost.get("values", None), year) for cost in proj.get("other_costs", []))

            # Ammortamento
            depreciation = proj.get("depreciation", [0]*years)[year]

            # Cash flow operativo
            cf = total_revenue - var_cost - fixed_cost - other_costs_total - capex_init - capex_rec
            cash_flows.append(cf)

        # Sconto CF al presente
        discounted_cf = [cf / ((1 + wacc) ** (year + 1)) for year, cf in enumerate(cash_flows)]
        npv_array[sim] = sum(discounted_cf)
        yearly_cash_flows[sim, :] = cash_flows

    expected_npv = np.mean(npv_array)
    car = np.percentile(npv_array, 5)  # Capital at Risk 95%
    cvar = np.mean(npv_array[npv_array <= car]) if np.any(npv_array <= car) else car
    downside_prob = np.mean(npv_array < 0)

    return {
        "npv_array": npv_array,
        "yearly_cash_flows": yearly_cash_flows.mean(axis=0),
        "expected_npv": expected_npv,
        "car": car,
        "cvar": cvar,
        "downside_prob": downside_prob
    }










