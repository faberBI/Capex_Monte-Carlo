"""
damodaran_data.py
=================
Accesso ai dati di mercato di Aswath Damodaran (Stern NYU) per il costo del capitale:
  - beta unlevered di settore, per regione (Europe / US / Global / Emerging / Japan);
  - equity risk premium totale per paese.

Il modulo carica un dataset VERSIONATO se presente (prodotto da damodaran_update.py),
altrimenti usa uno SNAPSHOT incorporato. Ogni valore porta con sé la propria provenienza
(fonte + data), così la valutazione è riproducibile: sai sempre quale annata hai usato.

  - Gli ERP dello snapshot sono valori reali Damodaran, annata gennaio 2026.
  - I beta dello snapshot sono INDICATIVI (pochi settori, per far funzionare la UI):
    esegui `python damodaran_update.py` per scaricare la tabella settoriale ufficiale e
    completa. Fino ad allora i beta sono etichettati come "indicativi".

Fonti ufficiali (scaricate dall'updater):
  ERP paese      : https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xls
  Beta Europe    : https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaEurope.xls
  Beta US        : https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls
  Beta Emerging  : https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaemerg.xls
  Beta Global    : https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaglobal.xls
"""

import json
import os

# percorso del dataset aggiornato (output dell'updater); se assente si usa lo snapshot
DATASET_PATH = os.environ.get(
    "DAMODARAN_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "damodaran_dataset.json"),
)

# ==========================================================================
# SNAPSHOT INCORPORATO (fallback)
# ==========================================================================
# ERP TOTALE per paese (%), annata Damodaran gennaio 2026.
# Fonte: pages.stern.nyu.edu/~adamodar (ctryprem), sintesi pubbliche gennaio 2026.
_ERP_SNAPSHOT = {
    "Mature market (base)": 4.33,
    "United States": 4.50,
    "Germany": 4.33,
    "Netherlands": 4.33,
    "Switzerland": 4.33,
    "Austria": 4.86,
    "Belgium": 5.13,
    "France": 5.13,          # Aa3 (indicativo)
    "United Kingdom": 5.13,  # Aa3 (indicativo)
    "Italy": 6.70,
    "Spain": 5.80,
    "Portugal": 5.80,
    "Greece": 7.10,
    "Australia": 4.33,
}
# quali ERP sono valori Damodaran puntuali vs stime indicative
_ERP_INDICATIVE = {"France", "United Kingdom"}

# BETA UNLEVERED di settore (INDICATIVI) per regione. Solo alcuni settori: la UI funziona,
# ma per l'elenco completo e i valori ufficiali va eseguito l'updater.
_BETA_SNAPSHOT_EUROPE = {
    # settore: (beta unlevered, beta unlevered corretto per la cassa)
    "Telecommunications (fixed line)": (0.60, 0.62),
    "Telecom (Wireless)": (0.75, 0.78),
    "Utility (General)": (0.40, 0.41),
    "Green & Renewable Energy": (0.60, 0.62),
    "Software (System & Application)": (1.15, 1.20),
    "Retail (General)": (0.90, 0.93),
    "Banks (Regional)": (0.45, 0.46),
    "Construction Supplies": (1.00, 1.03),
    "Healthcare Products": (0.90, 0.92),
    "Business & Consumer Services": (0.95, 0.98),
}
# per lo snapshot usiamo gli stessi valori indicativi per US/Global (verranno sovrascritti
# dall'updater con i valori regionali corretti)
_BETA_SNAPSHOT = {
    "Europe": _BETA_SNAPSHOT_EUROPE,
    "US": _BETA_SNAPSHOT_EUROPE,
    "Global": _BETA_SNAPSHOT_EUROPE,
}

_SNAPSHOT_META = {"source": "snapshot incorporato", "date": "2026-01 (indicativo per i beta)",
                  "is_snapshot": True}


# ==========================================================================
# CARICAMENTO
# ==========================================================================
class DamodaranData:
    """Contenitore dei dati con provenienza. Metodi di lookup con override sempre possibile
    a monte (l'app pre-compila ma lascia modificare i numeri)."""

    def __init__(self, betas, erp, meta, erp_indicative=None):
        self.betas = betas            # {region: {sector: (ub, ub_cash)}}
        self.erp = erp                # {country: value_pct}
        self.meta = meta              # {source, date, is_snapshot}
        self.erp_indicative = erp_indicative or set()

    # --- ERP ---
    def list_countries(self):
        return list(self.erp.keys())

    def get_erp(self, country):
        """Ritorna (erp_percentuale, fonte, data). erp in PUNTI PERCENTUALI (es. 6.70)."""
        val = self.erp.get(country)
        if val is None:
            return None, self.meta["source"], self.meta["date"]
        src = self.meta["source"]
        date = self.meta["date"]
        if country in self.erp_indicative:
            src = src + " · valore indicativo"
        return float(val), src, date

    # --- beta ---
    def list_regions(self):
        return list(self.betas.keys())

    def list_sectors(self, region):
        return sorted(self.betas.get(region, {}).keys())

    def get_unlevered_beta(self, region, sector, cash_adjusted=True):
        """Ritorna (beta_unlevered, fonte, data)."""
        rec = self.betas.get(region, {}).get(sector)
        if rec is None:
            return None, self.meta["source"], self.meta["date"]
        ub, ubc = rec if isinstance(rec, (list, tuple)) else (rec, rec)
        val = ubc if cash_adjusted else ub
        src = self.meta["source"]
        if self.meta.get("is_snapshot"):
            src = src + " · beta indicativo (eseguire updater)"
        return float(val), src, self.meta["date"]


def load(path=None):
    """Carica il dataset versionato (JSON dall'updater) se presente, altrimenti lo snapshot."""
    p = path or DATASET_PATH
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            betas = {reg: {s: tuple(v) for s, v in sec.items()}
                     for reg, sec in d.get("betas", {}).items()}
            erp = {k: float(v) for k, v in d.get("erp", {}).items()}
            meta = d.get("metadata", {})
            meta.setdefault("source", "Damodaran (dataset scaricato)")
            meta.setdefault("date", "n/d")
            meta["is_snapshot"] = False
            return DamodaranData(betas, erp, meta, erp_indicative=set())
        except Exception as e:  # dataset corrotto -> fallback prudente allo snapshot
            print(f"[damodaran_data] dataset non leggibile ({e}), uso lo snapshot.")
    return DamodaranData(_BETA_SNAPSHOT, _ERP_SNAPSHOT, dict(_SNAPSHOT_META),
                         erp_indicative=set(_ERP_INDICATIVE))


if __name__ == "__main__":
    dd = load()
    print("Provenienza:", dd.meta)
    print("Regioni:", dd.list_regions())
    print("Settori (Europe):", dd.list_sectors("Europe")[:5], "...")
    b, s, dt = dd.get_unlevered_beta("Europe", "Telecommunications (fixed line)")
    print(f"β_U telecom fixed line = {b}  [{s}, {dt}]")
    print("Paesi:", dd.list_countries())
    e, s, dt = dd.get_erp("Italy")
    print(f"ERP Italy = {e}%  [{s}, {dt}]")
