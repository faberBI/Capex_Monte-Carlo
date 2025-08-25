📊 CAPEX Risk Framework con WACC & Trend Annuali

Questo progetto fornisce un'interfaccia interattiva in Streamlit per analizzare il rischio e il rendimento di progetti d'investimento CAPEX utilizzando simulazioni Monte Carlo, tenendo conto del WACC, dei trend annuali di prezzi, quantità e costi fissi.

🔹 Funzionalità principali

Aggiunta progetti

Possibilità di aggiungere progetti multipli con parametri personalizzabili.

Parametri finanziari

Equity / Debito

Costo dell’Equity (ke) e del Debito (kd)

Tax rate

CAPEX iniziale

Orizzonte temporale (anni)

Calcolo automatico del WACC.

Parametri ricavi

Prezzo e quantità configurabili con distribuzioni: Normale, Triangolare, Lognormale, Uniforme.

Parametri della distribuzione personalizzabili (p1, p2, p3 per triangolare).

Parametri costi

Costi variabili (% sui ricavi)

Costi fissi annuali

Trend di inflazione dei costi fissi anno per anno.

Trend annuali

Crescita di prezzo e quantità per ciascun anno.

Possibilità di trend negativo.

Simulazione Monte Carlo

Numero di simulazioni configurabile.

Calcolo NPV, CaR (95%), probabilità di NPV < 0 e Conditional VaR (CVaR).

Visualizzazioni

Distribuzione NPV (istogramma)

Boxplot NPV

Cash flow annuo medio (bar chart)

Matrice rischio-rendimento per confronto tra progetti

🔹 Tecnologie

Python 3.x

Streamlit

NumPy

Matplotlib

🔹 Come usare

Installare le dipendenze:

pip install streamlit numpy matplotlib


Eseguire l'applicazione:

streamlit run app.py


Interfaccia interattiva:

Cliccare su ➕ Aggiungi progetto per creare un nuovo progetto.

Configurare i parametri finanziari, ricavi, costi e trend annuali.

Impostare il numero di simulazioni Monte Carlo.

Visualizzare i risultati e i grafici generati.

🔹 Note metodologiche

I ricavi e le quantità sono generati tramite distribuzioni casuali definite dall’utente.

Il WACC viene calcolato automaticamente in base al peso di equity e debito e ai relativi costi.

La simulazione Monte Carlo considera le variazioni casuali dei ricavi e calcola il NPV scontato con il WACC.

Gli indicatori di rischio includono:

CaR (Capital at Risk) 95%: differenza tra NPV atteso e il 5° percentile

Probabilità NPV < 0

Conditional VaR (CVaR 95%): media dei peggiori 5% scenari

🔹 Possibili sviluppi futuri

Aggiunta di più tipologie di distribuzioni casuali.

Esportazione dei risultati in CSV o Excel.

Integrazione con dashboard interattive avanzate.

Supporto per analisi multi-progetto con correlazioni tra flussi di cassa.
