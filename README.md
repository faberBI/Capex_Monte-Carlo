# 📊 CAPEX Risk Framework 

Un'interfaccia interattiva in **Streamlit** per analizzare rischio e rendimento di progetti di investimento **CAPEX** tramite **simulazioni Monte Carlo**, considerando distribuzioni stastistiche per ricavi, costi variabili e costi fissi.

---

## 🔹 Funzionalità principali

### ➕ Gestione Progetti

* Creazione di progetti multipli con parametri personalizzabili.
* Confronto tra progetti tramite **matrice rischio-rendimento**.

### 💰 Parametri Finanziari

* Struttura del capitale: **Equity / Debito**.
* **Costo dell’Equity (ke)** e **del Debito (kd)**.
* **Tax rate**.
* **CAPEX iniziale**.
* **Orizzonte temporale (anni)**.

### 📈 Parametri Ricavi

* Prezzo e quantità configurabili con distribuzioni:

  * Normale
  * Triangolare (p1, p2, p3)
  * Lognormale
  * Uniforme

### 🏭 Parametri Costi

* Costi variabili (% sui ricavi).
* Costi fissi annuali.

### 🎲 Simulazione Monte Carlo

* Numero simulazioni configurabile.
* Indicatori di rischio:

  * **NPV** (Net Present Value).
  * **CaR (Capital at Risk 95%)**.
  * **Probabilità NPV < 0**.
  * **CVaR (Conditional VaR 95%)**.
  * **IRR distribution**.
  * **Payback to period distribution**.
  * **Profitability index distribution**.

### 📊 Visualizzazioni

* Distribuzione NPV (**istogramma**).
* **Boxplot** NPV.
* **Cash flow annuo medio** (bar chart).
* **Matrice rischio-rendimento** per confronto progetti.

### 🤖 Open AI

* Commento risultati generato da **GPT** a partire dalle simulazioni Monte Carlo.

---

## 🔹 Tecnologie

* Python 3.x
* Streamlit
* NumPy
* Matplotlib
* Open AI

---

## 🔹 Come usare

### 1. Installare le dipendenze

```bash
pip install streamlit numpy matplotlib
```

### 2. Eseguire l’applicazione

```bash
streamlit run app.py
```

### 3. Interfaccia interattiva

* Cliccare su **➕ Aggiungi progetto**.
* Configurare parametri **finanziari, ricavi, costi, trend**.
* Impostare numero simulazioni Monte Carlo.
* Visualizzare **risultati e grafici**.
* GPT integrato per commento ed analisi dei **risultati**.

---

## 🔹 Note metodologiche

* Ricavi e quantità generati tramite distribuzioni casuali definite dall’utente.
* Il **WACC** calcolato automaticamente da equity, debito e rispettivi costi.
* Simulazione Monte Carlo con NPV scontato al WACC.
* Indicatori di rischio:

  * **CaR 95%**: differenza tra NPV atteso e 5° percentile.
  * **Probabilità NPV < 0**.
  * **CVaR 95%**: media dei peggiori 5% scenari.

---
## ✨ Autori
Il progetto è stato sviluppato da:  
- **Fabrizio Di Sciorio, PhD** – Senior Data Scientist  

