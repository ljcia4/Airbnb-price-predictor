# 🏠 Airbnb Price Predictor - Napoli

Questo progetto implementa una pipeline di machine learning end-to-end per la previsione dei prezzi degli alloggi Airbnb nella città di Napoli. Attraverso l'analisi dei dati storici, il sistema pulisce, elabora e modella le informazioni per fornire stime accurate dei prezzi di affitto.

## 📋 Descrizione del Progetto

L'obiettivo principale è costruire un modello predittivo affidabile che possa aiutare host e potenziali affittuari a stimare il valore di mercato di una proprietà. Il progetto copre tutte le fasi del ciclo di vita del machine learning:

1.  **Data Cleaning & Preprocessing**: Pulizia approfondita del dataset grezzo, gestione dei valori mancanti e rimozione degli outlier.
2.  **Feature Engineering**: Creazione di nuove variabili significative e trasformazione di quelle esistenti.
3.  **Model Training**: Addestramento e valutazione di diversi algoritmi di regressione.

## 📂 Struttura della Repository

La repository è organizzata in modo intuitivo per facilitare la navigazione:

```
Airbnb-price-predictor/
├── dataset/
│   ├── csv/                   # Contiene i file CSV (grezzi, puliti, train/test)
│   ├── data_cleaner.py        # Script Python per la pulizia e preparazione dei dati
│   └── inspect_data.ipynb     # Notebook per l'ispezione preliminare dei dati
├── gradient_boosting_testing.ipynb  # Training e valutazione con Gradient Boosting
├── randomforest_testing.ipynb       # Training e valutazione con Random Forest
├── requirements.txt           # Elenco delle dipendenze del progetto
└── README.md                  # Documentazione del progetto
```

## 🛠️ Requisiti e Installazione

Assicurati di avere Python installato sul tuo sistema. Per installare tutte le librerie necessarie, esegui:

```bash
pip install -r requirements.txt
```

Le principali librerie utilizzate includono:
- **Pandas**: Per la manipolazione e l'analisi dei dati.
- **Scikit-learn**: Per gli algoritmi di machine learning e il preprocessing.
- **Jupyter**: Per l'esecuzione dei notebook interattivi.

## 🚀 Utilizzo

### 1. Preparazione dei Dati

Il primo passo è pulire il dataset grezzo. Lo script `data_cleaner.py` si occupa di:
- Rimuovere colonne non necessarie.
- Gestire i valori mancanti (imputazione tramite moda/mediana).
- Eseguire il One-Hot Encoding per le variabili categoriche.
- Dividere i dati in set di training e test.

Esegui lo script dalla root del progetto:

```bash
python dataset/data_cleaner.py
```

I file generati (`clean_Airbnb_Napoli.csv`, `train_Airbnb_Napoli.csv`, `test_Airbnb_Napoli.csv`) verranno salvati nella cartella `dataset/csv/`.

### 2. Addestramento dei Modelli

Una volta preparati i dati, puoi procedere con l'addestramento dei modelli utilizzando i Jupyter Notebook forniti nella directory principale:

- **Random Forest**: Apri ed esegui `randomforest_testing.ipynb`.
- **Gradient Boosting**: Apri ed esegui `gradient_boosting_testing.ipynb`.

Ogni notebook include passaggi per il caricamento dei dati processati, il training del modello e la valutazione delle performance (es. tramite MSE, R^2).

## 👤 Autori

- **Felicia Riccio**