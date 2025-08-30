# ADSC1
Dieses Repository enthält den Code-Teil der Studienarbeit für das Modul ADSC21 Applied Data Science II: Machine Learning und Reporting.


### Installation

1. **Klonen Sie das Repository und wechseln Sie in das Verzeichnis**

    ```bash
    git clone git@github.com:juliisch/ADSC1.git
    ```
    ```bash
    cd ADSC1
    ```

2. **Bibliotheken installieren**

    Installieren Sie die benötigten Bibliotheken über die Datei `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    Nach der Installation der Bibliotheken ist es erforderlich, das Programm neu zu starten, damit diese wirksam werden.

2. **Führen Sie die Notebooks aus**

    - `notebooks/01_Merge.iypnb`: Zusammenführung der einzelnen Jahresdatensätze zu einem Gesamtdatensatz.
    - `notebooks/02_Exploration.iypnb`: Dateneinsicht und Datenaufbereitung
    - `notebooks/03_Visualization.iypnb`: Grafische Darstellung der aufbereiteten Daten 
    - `notebooks/04_Modelling.iypnb`: Training verschiedener ML-Modelle, Hyperparametertuning, Evaluation und Anwendung der XAI-Methoden
    
    
    **<span style="color:red">Hinweis:</span>**
    In der Datei `notebooks/functions.py` befinden sich die selbstgeschriebenen und verwendeten Funktionen. 