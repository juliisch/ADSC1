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

2. **Führen Sie das Notebook aus**

    - In dem Notebook `notebooks/01_Merge.iypnb` werden die Daten der einzelnen Jahre zu einem Datensatz zusammengeführt.
    - Das Notebook `notebooks/02_EDA.iypnb` beinhaltet die Dateneinsicht, Datenaufbereitung. 
    - In dem Notebook `notebooks/03_Visualization.iypnb` werden die zusammengeführten und aufbereiteten Daten grafisch dargestellt. 
    - Das Notebook `notebooks/04_Modelling.iypnb` beinhaltet das Training der verschiednen Modelle inklusive der Evaluation und anwendung der XAI Methoden. 
    
    Hinweis: In der Datei `notebooks/functions.py` befinden sich die verwendeten geschriebenen Funktionen. 