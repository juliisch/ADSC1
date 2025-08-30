# Importe
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt





def count_nanValue(df):
    '''
    Funktion:       Bestimmung der Anzahl der NaN-Werte pro Variable.
    Input:          df (Datensatz)
    Output:         df_nan (DF mit der absoluten und relativen Anzahl an NaN-Werte pro Variable)
    Funktionsweise: Basierend auf dem übergebenen Datensatz wird bestimmt, wie viele NaN-Werte es pro Spalte gibt.
                    Zusätzlich wird bestimmt, wie viel das in Prozent zum gesamten Datensatz ausmacht.
                    Beide Werte werden in einem DF gespeichert.
    '''
    # Anzahl der NaN-Werte
    count_nan = df.isna().sum()
    # Prozentsatz der NaN-Werte
    percent_nan = round((count_nan / len(df)) * 100,2)
    # Speicherung der Werte in einem DF
    df_nan = pd.DataFrame({
        'Anzahl NaN': count_nan,
        'Prozent NaN_': percent_nan
    }).sort_values(by='Anzahl NaN', ascending=False)

    return df_nan



def load_data():
    '''
    Funktion: Laden der aufbereiteten Daten, mit dem erforderlichen Datenformat.
    '''
    df = pd.read_csv("../data/output_data/property_sales_2004-2024_preped.csv",
        dtype={
            "district": "Int64",
            "year_built": "Int64",
            "units": "Int64",
            "bdrms": "Int64",
            "fbath": "Int64",
            "hbath": "Int64",
            "lotsize": "Int64",
            "sale_price": "Int64",
        }
    )
    return(df)



def calculate_metrics(y_true, y_pred, name_model):
    '''
    Funktion:       Bestimmung der Evaluationsmetriken.
    Input:          y_true (tatsächliche Zielwerte)
                    y_pred (Vom Modell vorhergesagte Zielwerte)
                    name_model (Name des Modells)
    Output:         df_result (Evaluationsmetriken als DF)
    Funktionsweise: Anhand der vom Modell vorhergesagten Zielwerte und der tatsächlichen Zielwerte werden die Evaluationsmetriken Mean Absolute Error, Root Mean Squared Error und das Bestimmtheitsmaß bestimmt.
                    Die Werte werden in einem DF gespeichert und zurückgegeben.
    '''

    # Berechnung der Evaluationsmetriken
    mae = round(mean_absolute_error(y_true, y_pred),3)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    r2 = round(r2_score(y_true, y_pred),3)
    
    # Ausgabe der Evaluationsmetriken
    print(name_model)
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print("-" * 40)
    
    # Speicherung der Evaluationsmetrike
    df_result = pd.DataFrame({
        "Model": [name_model],
        "MAE": [mae],
        "RMSE": [rmse],
        "R²": [r2]
    })
    
    return df_result



def generate_predictedVSactualPlot(y_pred, y_test, name_model):
    '''
    Funktion:       Erstellung eines Plot, in dem der vorhergesagte und tatsächliche Wert gegeinander geplottet werden.
    Input:          y_true (tatsächliche Zielwerte)
                    y_pred (Vom Modell bestimmte Zielwerte)
                    name_model (Name des Modells)
    Funktionsweise: Die vom Modell vorhergesagten Zielwerte und die tatsächlichen Zielwerte werden in einem Scatter-Plot grafisch dargestellt. 
                    Die Grafik wird anschließend gespeichert. 
    '''

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue", s=20) # Scatterplot
    plt.title("Vorhergesagter vs. Tatsächlicher Verkaufspreis (Sales Price)")
    plt.xlabel("Tatsächlicher Verkaufspreis", fontsize=12)
    plt.ylabel("Vorhergesagter Verkaufspreis", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../output/sales_price_predictionVsactual_{name_model}.png") # Speicherung der Grafik 
    plt.close()



def calculate_crossValidation(estimator_model,X_train,y_train, name_model):
    '''
    Funktion:       Durchführung einer Cross-Validation.
    Input:          estimator_model (Model)
                    X_train (Trainingsdaten)
                    y_train (Trainingsdaten-Zielwert)
                    name_model (Name des Modells)
    Output:         df_cv (Cross-Validation-Werte als DF)
    Funktionsweise: Anhand des Modell wird mit den Trainingsdaten und den Trainingsdaten-Zielwerte eine Cross-Validation mit 5 Folds durchgeführt.
                    Zudem werden die Ergebnisse der Cross-Validation in einem DF gespeichert und zurückgegeben.
    '''

    # Durchführung der Cross-Validation
    score_cv = cross_val_score(estimator=estimator_model, X=X_train, y=y_train, cv=5) 

    # Ausgabe der Cross-Validation Ergebnisse
    print(name_model)
    print(score_cv)
    print("-" * 60)

    # Speicherung der Cross-Validation Ergebnisse in einem DF
    results = {"Name": name_model}
    for i, score in enumerate(score_cv, start=1):
        results[f"score_cv_{i}"] = score
    df_cv = pd.DataFrame([results])

    return df_cv



def hyperparametertuning(estimator_model, parameters_list,X_train, y_train,  X_test, y_test, name_model):
    '''
    Funktion:       Durchführung einer Hyperparametertuning mit GridSearch.
    Input:          estimator_model (Model)
                    parameters_list (Paramter-Grid)
                    X_train (Trainingsdaten)
                    y_train (Trainingsdaten-Zielwert)
                    X_test (Testdaten)
                    y_test (Testdaten-Zielwert)
                    name_model (Name des Modells)
    Output:         best_model (Ermitteltes beste Modell)
                    df_result (DF mit den Evaluationskennzahlen)
    Funktionsweise: Für das übergebende Modell wird basierend auf dem Parameter-Grid das Hyperparamtertuning mit GridSearch durchgeführt.
                    Im Anschluss werden die besten Ergebnisse ausgegeben und das beste Modell gespeichert und zurückgegeben. 
                    Zudem werden für das beste Modell die Evaluationkennzahlen bestimmt und ebenfalls zurückgegeben.
    '''

    print(name_model)
    grid_search_temp = GridSearchCV(estimator=estimator_model, param_grid=parameters_list, cv=5, n_jobs=-1) # Definierung der Gridsearch-Objektes
    grid_search_temp.fit(X_train, y_train) # Durchführung des GridSearch

    print("Beste Parameter:", grid_search_temp.best_params_) # Ausgabe der optimalen Parameter
    print("Bester CV-Score:", grid_search_temp.best_score_) # Ausgabe des besten Cross-Validation Ergebnisses
    print("Test-Score:", grid_search_temp.score(X_test, y_test)) # Ausgabe des besten Score-Ergebnisses (Bestimmtheitsmaß)
    print("-" * 40) 

    best_model = grid_search_temp.best_estimator_ # Spicherung des besten Model
    y_pred_lr = best_model.predict(X_test) # Vorhersage mti dem besten Model
    df_result = calculate_metrics(y_test, y_pred_lr, name_model) # Bestimmung der Evaluationsmetriken mit dem besten Model

    return df_result, best_model



def calculate_shap(estimator_model, X_data, name_model):
    '''
    Funktion:       Anwendung der SHAP-Methode
    Input:          estimator_model (Model)
                    X_data (Daten ohne Zielvariable)
                    name_model (Name des Modells)
    Funktionsweise: Aus dem übergebenden Pipeline-Model werden die Preprocess-Schritte und Model-Schritte extrahiert. 
                    Die Daten werden Transformiert und die durch die Preprocess Entstandenen Spaltennamen werden gespeichert.
                    Abhängig vom Modell wird der Explainer bestimmt und entsprechend die Shap-Werte ermittelt.
                    Es wird ein Shap-Plot generiert und gespeicher. 
    Quelle:         In Anlehnung an https://github.com/shap/shap
    '''

    print(f"[INFO] Start SHAP {name_model}")

    step_preprocess   = estimator_model.named_steps["preprocess"] # Extrahierung des Preporcess-Schritts aus der Model-Pipeline
    step_model = estimator_model.named_steps["model"] # Extrahierung des Model-Schritts aus der Model-Pipeline
    X_data_transform  = step_preprocess.transform(X_data) # Transformation der Daten
    featureNames = step_preprocess.get_feature_names_out() # Bestimmung der Spaltennamen  
    
    if(name_model=="LineareRegression"):
        explainer = shap.Explainer(step_model, X_data_transform, feature_names=featureNames) # Erstellung des SHAP-Erklärers für LineareRegression
    else:
        explainer = shap.TreeExplainer(step_model)  # Erstellung des SHAP-Erklärers für die Baumbasierten Modelle
    
    shap_values = explainer.shap_values(X_data_transform) # Bestimmung der SHAP-Werte

    # Pro Feature wird der Mittelwert in einem df gespeichert
    df_shap = pd.DataFrame(shap_values, columns=featureNames)
    df_shap = df_shap.abs().mean().sort_values(ascending=False)
    df_shap = df_shap.reset_index()
    df_shap.columns = ["Feature", "MeanAbsSHAP"]
    df_shap['Model'] = name_model     # Speicherung des Modell-Namens
    df_save = df_shap.copy()

    # Plot: SHAP Plot (Model)
    plt.figure(figsize=(8, 6))  
    shap.summary_plot(shap_values, X_data_transform, feature_names=featureNames, show=False)
    plt.title(f"SHAP Plot {name_model}")
    plt.tight_layout()
    plt.savefig(f"../output/shap_{name_model}.png")
    plt.close()

    print(f"[INFO] End SHAP {name_model}")

    return df_save



def calculate_pfi(estimator_model, y_test, X_test, name_model):
    '''
    Funktion:       Anwendung der PFI-Methode
    Input:          estimator_model (Model)
                    y_test (Testdaten-Zielwert)
                    X_test (Testdaten)
                    name_model (Name des Modells)
    Output:         df_pfi (PFI-Werte der einzelnen Spalten)
    Funktionsweise: Aus dem übergebenden Pipeline-Model werden die Preprocess-Schritte und Model-Schritte extrahiert. 
                    Die Daten werden Transformiert und die durch die Preprocess Entstandenen Spaltennamen werden gespeichert.
                    Die PFI Methode wird mit 40 Wiederholungen durchlaufen, wobei das Bestimmtheitsmaß als Bewertungsgrundlage dient.
                    Die Ergebnisse der PFI-Methode werden in einem DF gespeichert und am Ende zurückgegeben.
                    Zudem wird ein PFI-Plot generiert und gespeicher. 
    Quelle:         In Anlehnung an https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    '''

    print(f"[INFO] Start PFI {name_model}")

    step_preprocess = estimator_model.named_steps["preprocess"] # Extrahierung des Preporcess-Schritts aus der Model-Pipeline
    step_model = estimator_model.named_steps["model"] # Extrahierung des Model-Schritts aus der Model-Pipeline
    X_test_transform = step_preprocess.transform(X_test) # Transformation der Daten
    featureNames = step_preprocess.get_feature_names_out() # Bestimmung der Spaltennamen  

    # Durchführung der Permutation Importance
    result = permutation_importance( 
        step_model,           
        X_test_transform,            
        y_test,
        n_repeats=40, # 40 wiederholungen 
        scoring="r2", # Bewertung nach dem Bestimmtheitsmaß
        random_state=123,
    )

    # Speicherung der PFI Ergebnisse pro Spalte
    df_pfi = pd.DataFrame({
        "feature": featureNames,
        "mean_importance": result.importances_mean # Mittlerer Bestimmtheitsmaß
    }).sort_values("mean_importance", ascending=False)
    df_pfi['Model'] = name_model     # Speicherung des Modell-Namens
    df_pfi = df_pfi.sort_values("mean_importance", ascending=True) 

    df_save = df_pfi.copy()

    # Plot: PFI Plot (Model)
    plt.figure(figsize=(8, max(4, 0.25*len(df_pfi))))
    plt.barh(df_pfi["feature"], df_pfi["mean_importance"])
    plt.xlabel("Permutation Importance")
    plt.title(f"PFI Plot {name_model}")
    plt.tight_layout()
    plt.savefig(f"../output/pfi_{name_model}.png")
    plt.close()

    print(f"[INFO] End PFI {name_model}")

    return df_save