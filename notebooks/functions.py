# Importe
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance



'''
Funktion:       Bestimmung der Anzahl der NaN-Werte pro Variable.
Input:          df (Datensatz)
Output:         df_nan (DF mit der absoluten und relativen Anzahl pro Variable)
Funktionsweise: Basierend auf dem übergebenden Datensatz wird bestimmt, wie viele NaN-Werte es pro Spalte gibt.
                Zusätzlich wird bestimmt, wie viel das in Prozent zum gesamzten Datensatz ausmacht.
                Beide Werte werden in einem DF gespeichert.
'''
def get_nanValue(df):

    count_nan = df.isna().sum()
    # Prozentsatz NaN
    nan_percent = round((count_nan / len(df)) * 100,2)
    # DataFrame
    df_nan = pd.DataFrame({
        'Anzahl NaN': count_nan,
        'Prozent NaN_': nan_percent
    }).sort_values(by='Anzahl NaN', ascending=False)

    return df_nan


'''
Funktion: Laden der aufbereiteten Daten, mit dem erforderlichen Datenformat.
'''
def load_data():
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
            "sale_date_month": "Int64",
            "sale_date_year": "Int64",
        }
    )
    return(df)





'''
Funktion:       Bestimmung der Evaluations-Metriken.
Input:          y_true (Echte Zielwerte)
                y_pred (Vom Modell bestimmte Zielwerte)
                name_model (Name des Modells)
Output:         df_result (Evaluationsmetriken als DF)
Funktionsweise: Anhand der vom Modell vorhergesagten Zielwerten und den tatsächlichen Zielwerte werden die Evaluationsmetriken Mean Absolute Error, Mean Squared Error und das Bestimmtheitsmaß bestimmt.
                Die Werte werden in einem DF gespeichert und zurückgegeben.
'''
def calculate_metrics(y_true, y_pred, name_model):
    # Berechnung der Evaluationsmetriken
    mae = round(mean_absolute_error(y_true, y_pred),3)
    mse = round(mean_squared_error(y_true, y_pred),3)
    r2 = round(r2_score(y_true, y_pred),3)
    
    # Ausgabe der Evaluationsmetriken
    print(name_model)
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")
    print("-" * 25)
    
    # Speicherung der Evaluationsmetrike
    df_result = pd.DataFrame({
        "Model": [name_model],
        "MAE": [mae],
        "MSE": [mse],
        "R²": [r2]
    })
    
    return df_result

'''
Funktion:       Erstellung eines Plot, in dem der vorhergesagte und tatsächliche Wert gegeinander geplottet werden.
Input:          y_true (Echte Zielwerte)
                y_pred (Vom Modell bestimmte Zielwerte)
                name_model (Name des Modells)
Funktionsweise: Die vom Modell vorhergesagten Zielwerte und die tatsächlichen Zielwerte werden in einem Scatter-Plot grafisch dargestellt. 
                Die Grafik wird anschließend gespeichert. 
'''
def generate_predictedVSactualPlot(y_pred, y_test, name_model):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Predictions
    ax1.scatter(y_test, y_pred, alpha=0.6, color="blue", s=20)
    #min_val, max_val = y_test.min(), y_test.max()
    #ax1.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect Prediction Line")

    # Axes titles and labels
    ax1.set_title("Vorhergesagter vs Tatsächlicher Verkaufspreis (Sales Price)")
    ax1.set_xlabel("Tatsächlicher Verkaufspreis", fontsize=12)
    ax1.set_ylabel("Vorhergesagter Verkaufspreis", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"../output/sales_price_predictionVsactual_{name_model}.png")
    plt.close(fig)

'''
Funktion:       Durchführung einer Cross-Validation.
Input:          estimator_model (Model)
                X_train (Trainingsdaten)
                y_train (Trainingsdaten-Zielwert)
                name_model (Name des Modells)
Output:         df_result (Cross-Validation-Werte als DF)
Funktionsweise: Anhand des Modell wird mit den Trainingsdaten und den Trainingsdaten-Zielwerte eine Cross-Validation mit 5 Folds durchgeführt.
                Zudem werden die Ergebnisse der Cross-Validation in einem DF gespeichert und zurückgegeben.
'''
def calculate_crossValidation(estimator_model,X_train,y_train, name_model):
    score_cv = cross_val_score(estimator=estimator_model, X=X_train, y=y_train, cv=5)

    print(name_model)
    print(score_cv)
    print("-" * 60)

    results = {"Name": name_model}
    for i, score in enumerate(score_cv, start=1):
        results[f"score_cv_{i}"] = score
    df_result = pd.DataFrame([results])

    return df_result

'''
Funktion:       Durchführung einer Hyperparametertuning mit GridSearch.
Input:          estimator_model (Model)
                parameters_model (Paramter-Grid)
                X_train (Trainingsdaten)
                y_train (Trainingsdaten-Zielwert)
                X_test (Testdaten)
                y_test (Testdaten-Zielwert)
                name_model (Name des Modells)
Output:         best_model (Ermitteltes beste Modell)
Funktionsweise: Für das übergebende Modell wird basierend auf dem Parameter-Grid das Hyperparamtertuning mit GridSearch durchgeführt.
                Im Anschluss werden die besten Ergebnisse ausgegeben und das beste Modell gespeichert und zurückgegeben. 
'''
def conduct_gridSearch(estimator_model,parameters_model, X_train,y_train, X_test, y_test, name_model):
    grid_search_dt = GridSearchCV(estimator=estimator_model, param_grid=parameters_model, cv=3, n_jobs=5)
    grid_search_dt.fit(X_train, y_train)

    print(name_model)
    print("Beste Parameter:", grid_search_dt.best_params_)
    print("Bester CV-Score:", grid_search_dt.best_score_)
    print("Test-Score:", grid_search_dt.score(X_test, y_test))
    print("-" * 40)

    best_model = grid_search_dt.best_estimator_

    return best_model

'''
Funktion:       Anwendung der SHAP-Methode
Input:          estimator_model (Model)
                X_train (Trainingsdaten)
                X_test (Testdaten)
                name_model (Name des Modells)
Funktionsweise: Aus dem übergebenden Pipeline-Model werden die Preprocess-Schritte und Model-Schritte extrahiert. 
                Die Trainings- und Testdaten werden Transformiert und die durch die Preprocess Entstandenen Spaltennamen werden gespeichert.
                Abhängig vom Modell wird der Explainer bestimmt und entsprechend die Shap-Werte ermittelt.
                Es wird ein Shap-Plot generiert und gespeicher. 
'''
def calculate_shap(estimator_model, X_train, X_test, name_model):
    step_preprocess   = estimator_model.named_steps["preprocess"]
    step_model = estimator_model.named_steps["model"]

    X_train_transform = step_preprocess.transform(X_train)
    X_test_transform  = step_preprocess.transform(X_test)
    featureNames = step_preprocess.get_feature_names_out()

    if(name_model=="LR"):
        explainer = shap.Explainer(step_model, X_train_transform, feature_names=featureNames)
    else:
        explainer = shap.TreeExplainer(step_model) 
    
    shap_values = explainer.shap_values(X_test_transform)

    plt.figure(figsize=(8, 6))  
    shap.summary_plot(shap_values, X_test_transform, feature_names=featureNames, show=False)
    plt.tight_layout()
    plt.savefig(f"../output/shap_{name_model}.png")
    plt.close()
    #plt.show()
    #shap.summary_plot(shap_values, X_test_transform, feature_names=featureNames) 

'''
Funktion:       Anwendung der PFI-Methode
Input:          estimator_model (Model)
                y_test (Testdaten-Zielwert)
                X_test (Testdaten)
                name_model (Name des Modells)
Funktionsweise: Aus dem übergebenden Pipeline-Model werden die Preprocess-Schritte und Model-Schritte extrahiert. 
                Die Trainings- und Testdaten werden Transformiert und die durch die Preprocess Entstandenen Spaltennamen werden gespeichert.
                Es wird die PFI-Methode mit 40 Wiederholungen angwendet und ....
                Es wird ein PFI-Plot generiert und gespeicher. 
'''
def calculate_pfi(estimator_model, y_test, X_test, name_model):
    step_preprocess = estimator_model.named_steps["preprocess"]
    step_model = estimator_model.named_steps["model"]
    X_test_transform = step_preprocess.transform(X_test)
    featureNames = step_preprocess.get_feature_names_out()

    # PFI auf dem END-MODELL, nicht auf der gesamten Pipeline,
    # und mit bereits transformierten Daten:
    result = permutation_importance(
        step_model,           # <- nur das Modell
        X_test_transform,            # <- transformierte Matrix
        y_test,
        n_repeats=40,
        scoring="r2",
        random_state=123,
    )

    imp_df = pd.DataFrame({
        "feature": featureNames,
        "mean_importance": result.importances_mean,
        "std": result.importances_std
    }).sort_values("mean_importance", ascending=False)

    topN = 30 
    plot_df = imp_df.sort_values("mean_importance", ascending=True).tail(topN)

    plt.figure(figsize=(8, max(4, 0.25*len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["mean_importance"], xerr=plot_df["std"])
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Feature Importancee")
    plt.tight_layout()
    plt.savefig(f"../output/pfi_{name_model}.png")
    plt.close()