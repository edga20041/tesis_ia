import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def main():
    print("=======================================")
    print(" Evaluacion de Resultados (Baseline)")
    print("=======================================\n")
    
    csv_path = "outputs/predictions_baseline.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el archivo {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    y_true = df["real"]
    y_pred = df["prediccion"]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"-> RMSE (Error Cuadrático Medio): {rmse:.2f} unidades")
    print(f"-> MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%\n")
    
    report_dir = "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "baseline_metrics.txt")
    
    with open(report_path, "w") as f:
        f.write("Métricas del Modelo Champion Inicial (Regresión Lineal)\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"RMSE: {rmse:.2f} unidades\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write("\nInterpretación:\n")
        f.write("- Este es el 'piso' de nuestro Torneo de Modelos.\n")
        f.write("- Los modelos desafiantes (XGBoost/LightGBM) deberán superar este desempeño.\n")
        
    print(f"[OK] Reporte guardado en {report_path}")

if __name__ == "__main__":
    main()
