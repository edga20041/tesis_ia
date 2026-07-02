import os
import joblib
import pandas as pd
from typing import Any, Dict
from tesis_ia.data.preprocessor import DataPreprocessor
from tesis_ia.models.factory import build_model

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = DataPreprocessor(config)

    def run(self) -> Dict[str, Any]:
        """Ejecuta el ciclo estructural de entrenamiento."""
        print("\n=== Iniciando Flujo del Torneo de Modelos ===")
        
        X_train, y_train, X_test, y_test = self.preprocessor.prepare_splits()
        
        model = build_model(self.config)
        
        print("[TRAIN] Entrenando algoritmo...")
        model.fit(X_train, y_train)
        
        artifacts_dir = self.config.get("paths", {}).get("artifacts_dir", "./artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        model_family = self.config.get("model", {}).get("family", "linear_model")
        model_path = os.path.join(artifacts_dir, f"{model_family}_trained.pkl")
        joblib.dump(model, model_path)
        print(f"[ARTIFACT] Guardado con exito en: {model_path}")
        
        outputs_dir = self.config.get("paths", {}).get("outputs_dir", "./outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        y_pred = model.predict(X_test)
        preds_df = pd.DataFrame({"real": y_test, "prediccion": y_pred})
        preds_df.to_csv(os.path.join(outputs_dir, "predictions_baseline.csv"), index=False)
        
        return {
            "status": "success",
            "model_path": model_path,
            "message": "Entrenamiento finalizado correctamente."
        }