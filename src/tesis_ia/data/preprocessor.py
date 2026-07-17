import pandas as pd
from typing import Any, Tuple

class DataPreprocessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_config = config.get("data", {})
        self.features = self.data_config.get("features", [])
        self.target = self.data_config.get("target_column", "unidades_vendidas")

    def prepare_splits(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        path = self.data_config.get("processed_path", "data/processed/dataset_enriquecido.csv")
        df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1", low_memory=False)
        
        # Limpieza de nulos inevitables por el calculo de lag_12
        df_clean = df.dropna(subset=self.features + [self.target]).copy()
        
        train_mask = df_clean["anio"].isin(self.data_config.get("train_years", [2021, 2022, 2023, 2024]))
        test_mask = df_clean["anio"].isin(self.data_config.get("test_years", [2025]))
        
        train_df = df_clean[train_mask]
        test_df = df_clean[test_mask]
        
        X_train = train_df[self.features]
        y_train = train_df[self.target]
        X_test = test_df[self.features]
        y_test = test_df[self.target]
        
        print(f"[DATA] Entrenamiento (2021-2024): {len(X_train):,} registros.")
        print(f"[DATA] Validacion (2025): {len(X_test):,} registros.")
        
        return X_train, y_train, X_test, y_test

    def describe(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "target": self.target,
            "train_years": self.data_config.get("train_years", [2021, 2022, 2023, 2024]),
            "test_years": self.data_config.get("test_years", [2025]),
        }