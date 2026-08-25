"""Script de prueba y validación del motor de Cold Start."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tesis_ia.models.cold_start import ColdStartEngine

def test_cold_start():
    engine = ColdStartEngine()
    n = 60
    engine.catalog_df = pd.DataFrame({
        'id_producto': [f'prod_{i}' for i in range(n)],
        'id_producto_clean': [f'prod_{i}' for i in range(n)],
        'sku': [f'SKU_{i}' for i in range(n)],
        'descripcion': [f'Aceite Girasol {i*100}ml' for i in range(n)],
        'nombre_departamento': ['Alimentos' for _ in range(n)],
        'id_departamento': [1 for _ in range(n)],
        'nombre_categoria': ['Aceites' for _ in range(n)],
        'id_categoria': [5 for _ in range(n)],
        'nombre_marca': ['Familiar' if i % 2 == 0 else 'Primor' for i in range(n)],
        'id_marca': ['marca_1' if i % 2 == 0 else 'marca_2' for i in range(n)],
        'contenido_neto': [900.0 if i % 2 == 0 else 1500.0 for i in range(n)],
        'unidad_medida': ['ml' for _ in range(n)],
        'valor_unitario': [12000.0 + i*100 for i in range(n)],
        'avg_unidades': [150.0 + i*2 for i in range(n)],
        'total_unidades': [1800.0 for _ in range(n)],
    })

    engine.seasonality_map = {
        (f'prod_{i}', m): (150.0 + i*2) * (1.25 if m == 12 else (0.75 if m == 2 else 1.0))
        for i in range(n) for m in range(1, 13)
    }

    numeric_features = ['log_precio', 'log_contenido']
    categorical_features = ['id_departamento', 'id_categoria', 'unidad_medida', 'id_marca']
    engine.catalog_df['log_precio'] = np.log1p(engine.catalog_df['valor_unitario'])
    engine.catalog_df['log_contenido'] = np.log1p(engine.catalog_df['contenido_neto'])

    engine.pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ]
    )
    X_transformed = engine.pipeline.fit_transform(engine.catalog_df)
    engine.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    engine.catalog_df['cluster_id'] = engine.kmeans.fit_predict(X_transformed)
    engine.knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    engine.knn.fit(X_transformed)
    engine.is_fitted = True

    res = engine.predict_new_product(
        id_departamento=1,
        id_categoria=5,
        id_marca='marca_1',
        descripcion='Nuevo Aceite de Maiz 900ml',
        contenido_neto=900.0,
        unidad_medida='ml',
        valor_unitario=13500.0,
        horizonte_meses=12,
        anio_inicio=2026,
        mes_inicio=1
    )

    print("=== Resultado de Validación Cold Start ===")
    print("Status:", res['status'])
    print("Cluster Asignado:", res['producto_nuevo']['cluster_asignado'])
    print("Unidades Base Estimadas:", res['producto_nuevo']['unidades_base_estimadas'])
    print(f"Productos Gemelos Identificados ({len(res['productos_similares_gemelos'])}):")
    for p in res['productos_similares_gemelos']:
        print(f"  - {p['descripcion']} (Similitud: {p['similitud_pct']}%, Ventas Prom: {p['ventas_promedio_mensual']} u)")
    print("Primeros 3 Meses Proyectados:")
    for m in res['proyeccion_12_meses'][:3]:
        print(f"  Mes {m['anio']}-{m['mes']:02d}: {m['unidades_predichas']} u | Monto: Gs {m['monto_predicho']}")
    print("==========================================")

if __name__ == '__main__':
    test_cold_start()
