"""Pruebas del motor de arranque en frio.

El modelo de pronostico necesita historia; un producto que se lanza manana no la
tiene. Este motor cubre ese hueco transfiriendo el perfil de demanda de los
productos mas afines del catalogo, atenuado por una rampa de penetracion
comercial.

Todo el metodo depende de MySQL, asi que las pruebas inyectan un catalogo
sintetico en lugar de la consulta: se sustituye load_catalog_and_history y el
resto del motor -pipeline de scikit-learn, K-Means, TF-IDF y la busqueda
jerarquica- se ejercita tal cual corre en produccion.
"""

import numpy as np
import pandas as pd
import pytest

from tesis_ia.models.cold_start import ColdStartEngine


def producto(id_producto, id_categoria, id_departamento, **cambios):
    """Una fila del catalogo con todas las columnas que el motor exige."""
    fila = {
        "id_producto": id_producto,
        "sku": f"SKU-{id_producto}",
        "descripcion": "gaseosa cola botella retornable",
        "contenido_neto": 1000.0,
        "unidad_medida": "ml",
        "valor_unitario": 10000.0,
        "id_marca": "M1",
        "nombre_marca": "Marca Uno",
        "id_empresa": "7",
        "id_categoria": id_categoria,
        "nombre_categoria": f"Categoria {id_categoria}",
        "id_departamento": id_departamento,
        "nombre_departamento": f"Departamento {id_departamento}",
        "avg_unidades": 100.0,
        "total_unidades": 1200.0,
    }
    fila.update(cambios)
    fila["id_producto_clean"] = str(fila["id_producto"]).strip().lower()
    return fila


CATALOGO = [
    producto("p1", 10, 1, avg_unidades=100.0),
    producto("p2", 10, 1, avg_unidades=120.0),
    producto("p3", 10, 1, avg_unidades=80.0),
    producto("p4", 10, 1, avg_unidades=90.0),
    producto("p5", 11, 1, avg_unidades=500.0, descripcion="jabon en polvo ropa blanca"),
    producto("p6", 20, 2, avg_unidades=700.0, descripcion="yerba mate compuesta suave"),
]


def motor(catalogo=None, estacionalidad=None):
    """Motor ajustado sobre un catalogo en memoria, sin conexion a la base."""
    filas = list(CATALOGO if catalogo is None else catalogo)
    catalog_df = pd.DataFrame(filas)
    mapa = dict(estacionalidad or {})

    engine = ColdStartEngine(engine=object())
    engine.load_catalog_and_history = lambda: (catalog_df, mapa)
    assert engine.fit(), "el motor no logro ajustarse sobre el catalogo sintetico"
    return engine


def lanzamiento(engine, **cambios):
    """Proyecta un producto nuevo con parametros por defecto razonables."""
    argumentos = {
        "id_departamento": 1,
        "id_categoria": 10,
        "id_marca": "M1",
        "contenido_neto": 1000.0,
        "unidad_medida": "ml",
        "valor_unitario": 10000.0,
        "descripcion": "gaseosa cola botella retornable",
        "anio_inicio": 2026,
        "mes_inicio": 1,
    }
    argumentos.update(cambios)
    return engine.predict_new_product(**argumentos)


@pytest.fixture(scope="module")
def motor_ajustado():
    """El ajuste entrena K-Means y TF-IDF: se reutiliza entre pruebas."""
    return motor()


@pytest.fixture(scope="module")
def proyeccion(motor_ajustado):
    return lanzamiento(motor_ajustado)


class TestAjusteDelMotor:

    def test_se_ajusta_sobre_un_catalogo_con_historia(self, motor_ajustado):
        assert motor_ajustado.is_fitted

    def test_no_se_ajusta_con_un_catalogo_vacio(self):
        # Una empresa recien dada de alta no tiene catalogo: el motor tiene que
        # rehusarse a ajustar, no inventar vecinos.
        vacio = ColdStartEngine(engine=object())
        vacio.load_catalog_and_history = lambda: (pd.DataFrame(), {})

        assert vacio.fit() is False
        assert vacio.is_fitted is False

    def test_asigna_un_clarter_a_cada_producto_del_catalogo(self, motor_ajustado):
        assert "cluster_id" in motor_ajustado.catalog_df.columns
        assert motor_ajustado.catalog_df["cluster_id"].notna().all()

    def test_un_catalogo_sin_ajustar_no_puede_proyectar_sin_base(self):
        # predict_new_product intenta ajustar solo; si el catalogo esta vacio
        # tiene que fallar de forma explicita y no devolver una proyeccion hueca.
        sin_datos = ColdStartEngine(engine=object())
        sin_datos.load_catalog_and_history = lambda: (pd.DataFrame(), {})

        with pytest.raises(RuntimeError, match="Cold Start"):
            lanzamiento(sin_datos)


class TestBusquedaJerarquica:
    """Categoria, y si no alcanza, departamento, y si tampoco, todo el catalogo.

    El umbral es de tres candidatos. Sin el, una categoria con un solo producto
    transferiria su curva entera al lanzamiento, sin ninguna diversificacion.
    """

    def test_resuelve_dentro_de_la_categoria_cuando_tiene_suficientes_productos(
        self, motor_ajustado
    ):
        resultado = lanzamiento(motor_ajustado, id_categoria=10, id_departamento=1)

        categorias = {p["categoria"] for p in resultado["productos_similares_gemelos"]}
        assert categorias == {"Categoria 10"}

    def test_sube_al_departamento_cuando_la_categoria_es_demasiado_chica(
        self, motor_ajustado
    ):
        resultado = lanzamiento(motor_ajustado, id_categoria=11, id_departamento=1)

        departamentos = {
            p["departamento"] for p in resultado["productos_similares_gemelos"]
        }
        assert departamentos == {"Departamento 1"}
        assert len(resultado["productos_similares_gemelos"]) >= 3

    def test_recorre_el_catalogo_entero_cuando_el_departamento_tampoco_alcanza(
        self, motor_ajustado
    ):
        resultado = lanzamiento(motor_ajustado, id_categoria=20, id_departamento=2)

        departamentos = {
            p["departamento"] for p in resultado["productos_similares_gemelos"]
        }
        assert len(departamentos) > 1

    def test_nunca_transfiere_desde_mas_de_cinco_gemelos(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, id_categoria=20, id_departamento=2)
        assert len(resultado["productos_similares_gemelos"]) <= 5

    def test_los_gemelos_vienen_ordenados_de_mayor_a_menor_afinidad(self, proyeccion):
        similitudes = [p["similitud_pct"] for p in proyeccion["productos_similares_gemelos"]]
        assert similitudes == sorted(similitudes, reverse=True)

    def test_prefiere_al_gemelo_de_la_misma_marca_y_el_mismo_precio(self, motor_ajustado):
        # La afinidad pondera marca compartida y cercania de precio: el candidato
        # identico al lanzamiento tiene que encabezar la lista.
        catalogo = [
            producto("clon", 10, 1, valor_unitario=10000.0, id_marca="M1"),
            producto("caro", 10, 1, valor_unitario=90000.0, id_marca="M9"),
            producto("barato", 10, 1, valor_unitario=500.0, id_marca="M8"),
        ]
        resultado = lanzamiento(motor(catalogo))

        assert resultado["productos_similares_gemelos"][0]["id_producto"] == "clon"


class TestRampaDePenetracion:
    """Un lanzamiento no alcanza su nivel de regimen el primer mes.

    Sin la rampa, la proyeccion le atribuiria a un producto desconocido la venta
    promedio de productos ya instalados en gondola desde el mes uno.
    """

    def test_el_primer_mes_se_proyecta_al_sesenta_y_cinco_por_ciento(self, proyeccion):
        assert proyeccion["proyeccion_12_meses"][0]["factor_penetracion"] == 0.65

    def test_el_segundo_mes_se_proyecta_al_ochenta_y_cinco_por_ciento(self, proyeccion):
        assert proyeccion["proyeccion_12_meses"][1]["factor_penetracion"] == 0.85

    def test_desde_el_tercer_mes_se_proyecta_al_cien_por_ciento(self, proyeccion):
        restantes = proyeccion["proyeccion_12_meses"][2:]
        assert {m["factor_penetracion"] for m in restantes} == {1.0}

    def test_la_rampa_es_monotona_creciente(self, proyeccion):
        factores = [m["factor_penetracion"] for m in proyeccion["proyeccion_12_meses"]]
        assert factores == sorted(factores)

    def test_el_primer_mes_vende_menos_que_el_tercero(self, proyeccion):
        # Consecuencia observable de la rampa sobre las unidades proyectadas.
        meses = proyeccion["proyeccion_12_meses"]
        assert meses[0]["unidades_predichas"] < meses[2]["unidades_predichas"]


class TestElasticidadPrecio:
    """Ajuste de la demanda heredada por la distancia de precio contra los pares."""

    def test_un_precio_igual_al_de_los_pares_no_altera_la_demanda(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, valor_unitario=10000.0)
        factor = resultado["producto_nuevo"]["factor_elasticidad_precio"]
        assert factor == pytest.approx(1.0, abs=0.05)

    def test_un_precio_mas_alto_que_el_de_los_pares_reduce_la_demanda(
        self, motor_ajustado
    ):
        resultado = lanzamiento(motor_ajustado, valor_unitario=40000.0)
        assert resultado["producto_nuevo"]["factor_elasticidad_precio"] < 1.0

    def test_un_precio_mas_bajo_que_el_de_los_pares_aumenta_la_demanda(
        self, motor_ajustado
    ):
        resultado = lanzamiento(motor_ajustado, valor_unitario=2500.0)
        assert resultado["producto_nuevo"]["factor_elasticidad_precio"] > 1.0

    def test_el_factor_nunca_baja_del_sesenta_por_ciento(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, valor_unitario=50_000_000.0)
        assert resultado["producto_nuevo"]["factor_elasticidad_precio"] >= 0.6

    def test_el_factor_nunca_supera_el_ciento_cuarenta_por_ciento(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, valor_unitario=1.0)
        assert resultado["producto_nuevo"]["factor_elasticidad_precio"] <= 1.4


class TestEstacionalidadHeredada:

    def test_sin_historia_estacional_el_factor_es_neutro(self, proyeccion):
        factores = {
            m["factor_estacional_aplicado"] for m in proyeccion["proyeccion_12_meses"]
        }
        assert factores == {1.0}

    def test_hereda_el_perfil_estacional_de_los_gemelos(self):
        estacionalidad = {(f"p{i}", 12): 300.0 for i in range(1, 5)}
        resultado = lanzamiento(motor(estacionalidad=estacionalidad))

        diciembre = next(
            m for m in resultado["proyeccion_12_meses"] if m["mes"] == 12
        )
        enero = resultado["proyeccion_12_meses"][0]
        assert diciembre["factor_estacional_aplicado"] > enero["factor_estacional_aplicado"]

    def test_el_factor_estacional_esta_acotado_por_arriba(self):
        estacionalidad = {(f"p{i}", 12): 100_000.0 for i in range(1, 5)}
        resultado = lanzamiento(motor(estacionalidad=estacionalidad))

        factores = [m["factor_estacional_aplicado"] for m in resultado["proyeccion_12_meses"]]
        assert max(factores) <= 1.40

    def test_el_factor_estacional_esta_acotado_por_abajo(self):
        estacionalidad = {(f"p{i}", 12): 0.01 for i in range(1, 5)}
        resultado = lanzamiento(motor(estacionalidad=estacionalidad))

        factores = [m["factor_estacional_aplicado"] for m in resultado["proyeccion_12_meses"]]
        assert min(factores) >= 0.70


class TestProyeccion:

    def test_proyecta_doce_meses_por_defecto(self, proyeccion):
        assert len(proyeccion["proyeccion_12_meses"]) == 12

    def test_respeta_el_horizonte_solicitado(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, horizonte_meses=6)
        assert len(resultado["proyeccion_12_meses"]) == 6

    def test_arranca_en_el_mes_indicado(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, anio_inicio=2026, mes_inicio=9)
        assert (resultado["proyeccion_12_meses"][0]["anio"],
                resultado["proyeccion_12_meses"][0]["mes"]) == (2026, 9)

    def test_cruza_correctamente_el_cambio_de_anio(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, anio_inicio=2026, mes_inicio=11)

        meses = [(m["anio"], m["mes"]) for m in resultado["proyeccion_12_meses"]]
        assert meses[:4] == [(2026, 11), (2026, 12), (2027, 1), (2027, 2)]

    def test_avanza_mes_a_mes_sin_saltos_ni_repeticiones(self, proyeccion):
        meses = [(m["anio"], m["mes"]) for m in proyeccion["proyeccion_12_meses"]]
        assert len(set(meses)) == len(meses)
        assert meses == sorted(meses)

    def test_el_monto_proyectado_es_las_unidades_por_el_precio(self, proyeccion):
        for mes in proyeccion["proyeccion_12_meses"]:
            assert mes["monto_predicho"] == pytest.approx(
                mes["unidades_predichas"] * 10000.0, rel=1e-6
            )

    def test_el_intervalo_de_confianza_encierra_a_la_prediccion(self, proyeccion):
        for mes in proyeccion["proyeccion_12_meses"]:
            assert mes["intervalo_confianza_inf"] <= mes["unidades_predichas"]
            assert mes["unidades_predichas"] <= mes["intervalo_confianza_sup"]

    def test_el_intervalo_de_confianza_es_de_mas_menos_veinte_por_ciento(self, proyeccion):
        # Banda fija, no derivada de la varianza del ensamble: es una estimacion
        # preliminar y la interfaz la presenta como tal.
        mes = proyeccion["proyeccion_12_meses"][5]
        assert mes["intervalo_confianza_inf"] == pytest.approx(
            mes["unidades_predichas"] * 0.80, abs=0.01
        )
        assert mes["intervalo_confianza_sup"] == pytest.approx(
            mes["unidades_predichas"] * 1.20, abs=0.01
        )

    def test_nunca_proyecta_unidades_negativas(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, valor_unitario=50_000_000.0)
        assert all(m["unidades_predichas"] >= 0 for m in resultado["proyeccion_12_meses"])

    def test_la_demanda_base_es_de_al_menos_una_unidad(self, motor_ajustado):
        resultado = lanzamiento(motor_ajustado, valor_unitario=50_000_000.0)
        assert resultado["producto_nuevo"]["unidades_base_estimadas"] >= 1.0


class TestNivelDeConfianza:
    """La etiqueta que la interfaz le muestra al analista.

    Importa que sea honesta: una proyeccion transferida desde gemelos flojos no
    puede presentarse con la misma autoridad que una respaldada por un clon.
    """

    def test_declara_confianza_alta_cuando_los_gemelos_son_muy_afines(self):
        catalogo = [producto(f"clon{i}", 10, 1) for i in range(4)]
        resultado = lanzamiento(motor(catalogo))

        assert resultado["producto_nuevo"]["nivel_confianza"] == "ALTA"
        assert resultado["producto_nuevo"]["confianza_score_pct"] >= 75.0

    def test_declara_confianza_preliminar_cuando_los_gemelos_son_lejanos(self):
        catalogo = [
            producto(
                f"lejano{i}", 99, 9,
                id_marca=f"M{i}", valor_unitario=900_000.0, contenido_neto=1.0,
                unidad_medida="kg", descripcion="herramienta electrica taladro",
            )
            for i in range(4)
        ]
        resultado = lanzamiento(motor(catalogo))

        assert resultado["producto_nuevo"]["nivel_confianza"] == "PRELIMINAR"
        assert resultado["producto_nuevo"]["confianza_score_pct"] < 60.0

    def test_la_etiqueta_se_corresponde_con_el_puntaje(self, proyeccion):
        puntaje = proyeccion["producto_nuevo"]["confianza_score_pct"]
        etiqueta = proyeccion["producto_nuevo"]["nivel_confianza"]

        esperada = "ALTA" if puntaje >= 75.0 else ("MODERADA" if puntaje >= 60.0 else "PRELIMINAR")
        assert etiqueta == esperada

    def test_la_similitud_reportada_nunca_supera_el_cien_por_ciento(self, proyeccion):
        # Los pesos de afinidad suman 1.0, de modo que el puntaje es una
        # proporcion y no puede pasarse de escala al presentarse como porcentaje.
        for par in proyeccion["productos_similares_gemelos"]:
            assert 10.0 <= par["similitud_pct"] <= 100.0


class TestContratoDeLaRespuesta:
    """La vista de arranque en frio del frontend consume esta forma exacta."""

    def test_informa_exito(self, proyeccion):
        assert proyeccion["status"] == "success"

    def test_expone_las_tres_secciones_que_consume_la_interfaz(self, proyeccion):
        assert set(proyeccion) >= {
            "producto_nuevo",
            "productos_similares_gemelos",
            "proyeccion_12_meses",
        }

    def test_devuelve_el_producto_nuevo_con_su_diagnostico(self, proyeccion):
        assert set(proyeccion["producto_nuevo"]) >= {
            "cluster_asignado",
            "unidades_base_estimadas",
            "factor_elasticidad_precio",
            "nivel_confianza",
            "confianza_score_pct",
        }

    def test_cada_gemelo_trae_lo_necesario_para_justificar_la_transferencia(
        self, proyeccion
    ):
        # El analista tiene que poder ver de donde sale la proyeccion: es el
        # componente de explicabilidad del arranque en frio.
        for par in proyeccion["productos_similares_gemelos"]:
            assert set(par) >= {
                "sku",
                "descripcion",
                "categoria",
                "marca",
                "ventas_promedio_mensual",
                "similitud_pct",
            }

    def test_los_valores_numericos_son_tipos_nativos_y_no_de_numpy(self, proyeccion):
        par = proyeccion["productos_similares_gemelos"][0]
        assert isinstance(par["ventas_promedio_mensual"], float)
        assert isinstance(par["cluster_id"], int)
        assert not isinstance(par["ventas_promedio_mensual"], np.floating)
