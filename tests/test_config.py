"""Pruebas del cargador de configuraciones.

La version anterior de este archivo afirmaba que el proyecto se llamaba
"tesis_ia" y que la arquitectura del modelo era "dense". Ambas cosas dejaron de
ser ciertas cuando el proyecto paso de la plantilla generica al modelo real, y
las pruebas quedaron en rojo sin que nadie lo notara. Lo que se verifica aca es
la mecanica de fusion, que es codigo vivo -la usa la linea de comandos-, y la
correspondencia entre la configuracion declarada y el modelo que efectivamente
se entrena.
"""

import pytest
import yaml

from tesis_ia.config import (
    config_dir,
    deep_merge,
    load_config,
    load_default_config,
    load_experiment_config,
    load_yaml,
    project_root,
)


class TestDeepMerge:
    """Fusion recursiva de configuraciones.

    Es lo que permite que un experimento redefina un solo parametro sin repetir
    el resto del apilado. Un error aca no rompe nada de forma visible: entrena
    con una configuracion distinta a la que dice el archivo.
    """

    def test_incorpora_las_claves_que_no_estaban(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_la_configuracion_de_encima_gana_sobre_la_de_abajo(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_desciende_a_los_diccionarios_anidados_en_vez_de_reemplazarlos(self):
        fusion = deep_merge(
            {"model": {"family": "random_forest", "n_estimators": 100, "max_depth": 12}},
            {"model": {"n_estimators": 500}},
        )
        assert fusion["model"] == {
            "family": "random_forest",
            "n_estimators": 500,
            "max_depth": 12,
        }

    def test_fusiona_a_cualquier_profundidad(self):
        fusion = deep_merge(
            {"a": {"b": {"c": {"d": 1, "e": 2}}}},
            {"a": {"b": {"c": {"d": 9}}}},
        )
        assert fusion["a"]["b"]["c"] == {"d": 9, "e": 2}

    def test_las_listas_se_reemplazan_enteras_y_no_se_concatenan(self):
        # train_years es una lista: concatenarla al redefinirla meteria anios
        # repetidos en el conjunto de entrenamiento.
        fusion = deep_merge({"data": {"train_years": [2021, 2022]}},
                            {"data": {"train_years": [2023]}})
        assert fusion["data"]["train_years"] == [2023]

    def test_un_valor_simple_reemplaza_a_un_diccionario(self):
        assert deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}

    def test_un_diccionario_reemplaza_a_un_valor_simple(self):
        assert deep_merge({"a": 5}, {"a": {"b": 1}}) == {"a": {"b": 1}}

    def test_no_altera_la_configuracion_base(self):
        # Se la llama en cadena sobre el mismo acumulador: si mutara la entrada,
        # cada archivo del apilado contaminaria al anterior.
        base = {"model": {"n_estimators": 100}}
        deep_merge(base, {"model": {"n_estimators": 500}})
        assert base == {"model": {"n_estimators": 100}}

    def test_no_altera_la_configuracion_superpuesta(self):
        override = {"model": {"n_estimators": 500}}
        deep_merge({"model": {"n_estimators": 100}}, override)
        assert override == {"model": {"n_estimators": 500}}

    def test_el_resultado_no_comparte_estructuras_con_las_entradas(self):
        base = {"model": {"metrics": ["mape"]}}
        fusion = deep_merge(base, {})
        fusion["model"]["metrics"].append("rmse")
        assert base["model"]["metrics"] == ["mape"]


class TestLoadYaml:

    def test_resuelve_las_rutas_relativas_desde_la_raiz_del_proyecto(self):
        assert load_yaml("config/model.yaml") == load_yaml(config_dir() / "model.yaml")

    def test_acepta_una_ruta_absoluta(self):
        assert "model" in load_yaml(project_root() / "config" / "config.yaml")

    def test_un_archivo_vacio_produce_una_configuracion_vacia(self, tmp_path):
        vacio = tmp_path / "vacio.yaml"
        vacio.write_text("", encoding="utf-8")
        assert load_yaml(vacio) == {}

    def test_rechaza_un_yaml_que_no_sea_un_diccionario(self, tmp_path):
        # Una lista en la raiz haria fallar a deep_merge mucho mas adelante y con
        # un mensaje que no apunta al archivo culpable.
        lista = tmp_path / "lista.yaml"
        lista.write_text("- uno\n- dos\n", encoding="utf-8")

        with pytest.raises(ValueError, match="debe ser un diccionario"):
            load_yaml(lista)

    def test_el_mensaje_de_error_nombra_el_archivo_culpable(self, tmp_path):
        lista = tmp_path / "culpable.yaml"
        lista.write_text("- uno\n", encoding="utf-8")

        with pytest.raises(ValueError, match="culpable.yaml"):
            load_yaml(lista)


class TestApiladoDeConfiguraciones:

    def test_la_configuracion_por_defecto_reune_los_cinco_dominios(self):
        config = load_default_config()
        for seccion in ("project", "data", "model", "training", "evaluation"):
            assert seccion in config, f"falta la seccion {seccion}"

    def test_el_experimento_arma_su_configuracion_desde_el_apilado_declarado(self):
        config = load_experiment_config("config/experiments/baseline.yaml")

        assert config["experiment"]["name"] == "baseline_dense_v1"
        assert "model" in config
        assert "training" in config

    def test_el_ultimo_archivo_del_apilado_tiene_la_ultima_palabra(self, tmp_path):
        primero = tmp_path / "primero.yaml"
        segundo = tmp_path / "segundo.yaml"
        experimento = tmp_path / "exp.yaml"
        primero.write_text("model:\n  n_estimators: 100\n  max_depth: 12\n", encoding="utf-8")
        segundo.write_text("model:\n  n_estimators: 500\n", encoding="utf-8")
        experimento.write_text(
            yaml.safe_dump(
                {
                    "experiment": {"name": "prueba"},
                    "config_stack": [str(primero), str(segundo)],
                }
            ),
            encoding="utf-8",
        )

        config = load_experiment_config(experimento)

        assert config["model"]["n_estimators"] == 500
        assert config["model"]["max_depth"] == 12

    def test_un_experimento_sin_apilado_conserva_su_identidad(self, tmp_path):
        experimento = tmp_path / "suelto.yaml"
        experimento.write_text(
            yaml.safe_dump({"experiment": {"name": "suelto"}}), encoding="utf-8"
        )

        config = load_experiment_config(experimento)

        assert config["experiment"]["name"] == "suelto"


class TestLoadConfig:

    def test_sin_ruta_devuelve_la_configuracion_por_defecto(self):
        assert load_config() == load_default_config()

    def test_una_ruta_bajo_experiments_se_resuelve_como_apilado(self):
        config = load_config("config/experiments/baseline.yaml")
        assert "experiment" in config
        assert "model" in config

    def test_una_ruta_comun_se_carga_tal_cual(self):
        assert load_config("config/model.yaml") == load_yaml("config/model.yaml")


class TestCorrespondenciaConElProyectoReal:
    """Ata la configuracion declarada al sistema implementado.

    Es la clase de desincronizacion que dejo las pruebas anteriores en rojo: el
    codigo avanzo y los archivos que lo describen quedaron atras.
    """

    def test_el_modelo_declarado_es_el_champion_del_torneo(self):
        config = load_config("config/config.yaml")
        assert config["model"]["family"] == "random_forest"

    def test_las_metricas_declaradas_son_las_que_reporta_la_tesis(self):
        config = load_config("config/config.yaml")
        assert "mape" in config["evaluation"]["metrics"]
        assert "accuracy" not in config["evaluation"]["metrics"]

    def test_el_objetivo_declarado_son_las_unidades_vendidas(self):
        config = load_config("config/config.yaml")
        assert config["data"]["target_column"] == "unidades_vendidas"

    def test_entrenamiento_y_validacion_no_comparten_anios(self):
        # Solapar las ventanas seria fuga temporal: el modelo validaria sobre
        # periodos que ya vio durante el ajuste.
        config = load_config("config/config.yaml")
        train = set(config["data"]["train_years"])
        test = set(config["data"]["test_years"])
        assert train & test == set()

    def test_la_validacion_es_posterior_al_entrenamiento(self):
        config = load_config("config/config.yaml")
        assert min(config["data"]["test_years"]) > max(config["data"]["train_years"])

    def test_la_semilla_esta_fijada_para_que_las_corridas_sean_reproducibles(self):
        config = load_config("config/config.yaml")
        assert config["project"]["seed"] == 42
