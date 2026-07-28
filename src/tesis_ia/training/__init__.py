"""Orquestacion del pipeline de entrenamiento."""

from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor
from tesis_ia.training.trainer import Trainer

__all__ = ["Trainer", "DbTrainerPredictor"]

