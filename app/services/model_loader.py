"""
Model Loader — Singleton pattern.
Loads models once at startup. Records ALL errors (model + conformal).
"""
import os
import joblib
import pandas as pd
from config import MODEL_CONFIG


class ModelLoader:
    _models        = {}
    _conformal     = {}
    _errors        = {}
    _loaded        = False

    @classmethod
    def load_all(cls):
        if cls._loaded:
            return cls._models, cls._conformal, cls._errors

        cls._models = {}; cls._conformal = {}; cls._errors = {}

        for key, cfg in MODEL_CONFIG.items():
            # — Model —
            if not os.path.exists(cfg["model_path"]):
                cls._errors[key] = f"{cfg['name']} model file missing: {cfg['model_path']}"
                continue
            try:
                cls._models[key] = joblib.load(cfg["model_path"])
            except Exception as e:
                cls._errors[key] = f"{cfg['name']} model load error: {e}"
                continue

            # — Conformal CSV —
            if os.path.exists(cfg["conformal_path"]):
                try:
                    cls._conformal[key] = pd.read_csv(cfg["conformal_path"])
                except Exception as e:
                    cls._conformal[key] = None
                    cls._errors[f"{key}_conformal"] = f"{cfg['name']} conformal CSV error: {e}"
            else:
                cls._conformal[key] = None
                cls._errors[f"{key}_conformal"] = f"{cfg['name']} conformal CSV not found."

        cls._loaded = True
        return cls._models, cls._conformal, cls._errors

    @classmethod
    def get_model(cls, key):
        cls.load_all()
        if key in cls._errors:
            raise FileNotFoundError(cls._errors[key])
        if key not in cls._models:
            raise FileNotFoundError(f"Model '{key}' not available.")
        return cls._models[key]

    @classmethod
    def get_conformal(cls, key):
        cls.load_all()
        return cls._conformal.get(key)

    @classmethod
    def get_all_errors(cls):
        cls.load_all()
        return cls._errors
