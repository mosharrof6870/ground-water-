"""
Model Loader Service.
Responsible for loading saved scikit-learn models and conformal artifacts once at startup.
Never silently substitutes missing models or fabricates fallback predictions.
"""
import os
import joblib
import pandas as pd
from config import MODEL_CONFIG


class ModelLoader:
    _models = {}
    _conformal_data = {}
    _loaded = False
    _load_errors = {}

    @classmethod
    def load_all_models(cls):
        """Loads Ni and Cd models and conformal artifacts into memory once."""
        if cls._loaded:
            return cls._models, cls._conformal_data, cls._load_errors

        cls._models = {}
        cls._conformal_data = {}
        cls._load_errors = {}

        for key, cfg in MODEL_CONFIG.items():
            model_path = cfg["model_path"]
            conformal_path = cfg["conformal_path"]

            # 1. Load trained model object
            if not os.path.exists(model_path):
                cls._load_errors[key] = f"{cfg['name']} model file not found at '{model_path}'."
                continue
            
            try:
                model_obj = joblib.load(model_path)
                cls._models[key] = model_obj
            except Exception as e:
                cls._load_errors[key] = f"{cfg['name']} model could not be loaded: {str(e)}"
                continue

            # 2. Load conformal calibration dataset (if available)
            if os.path.exists(conformal_path):
                try:
                    cdf = pd.read_csv(conformal_path)
                    cls._conformal_data[key] = cdf
                except Exception as e:
                    cls._conformal_data[key] = None
            else:
                cls._conformal_data[key] = None

        cls._loaded = True
        return cls._models, cls._conformal_data, cls._load_errors

    @classmethod
    def get_model(cls, metal_key):
        """Returns loaded model object or raises FileNotFoundError."""
        cls.load_all_models()
        if metal_key in cls._load_errors:
            raise FileNotFoundError(cls._load_errors[metal_key])
        if metal_key not in cls._models:
            raise FileNotFoundError(f"Model for '{metal_key}' is not available.")
        return cls._models[metal_key]

    @classmethod
    def get_conformal_data(cls, metal_key):
        """Returns conformal calibration dataset or None."""
        cls.load_all_models()
        return cls._conformal_data.get(metal_key, None)
