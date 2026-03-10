# ===============================
# Model configuration registry
# ===============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ===============================
# Local imports
# ===============================
from src.config.settings import SEED
from src.tuning.search_spaces import (
    suggest_rf_params,
    suggest_lr_params,
    suggest_knn_params,
    suggest_mlp_params,
    suggest_xgb_params,
)


MODEL_CONFIGS = {
    "random_forest": {
        "model_class": RandomForestClassifier,
        "suggest_fn": suggest_rf_params,
        "fixed_params": {
            "random_state": SEED,
            "n_jobs": -1,
        },
    },
    "logreg": {
        "model_class": LogisticRegression,
        "suggest_fn": suggest_lr_params,
        "fixed_params": {
            "max_iter": 1000,
            "random_state": SEED,
        },
    },
    "knn": {
        "model_class": KNeighborsClassifier,
        "suggest_fn": suggest_knn_params,
        "fixed_params": {},
    },
    "mlp": {
        "model_class": MLPClassifier,
        "suggest_fn": suggest_mlp_params,
        "fixed_params": {
            "max_iter": 150,
            "random_state": SEED,
            "early_stopping": True,
            "n_iter_no_change": 10,
        },
    },
    "xgb": {
        "model_class": XGBClassifier,
        "suggest_fn": suggest_xgb_params,
        "fixed_params": {
            "random_state": SEED,
            "use_label_encoder": False,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        },
    },
}