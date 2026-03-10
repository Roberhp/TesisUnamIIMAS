from dataclasses import dataclass
from typing import Dict, Any, Optional

import joblib

@dataclass
class TrainingArtifacts:
    vectorizers: Dict[str, Any]
    classical_models: Dict[str, Any]
    label_encoder: Any
    class_names: list
    n_classes: int

    def save(self, path: str, fan_model=None):
        joblib.dump({
            "vectorizers": self.vectorizers,
            "classical_models": self.classical_models,
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "n_classes": self.n_classes
        }, f"{path}/artifacts.pkl")

        if fan_model is not None:
            torch.save(fan_model.state_dict(), f"{path}/fan.pt")