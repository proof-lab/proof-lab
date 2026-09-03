"""CPU dense classifier with validation-only checkpoint selection."""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import StandardScaler
from torch import nn

from prooflab.models.base import BaseModelWrapper


class NeuralNetworkConfig(BaseModel):
    """Explicit architecture and optimization settings for reproducible CPU fits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    hidden_units: tuple[Annotated[int, Field(gt=0, strict=True)], ...] = Field(
        default=(64, 32), min_length=1,
    )
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=64, gt=0, strict=True)
    epochs: int = Field(default=100, gt=0, strict=True)
    weight_decay: float = Field(default=0, ge=0)
    patience: int = Field(default=10, gt=0, strict=True)
    min_delta: float = Field(default=0, ge=0)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)


class NeuralNetworkModel(BaseModelWrapper):
    """Train on supplied training rows; select epochs only with validation loss.

    A validation partition is required. An unseen validation class is rejected
    rather than using future labels to expand the learned class vocabulary.
    Batches retain input order. CPU RNG state is restored after each fit.
    """

    def __init__(self, config: NeuralNetworkConfig | None = None) -> None:
        super().__init__("neural_network")
        self.config = (config or NeuralNetworkConfig()).model_copy(deep=True)
        self.preprocessor = StandardScaler()
        self.network: nn.Sequential | None = None
        self.history_: list[dict[str, float | int]] = []
        self.best_epoch_: int | None = None
        self.stopped_epoch_: int | None = None

    def _build_network(self, input_size: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = input_size
        for units in self.config.hidden_units:
            layers.extend([nn.Linear(previous, units), nn.ReLU(), nn.Dropout(self.config.dropout)])
            previous = units
        layers.append(nn.Linear(previous, len(self.classes_)))
        return nn.Sequential(*layers)

    def _tensor(self, features: pd.DataFrame) -> torch.Tensor:
        values = np.asarray(self.preprocessor.transform(features), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Scaled features overflowed float32.")
        return torch.from_numpy(values)

    def _validation_loss(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        assert self.network is not None
        self.network.eval()
        with torch.inference_mode():
            return float(nn.functional.cross_entropy(self.network(features), labels).item())

    def _fit_internal(
        self, features: pd.DataFrame, labels: np.ndarray,
        val_data: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> None:
        self.history_ = []
        self.best_epoch_ = self.stopped_epoch_ = None
        if len(self.classes_) < 2:
            raise ValueError("Neural network requires at least two training classes.")
        if val_data is None:
            raise ValueError("Validation data is required for early stopping.")
        val_x, val_y = val_data
        if not np.isin(val_y, self.classes_).all():
            raise ValueError("Validation contains a class absent from training.")
        self.preprocessor = StandardScaler().fit(features)
        train_x, validation_x = self._tensor(features), self._tensor(val_x)
        train_y = torch.tensor(np.searchsorted(self.classes_, labels), dtype=torch.long)
        validation_y = torch.tensor(np.searchsorted(self.classes_, val_y), dtype=torch.long)
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(self.config.random_state)
            torch.random.set_rng_state(generator.get_state())
            self.network = self._build_network(features.shape[1])
            optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            best_loss, wait = float("inf"), 0
            best_state: dict[str, Any] | None = None
            for epoch in range(1, self.config.epochs + 1):
                self.network.train()
                total_loss = 0.0
                for start in range(0, len(features), self.config.batch_size):
                    batch_x = train_x[start:start + self.config.batch_size]
                    batch_y = train_y[start:start + self.config.batch_size]
                    optimizer.zero_grad()
                    loss = nn.functional.cross_entropy(self.network(batch_x), batch_y)
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is non-finite.")
                    torch.autograd.backward(loss)
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch_x)
                validation_loss = self._validation_loss(validation_x, validation_y)
                if not np.isfinite(validation_loss):
                    raise ValueError("Validation loss is non-finite.")
                self.history_.append({"epoch": epoch, "train_loss": total_loss / len(features),
                                      "validation_loss": validation_loss})
                self.stopped_epoch_ = epoch
                if validation_loss < best_loss - self.config.min_delta:
                    best_loss = validation_loss
                    best_state = deepcopy(self.network.state_dict())
                    self.best_epoch_ = epoch
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.config.patience:
                        break
            assert best_state is not None
            self.network.load_state_dict(best_state)
            self.network.eval()

    def _predict_internal(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self._predict_proba_internal(features)
        return np.asarray(self.classes_)[probabilities.argmax(axis=1)]

    def _predict_proba_internal(self, features: pd.DataFrame) -> np.ndarray:
        assert self.network is not None
        self.network.eval()
        with torch.inference_mode():
            return np.asarray(torch.softmax(self.network(self._tensor(features)), dim=1).numpy())

    def get_params(self) -> dict[str, Any]:
        return self.config.model_dump()
