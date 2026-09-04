"""Neural training, checkpoint restoration, and preprocessing isolation."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch

from prooflab.models.neural import NeuralNetworkConfig, NeuralNetworkModel


@pytest.fixture
def data():
    x = pd.DataFrame({"signal": [-3., -2., -1., 1., 2., 3.]})
    return x, np.array([-1, -1, -1, 1, 1, 1])


def test_real_fit_repeatability_and_rng_isolation(data):
    x, y = data
    config = NeuralNetworkConfig(hidden_units=(4, 3), epochs=3, batch_size=4)
    before = torch.random.get_rng_state().clone()
    first = NeuralNetworkModel(config).fit(x, y, (x + 0.1, y))
    assert torch.equal(before, torch.random.get_rng_state())
    second = NeuralNetworkModel(config).fit(x, y, (x + 0.1, y))
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    np.testing.assert_array_equal(first.predict_proba(x), first.predict_proba(x))
    assert not first.network.training
    assert first.classes_ == [-1, 1]
    assert len(first.network) == 7
    assert first.get_params() == config.model_dump()
    np.testing.assert_allclose(first.preprocessor.mean_, x.mean().to_numpy())


def test_best_checkpoint_restored(data, monkeypatch):
    x, y = data
    model = NeuralNetworkModel(NeuralNetworkConfig(hidden_units=(4,), epochs=9, patience=2))
    states, seen_validation = [], []
    losses = iter([0.1, 0.2, 0.3])

    def validation_loss(features, labels):
        states.append(deepcopy(model.network.state_dict()))
        seen_validation.append(features.clone())
        return next(losses)

    monkeypatch.setattr(model, "_validation_loss", validation_loss)
    model.fit(x, y, (x + 100, y))
    assert model.best_epoch_ == 1
    assert model.stopped_epoch_ == 3
    for name, parameter in model.network.state_dict().items():
        assert torch.equal(parameter, states[0][name])
    assert any(not torch.equal(states[0][name], states[2][name]) for name in states[0])
    np.testing.assert_allclose(model.preprocessor.mean_, [0])
    np.testing.assert_allclose(seen_validation[0].numpy(),
                               model.preprocessor.transform(x + 100), rtol=1e-6)


def test_required_validation_and_unseen_class(data):
    x, y = data
    with pytest.raises(ValueError, match="Validation data is required"):
        NeuralNetworkModel().fit(x, y)
    with pytest.raises(ValueError, match="absent from training"):
        NeuralNetworkModel().fit(x, y, (x, np.zeros(len(x))))
    with pytest.raises(ValueError, match="two training classes"):
        NeuralNetworkModel().fit(x, np.zeros(len(x)), (x, y))


@pytest.mark.parametrize("params", [
    {"hidden_units": ()}, {"hidden_units": (0,)}, {"dropout": 1}, {"epochs": 0},
    {"learning_rate": 0}, {"weight_decay": -1}, {"batch_size": 0}, {"patience": 0},
])
def test_invalid_config(params):
    with pytest.raises(ValueError):
        NeuralNetworkConfig(**params)


def test_nonfinite_validation_loss_rejected(data, monkeypatch):
    x, y = data
    model = NeuralNetworkModel(NeuralNetworkConfig(hidden_units=(4,), epochs=1))
    monkeypatch.setattr(model, "_validation_loss", lambda *_: float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(x, y, (x, y))
    assert not model.is_fitted
