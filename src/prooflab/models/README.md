# M04 model usage

Models consume a finite numeric pandas DataFrame and canonical labels (-1, 0,
1). Series labels must have exactly the same index as their feature rows; NumPy
labels are positional. Feature names must be unique. Prediction accepts reordered
columns but rejects missing or additional columns. Probability columns follow
`model.classes_`, which contains only classes observed during training.

The wrapper validates input schemas; the caller must establish chronological
partitions and purge labels whose future observations cross partition boundaries.
Do not pass blind data to `fit`. The integrated training pipeline is still pending.

## Implemented models

| Model | Configuration | Preprocessing |
| --- | --- | --- |
| `baselines.RandomClassifier` | `strategy`, `random_state` | Identity |
| `baselines.MajorityClassifier` | None | Identity |
| `logistic.LogisticRegressionBaseline` | `LogisticRegressionConfig` | Training-fitted standardization |
| `xgboost.XGBoostModel` | `XGBoostConfig` | Identity pipeline |
| `neural.NeuralNetworkModel` | `NeuralNetworkConfig` | Training-fitted standardization |

ML models require `pip install -e ".[ml]"`. Core baselines do not require the ML
extra. Configuration objects reject unknown settings and invalid parameter ranges;
`get_params()` returns their versioned settings for persistence. Random and
Majority support a single training class; the other implemented models require at
least two. Majority ties select the smallest canonical class. Its probability
output is the empirical training frequency, not one-hot certainty. Random outputs
are successive seeded draws; refitting resets the sequence, and probability
queries do not advance it.

```python
from prooflab.models.logistic import LogisticRegressionBaseline, LogisticRegressionConfig

model = LogisticRegressionBaseline(LogisticRegressionConfig(c_param=1.0))
model.fit(train_features, train_labels)
probabilities = model.predict_proba(validation_features)
```

Logistic regression and XGBoost ignore validation values during fitting; they do
not perform automatic tuning. Neural training requires `val_data=(features,
labels)`, rejects unseen validation classes, and selects a checkpoint using
validation cross-entropy, configured patience, and minimum improvement. It restores
that checkpoint and disables dropout for inference. Training uses CPU, preserves
batch order, and restores the surrounding CPU random-generator state. Seeds make
repeated fits reproducible in the tested environment; exact results across library
versions or hardware are not guaranteed.

## Native research artifacts

`artifacts.save_artifact(model, path, training=..., feature_metadata=...)` writes a
new `.plmodel` file containing a JSON manifest and a joblib model payload. The
payload preserves weights, fitted preprocessing, and model state, including the
Random classifier's generator state. The manifest records feature order, numeric
schema, feature definitions, canonical classes, configuration, dataset checksum,
training intervals, dependency versions, and neural training history where present.
Supply `TrainingMetadata` and ordered `FeatureMetadata` objects describing the
actual training run. Existing files are never overwritten.

`inspect_artifact(path)` reads the manifest without unpickling. For an artifact
created by a trusted local research workflow, use
`load_artifact(path, trusted=True).model` and call the normal prediction methods.
Loading rejects incompatible versions, incomplete schemas, and corrupt payloads.
These are integrity checks, not authentication: joblib can execute Python during
loading, so never assert trust for an unknown artifact. `.plb` strategy packages
are explicitly unsupported by this loader.

Artifacts accept generated feature frames for inference. Raw market-data feature
generation remains the responsibility of the existing feature engine and its
recorded feature definitions. No strategy-package import, live execution, or
ensemble behavior is provided here.
