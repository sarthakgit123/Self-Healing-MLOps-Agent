# Self-Healing-MLOps-Agent

## Phase 3: LangGraph workflow

Run the orchestration from the project root with:

```powershell
.\.venv\Scripts\python.exe -m src.workflow
```

The compiled graph validates the CSV, applies the existing Phase 2 cleaning and
features, trains and evaluates the existing median baseline and XGBoost model,
then applies a configurable validation error gate. The default gate is MAE at
most $2.00. This is a documented margin above the Phase 2 validation MAE of
about $1.81 and remains far below the baseline MAE of about $6.52. Override it
when calling `run_workflow(quality_metric="mae", quality_threshold=...)`.

Passing candidates are saved as `taxi_fare_model.joblib`,
`model_features.json`, and `metrics.json` under a unique
`models/candidates/<run_id>/` directory. A rejected candidate is not saved and
does not replace the existing Phase 2 model. The returned workflow state
contains the final status, metrics, artifact paths (on success), and an error
message when a stage fails.

## Phase 4: Local MLflow experiment tracking

Direct `run_training()` calls and LangGraph `run_workflow()` calls each create
one run in the `NYC Taxi Fare Prediction` experiment. Tracking is stored locally
in `mlruns/`; there is no remote tracking server or Model Registry. Runs record
the XGBoost parameters, sample and split metadata, baseline and candidate
MAE/RMSE/R², training duration, evaluation JSON, feature schema, and the
trained XGBoost model for every completed training attempt. Rejected workflow
candidates still have their metrics, model, and rejection status recorded in
their MLflow run, but are not written to `models/candidates/`. The existing
local candidate artifact behavior is unchanged.

Launch the local UI from the project root in a separate PowerShell window:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` and select the `NYC Taxi Fare Prediction`
experiment. Stop the UI with Ctrl+C. The `mlruns/` directory is ignored by Git.
