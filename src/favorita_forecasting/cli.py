"""CLI entry point for the Favorita Forecasting pipeline."""

from typing import Annotated

import typer

app = typer.Typer(
    name="favorita",
    help="Favorita Forecasting - Grocery sales prediction pipeline.",
)


@app.command()
def prepare() -> None:
    """Load, clean, merge, and build features from raw data."""
    from favorita_forecasting.features.pipeline import run_feature_pipeline

    typer.echo("Running feature pipeline...")
    run_feature_pipeline()
    typer.echo("Feature pipeline complete. Output in data/processed/")


@app.command()
def train(
    model: Annotated[str, typer.Option(help="Model type: lightgbm or xgboost")] = "lightgbm",
    skip_cv: Annotated[bool, typer.Option(help="Skip cross-validation, train directly")] = False,
) -> None:
    """Train a model with expanding window cross-validation."""
    from favorita_forecasting.pipeline.train_pipeline import run_training

    typer.echo(f"Training {model} model (skip_cv={skip_cv})...")
    run_training(model_type=model, skip_cv=skip_cv)
    typer.echo("Training complete.")


@app.command()
def predict(
    model_path: Annotated[str, typer.Option(help="Path to trained model")] = "",
) -> None:
    """Generate predictions on the test set."""
    from favorita_forecasting.pipeline.predict_pipeline import run_prediction

    typer.echo("Generating predictions...")
    run_prediction(model_path=model_path or None)
    typer.echo("Predictions saved to data/submissions/")


@app.command()
def tune(
    model: Annotated[str, typer.Option(help="Model type to tune")] = "lightgbm",
    n_trials: Annotated[int, typer.Option(help="Number of Optuna trials")] = 100,
    timeout: Annotated[int | None, typer.Option(help="Timeout in seconds (None=unlimited)")] = None,
) -> None:
    """Run Optuna hyperparameter optimization."""
    from favorita_forecasting.models.hyperopt import run_hyperopt

    timeout_str = f", timeout={timeout}s" if timeout else ""
    typer.echo(f"Tuning {model} with {n_trials} trials{timeout_str}...")
    run_hyperopt(model_type=model, n_trials=n_trials, timeout=timeout)
    typer.echo("Tuning complete.")


@app.command()
def evaluate(
    model_path: Annotated[str, typer.Option(help="Path to trained model")] = "",
) -> None:
    """Evaluate a trained model on validation data."""
    from favorita_forecasting.evaluation.analysis import run_evaluation

    typer.echo("Evaluating model...")
    run_evaluation(model_path=model_path or None)
    typer.echo("Evaluation complete.")


if __name__ == "__main__":
    app()
