"""DVC-verified training and external evaluation for Arabic hate speech."""

from datetime import datetime
import re

from airflow.sdk import Param, dag, get_current_context, task


@dag(
    dag_id="arabic_hate_speech_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    params={
        "dataset": Param(
            "data/raw/ar_dataset.csv", type="string", minLength=1,
            description="Project-relative CSV inside data/raw with an adjacent .dvc file",
        ),
        "mode": Param("train", type="string", enum=["train", "evaluate"]),
        "model_id": Param(
            "", type="string", pattern=r"^(|m-[0-9a-f]{32})$",
            description="Optional MLflow logged model ID for evaluate mode; blank uses the deployed model",
        ),
    },
    tags=["mlops", "arabic", "hate-speech", "dvc"],
)
def ml_pipeline():
    @task(task_id="verify_dvc_dataset")
    def verify() -> dict:
        from src.data.verify_dvc import verify_dataset
        from src.pipeline.paths import RAW_DIR, project_path, run_key

        context = get_current_context()
        params = context["params"]
        mode = params["mode"]
        model_id = params.get("model_id") or None
        if mode not in {"train", "evaluate"}:
            raise ValueError("mode must be train or evaluate")
        if model_id and (mode != "evaluate" or not re.fullmatch(r"m-[0-9a-f]{32}", model_id)):
            raise ValueError("model_id is only valid in evaluate mode")
        dataset = project_path(params["dataset"], RAW_DIR, file=True)
        checksum = verify_dataset(dataset)
        return {
            "dataset": str(dataset), "mode": mode, "model_id": model_id,
            "dataset_md5": checksum, "airflow_run_id": context["run_id"],
            "run_key": run_key(context["run_id"]),
        }

    @task.branch(task_id="select_mode")
    def select_mode(config: dict) -> str:
        return "preprocess_data" if config["mode"] == "train" else "preprocess_external"

    @task(task_id="preprocess_data")
    def preprocess_data(config: dict) -> str:
        from src.data.preprocess import preprocess_arabic
        from src.pipeline.paths import PROCESSED_DIR

        output_dir = PROCESSED_DIR / "runs" / config["run_key"]
        preprocess_arabic(config["dataset"], output_dir, mode="train")
        return str(output_dir)

    @task(task_id="split_data")
    def split_data(config: dict, data_dir: str) -> str:
        from pathlib import Path
        from src.data.split_data import split_data as split

        split(Path(data_dir) / "ar_ready.csv", data_dir)
        return data_dir

    @task(task_id="train_model")
    def train_model(config: dict, data_dir: str) -> str:
        from src.models.train import train_model as train
        from src.pipeline.paths import MODELS_DIR

        return train(
            data_dir=data_dir,
            model_dir=MODELS_DIR / "runs" / config["run_key"],
            dataset_path=config["dataset"],
            dataset_md5=config["dataset_md5"],
            airflow_run_id=config["airflow_run_id"],
        )

    @task(task_id="evaluate_model")
    def evaluate_model(config: dict, data_dir: str, mlflow_run_id: str) -> str:
        from pathlib import Path
        from src.models.evaluate import evaluate_model as evaluate
        from src.pipeline.paths import ARTIFACTS_DIR, MODELS_DIR

        model_dir = MODELS_DIR / "runs" / config["run_key"]
        return evaluate(
            data_path=Path(data_dir) / "test.csv",
            model_path=model_dir / "tfidf_logreg.joblib",
            report_dir=ARTIFACTS_DIR / "runs" / config["run_key"],
            run_id_path=model_dir / "run_id.txt",
            mode="train",
            dataset_path=config["dataset"],
            dataset_md5=config["dataset_md5"],
            airflow_run_id=config["airflow_run_id"],
            run_id=mlflow_run_id,
        )

    @task(task_id="preprocess_external")
    def preprocess_external(config: dict) -> str:
        from src.data.preprocess import preprocess_arabic
        from src.pipeline.paths import PROCESSED_DIR

        output_dir = PROCESSED_DIR / "runs" / config["run_key"]
        preprocess_arabic(config["dataset"], output_dir, mode="evaluate")
        return str(output_dir / "ar_ready.csv")

    @task(task_id="evaluate_external")
    def evaluate_external(config: dict, data_path: str) -> str:
        from src.models.evaluate import evaluate_model as evaluate
        from src.pipeline.paths import ARTIFACTS_DIR

        return evaluate(
            data_path=data_path,
            report_dir=ARTIFACTS_DIR / "runs" / config["run_key"],
            mode="evaluate",
            dataset_path=config["dataset"],
            dataset_md5=config["dataset_md5"],
            airflow_run_id=config["airflow_run_id"],
            model_id=config["model_id"],
        )

    config = verify()
    branch = select_mode(config)
    prepared = preprocess_data(config)
    branch >> prepared
    split = split_data(config, prepared)
    trained = train_model(config, split)
    evaluate_model(config, split, trained)
    external = preprocess_external(config)
    branch >> external
    evaluate_external(config, external)


ml_pipeline()
