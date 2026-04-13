import argparse
import ast
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install it with: pip install optuna"
    ) from exc


DEFAULT_BASE_ARGS = {
    "dataset": "cameraCOQE_quintuple",
    "pretrained_deberta_name": "microsoft/deberta-v3-base",
    "emb_dim": 768,
    "deberta_feature_dim": 768,
    "hidden_dim": 384,
    "batch_size": 4,
    "epochs": 10,
    "lr": 1e-5,
    "sampling_processes": 0,
    "sentence_loss_weight": 0.2,
    "neg_entity_count": 25,
    "neg_triple_count": 60,
    "eval_match_mode": "both",
}


def build_train_command(train_py: str, base_args: dict, sampled: dict, log_path: str, save_path: str):
    args = {
        **base_args,
        **sampled,
        "log_path": log_path,
        "save_path": save_path,
    }

    cmd = [sys.executable, train_py]
    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def find_latest_run_dir(log_root: str, dataset: str):
    pattern = os.path.join(log_root, dataset, "*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_best_dev_f1_from_csv(run_dir: str):
    """Read max senti_nec_f1_micro from test_dev.csv."""
    csv_path = os.path.join(run_dir, "test_dev.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing expected eval csv: {csv_path}")

    best = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = float(row["senti_nec_f1_micro"])
            except (KeyError, ValueError):
                continue
            if best is None or value > best:
                best = value

    if best is None:
        raise RuntimeError(f"No valid senti_nec_f1_micro values found in {csv_path}")
    return best


def read_best_dev_et5_macro_f1_from_result(run_dir: str, match_view: str = "index"):
    """Read E-T5-MACRO F1 from dev blocks in result/result*.txt.

    match_view:
      - index: read from coqe_metrics_full_index_match
      - span:  read from coqe_metrics_full_span_match
    """
    result_glob = os.path.join(run_dir, "result", "result*.txt")
    files = sorted(glob.glob(result_glob), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No result file found under: {result_glob}")

    target_header = "coqe_metrics_full_index_match" if match_view == "index" else "coqe_metrics_full_span_match"
    best_f1 = None

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if "(dev)" not in line:
                continue

            j = idx + 1
            while j < len(lines) and "No." not in lines[j]:
                if target_header in lines[j]:
                    if j + 1 < len(lines):
                        metric_line = lines[j + 1].strip()
                        try:
                            metric_dict = ast.literal_eval(metric_line)
                            f1 = float(metric_dict["E-T5-MACRO"]["F1"]) * 100.0
                            if best_f1 is None or f1 > best_f1:
                                best_f1 = f1
                        except Exception:
                            pass
                j += 1

    if best_f1 is None:
        raise RuntimeError(
            f"Could not find E-T5-MACRO F1 in dev result blocks (header={target_header})."
        )
    return best_f1


def create_objective(args):
    train_py = os.path.join(args.project_root, "train.py")
    if not os.path.exists(train_py):
        raise FileNotFoundError(f"Cannot find train.py at: {train_py}")

    base_args = dict(DEFAULT_BASE_ARGS)
    base_args["epochs"] = args.epochs
    base_args["batch_size"] = args.batch_size
    if args.lr is not None:
        base_args["lr"] = args.lr

    if args.fixed_params:
        fixed = json.loads(args.fixed_params)
        if not isinstance(fixed, dict):
            raise ValueError("--fixed_params must be a JSON object")
        base_args.update(fixed)

    def objective(trial):
        sampled = {
            "max_role_candidates": trial.suggest_int("max_role_candidates", 3, 8),
            "max_pairs": trial.suggest_int("max_pairs", 600, 2000, step=100),
            "sen_filter_threshold": trial.suggest_float("sen_filter_threshold", 0.45, 0.80),
            "sentence_filter_threshold": trial.suggest_float("sentence_filter_threshold", 0.45, 0.70),
            "neg_entity_count": trial.suggest_int("neg_entity_count", 15, 60, step=5),
            "neg_triple_count": trial.suggest_int("neg_triple_count", 40, 120, step=10),
            "sentence_loss_weight": trial.suggest_float("sentence_loss_weight", 0.10, 0.40),
        }

        trial_tag = f"trial_{trial.number:04d}"
        trial_log_root = os.path.join(args.study_dir, "logs", trial_tag)
        trial_save_root = os.path.join(args.study_dir, "models", trial_tag)
        os.makedirs(trial_log_root, exist_ok=True)
        os.makedirs(trial_save_root, exist_ok=True)

        cmd = build_train_command(
            train_py=train_py,
            base_args=base_args,
            sampled=sampled,
            log_path=trial_log_root,
            save_path=trial_save_root,
        )

        print("\n[OPTUNA] Running", trial_tag)
        print("[OPTUNA] Command:", " ".join(cmd))

        try:
            subprocess.run(cmd, cwd=args.project_root, check=True)
        except subprocess.CalledProcessError as exc:
            # Penalize failed trials so the study can continue.
            print(f"[OPTUNA] Trial failed with code {exc.returncode}")
            return 0.0

        run_dir = find_latest_run_dir(trial_log_root, base_args["dataset"])
        if run_dir is None:
            print("[OPTUNA] No run directory found, returning 0.0")
            return 0.0

        try:
            if args.objective_metric == "E-T5-MACRO":
                dev_f1 = read_best_dev_et5_macro_f1_from_result(
                    run_dir,
                    match_view=args.objective_match_view,
                )
            else:
                dev_f1 = read_best_dev_f1_from_csv(run_dir)
        except Exception as exc:
            print(f"[OPTUNA] Could not parse dev metric: {exc}")
            return 0.0

        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("log_root", trial_log_root)
        trial.set_user_attr("save_root", trial_save_root)

        # Optional cleanup for disk control
        if args.cleanup_after_trial:
            shutil.rmtree(trial_save_root, ignore_errors=True)

        print(f"[OPTUNA] Trial {trial.number} best dev {args.objective_metric} = {dev_f1:.4f}")
        return dev_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuner for D2E2S train.py")
    parser.add_argument("--project_root", type=str, default=".", help="Path containing train.py")
    parser.add_argument("--study_dir", type=str, default="./optuna_runs", help="Output directory for study artifacts")
    parser.add_argument("--study_name", type=str, default="d2e2s_camera_coqe", help="Optuna study name")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URI, e.g. sqlite:///optuna.db")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per trial")
    parser.add_argument("--lr", type=float, default=None, help="Fixed learning rate; if omitted uses default")
    parser.add_argument(
        "--objective_metric",
        type=str,
        default="E-T5-MACRO",
        choices=["E-T5-MACRO", "senti_nec_f1_micro"],
        help="Primary objective metric for Optuna",
    )
    parser.add_argument(
        "--objective_match_view",
        type=str,
        default="index",
        choices=["index", "span"],
        help="When objective_metric=E-T5-MACRO, choose which evaluation view to optimize",
    )
    parser.add_argument("--fixed_params", type=str, default="", help="JSON object to override default base args")
    parser.add_argument("--cleanup_after_trial", action="store_true", default=False, help="Delete trial model folders after each trial")
    args = parser.parse_args()

    args.project_root = os.path.abspath(args.project_root)
    args.study_dir = os.path.abspath(args.study_dir)
    os.makedirs(args.study_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_json_path = os.path.join(args.study_dir, f"best_params_{timestamp}.json")

    objective = create_objective(args)

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Optuna Finished ===")
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    payload = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved:", best_json_path)


if __name__ == "__main__":
    main()
