"""Train a Bangla character CNN on BanglaLekha-Isolated and track it in MLflow.

Two dataset layouts are supported (inspect your download and pick one):
  A) one sub-folder per class:      python train.py --data-dir data/Images
  B) flat image folder + label CSV: python train.py --images-dir data/Images \
                                        --csv data/data.csv \
                                        --filename-col <col> --label-col <col>
"""
import argparse
import json
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from common import crop_square, to_ink

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sort_key(name: str):
    return (0, int(name)) if name.isdigit() else (1, name)


def collect_from_folders(data_dir: Path):
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()], key=_sort_key)
    if not classes:
        raise SystemExit(f"No class sub-folders found in {data_dir}")
    paths, labels = [], []
    for c in classes:
        for p in (data_dir / c).rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(c)
    return paths, labels


def collect_from_csv(images_dir: Path, csv_path: Path, fcol: str, lcol: str):
    df = pd.read_csv(csv_path)
    if fcol not in df.columns or lcol not in df.columns:
        raise SystemExit(f"CSV columns are {list(df.columns)}; set --filename-col/--label-col")
    paths, labels = [], []
    for name, lab in zip(df[fcol], df[lcol]):
        p = images_dir / str(name)
        if not p.exists():
            for ext in IMG_EXT:  # CSV may omit the extension
                if (images_dir / f"{name}{ext}").exists():
                    p = images_dir / f"{name}{ext}"
                    break
        if p.exists():
            paths.append(p)
            labels.append(str(lab))
    return paths, labels


def load_images(paths, size):
    X, keep = [], []
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        X.append(crop_square(to_ink(g), size))
        keep.append(i)
        if (len(keep)) % 20000 == 0:
            print(f"  loaded {len(keep)} images...")
    return np.asarray(X, dtype=np.uint8)[..., None], np.asarray(keep)


def build_model(size, n_classes, lr):
    aug = keras.Sequential(
        [
            layers.RandomRotation(0.04, fill_mode="constant"),
            layers.RandomTranslation(0.04, 0.04, fill_mode="constant"),
            layers.RandomZoom(0.06, fill_mode="constant"),
        ],
        name="augment",
    )
    inp = keras.Input((size, size, 1))
    x = layers.Rescaling(1.0 / 255)(inp)
    x = aug(x)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MLflowLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, help="layout A: folder with one sub-folder per class")
    ap.add_argument("--images-dir", type=Path, help="layout B: folder with all images")
    ap.add_argument("--csv", type=Path, help="layout B: CSV with image name and label")
    ap.add_argument("--filename-col", default="filename")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--max-images", type=int, default=0, help="debug: limit dataset size")
    args = ap.parse_args()

    if args.data_dir:
        paths, labels = collect_from_folders(args.data_dir)
    elif args.images_dir and args.csv:
        paths, labels = collect_from_csv(args.images_dir, args.csv, args.filename_col, args.label_col)
    else:
        raise SystemExit("Give --data-dir (layout A) or --images-dir and --csv (layout B)")
    if not paths:
        raise SystemExit("No images found - check your paths")

    if args.max_images:
        rng = np.random.default_rng(0)
        sel = rng.permutation(len(paths))[: args.max_images]
        paths = [paths[i] for i in sel]
        labels = [labels[i] for i in sel]

    classes = sorted(set(labels), key=_sort_key)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(paths)} images, {len(classes)} classes")

    labels_path = Path("labels.json")
    if not labels_path.exists():
        labels_path.write_text(
            json.dumps({str(i): c for i, c in enumerate(classes)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Wrote labels.json with raw class names -> replace values with Bangla characters!")
    else:
        existing = json.loads(labels_path.read_text(encoding="utf-8"))
        if len(existing) != len(classes):
            print(f"WARNING: labels.json has {len(existing)} entries but data has {len(classes)} classes")

    mlflow_dir = Path("artifacts/mlflow").resolve()
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name("bangla-ocr")
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        "bangla-ocr", artifact_location=(mlflow_dir / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=exp_id)

    # --- AUTO-CHECK COMPLETED RUNS IN MLFLOW ---
    client = mlflow.tracking.MlflowClient()
    completed_run_names = set()
    if exp:
        runs = client.search_runs(experiment_ids=[exp_id], filter_string="")
        for r in runs:
            if r.info.status == "FINISHED" and r.data.tags.get("mlflow.runName"):
                completed_run_names.add(r.data.tags.get("mlflow.runName"))

    # --- TWO RUNS CONFIGURATION ---
    runs_config = [
        {"img_size": 32, "lr": 1e-3, "run_name": "cnn-32px"},
        {"img_size": 64, "lr": 3e-4, "run_name": "cnn-64px"}
    ]

    for config in runs_config:
        if config["run_name"] in completed_run_names:
            print(f"\n[INFO] Run '{config['run_name']}' already finished previously in MLflow. Skipping automatically.")
            continue

        print(f"\n==================================================")
        print(f"STARTING MLFLOW RUN: {config['run_name']}")
        print(f"==================================================")

        X, keep = load_images(paths, config["img_size"])
        y = np.asarray([cls_to_idx[labels[i]] for i in keep])
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=args.val_split, stratify=y, random_state=42
        )

        with mlflow.start_run(run_name=config["run_name"]):
            mlflow.log_params(
                {
                    "img_size": config["img_size"],
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": config["lr"],
                    "n_classes": len(classes),
                    "n_train": len(X_tr),
                    "n_val": len(X_va),
                }
            )
            model = build_model(config["img_size"], len(classes), config["lr"])
            callbacks = [
                MLflowLogger(),
                keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            ]
            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=2,
            )
            val_loss, val_acc = model.evaluate(X_va, y_va, verbose=0)
            mlflow.log_metrics({"final_val_loss": float(val_loss), "final_val_accuracy": float(val_acc)})
            print(f"Final val accuracy for {config['run_name']}: {val_acc:.4f}")

            run_model = Path("models") / f"model_{mlflow.active_run().info.run_id[:8]}.keras"
            model.save(run_model)
            mlflow.log_artifact(str(run_model))
            if labels_path.exists():
                mlflow.log_artifact(str(labels_path))

            best_file = Path("models/best_val_accuracy.txt")
            best = float(best_file.read_text()) if best_file.exists() else -1.0
            if val_acc > best:
                model.save("models/model.keras")
                best_file.write_text(str(float(val_acc)))
                print(f"*** New best model saved! (Accuracy: {val_acc:.4f}) ***")
            
            run_model.unlink(missing_ok=True)

if __name__ == "__main__":
    main()