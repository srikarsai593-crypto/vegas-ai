from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from utils.features import create_features


FEATURE_COLUMNS = [
    "num_games",
    "win_rate",
    "variance",
    "profit",
    "avg_bet",
    "streak",
    "profit_per_game",
    "consistency_score",
    "risk_index",
]


def _build_benchmark_model():
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )


@lru_cache(maxsize=1)
def load_engineered_dataset(dataset_path="data/dataset.csv"):
    """Load data WITH noise for training/cross-validation"""
    df = pd.read_csv(dataset_path).copy()
    df["streak"] = df["streak_length"]
    
    # ===== APPLY SAME NOISE AS train.py =====
    NOISE_LEVEL = 0.30
    rng = np.random.default_rng(42)
    
    df["profit"] = df["profit"] + rng.normal(0, df["profit"].std() * NOISE_LEVEL, len(df))
    df["win_rate"] = np.clip(df["win_rate"] + rng.normal(0, 0.04 * NOISE_LEVEL, len(df)), 0, 1)
    df["variance"] = np.maximum(df["variance"] * (1 + rng.normal(0, 0.15 * NOISE_LEVEL, len(df))), 0.001)
    df["num_games"] = np.maximum(df["num_games"] * (1 + rng.normal(0, 0.05 * NOISE_LEVEL, len(df))), 10).astype(int)
    
    df = create_features(df)
    return df


def load_clean_dataset(dataset_path="data/dataset.csv"):
    """Load data WITHOUT noise for calibration evaluation on clean data"""
    df = pd.read_csv(dataset_path).copy()
    df["streak"] = df["streak_length"]
    df = create_features(df)  # NO noise applied
    return df


def _summarize_scores(scores):
    return {
        "mean": round(float(np.mean(scores)) * 100, 2),
        "std": round(float(np.std(scores)) * 100, 2),
    }


def _build_stress_test_frame(X_test):
    stressed = X_test.copy()
    rng = np.random.default_rng(42)

    stressed["win_rate"] = np.clip(
        stressed["win_rate"] + rng.normal(0, 0.025, len(stressed)),
        0,
        1,
    )
    stressed["variance"] = np.maximum(
        stressed["variance"] * (1 + rng.normal(0.18, 0.08, len(stressed))),
        0.001,
    )
    stressed["profit"] = stressed["profit"] + rng.normal(0, 550, len(stressed))
    stressed["avg_bet"] = np.maximum(
        stressed["avg_bet"] * (1 + rng.normal(0.12, 0.06, len(stressed))),
        1,
    )
    stressed["profit_per_game"] = stressed["profit"] / np.maximum(stressed["num_games"], 1)
    stressed["consistency_score"] = 1 / (stressed["variance"] + 0.001)
    stressed["risk_index"] = stressed["avg_bet"] * stressed["variance"]

    return stressed


@lru_cache(maxsize=1)
def get_reference_metrics(dataset_path="data/dataset.csv"):
    df = load_engineered_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df["is_skilled"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    cv_results = cross_validate(
        _build_benchmark_model(),
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )

    oof_proba = cross_val_predict(
        _build_benchmark_model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)

    prob_true, prob_pred = calibration_curve(y, oof_proba, n_bins=6, strategy="quantile")

    # Also compute calibration on CLEAN data for comparison
    df_clean = load_clean_dataset(dataset_path)
    X_clean = df_clean[FEATURE_COLUMNS]
    y_clean = df_clean["is_skilled"]
    
    clean_oof_proba = cross_val_predict(
        _build_benchmark_model(),
        X_clean,
        y_clean,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=99),  # Different seed
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    prob_true_clean, prob_pred_clean = calibration_curve(y_clean, clean_oof_proba, n_bins=6, strategy="quantile")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    holdout_model = _build_benchmark_model()
    holdout_model.fit(X_train, y_train)
    holdout_pred = holdout_model.predict(X_test)
    holdout_proba = holdout_model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, holdout_pred)
    stressed_test = _build_stress_test_frame(X_test)
    stressed_pred = holdout_model.predict(stressed_test)

    limitations = [
        "The dataset appears synthetic and relatively clean, so offline metrics may overestimate real-world performance.",
        "The benchmark uses tabular gameplay features only and does not model live table context, opponent behavior, or temporal drift.",
        "Calibration is estimated from out-of-fold probabilities, which is more realistic than a single split but still limited by dataset size.",
        "Risk and addiction outputs are heuristic decision layers, not clinical or regulatory assessments.",
    ]

    return {
        "holdout": {
            "accuracy": round(float(accuracy_score(y_test, holdout_pred)) * 100, 2),
            "precision": round(float(precision_score(y_test, holdout_pred)) * 100, 2),
            "recall": round(float(recall_score(y_test, holdout_pred)) * 100, 2),
            "f1": round(float(f1_score(y_test, holdout_pred)) * 100, 2),
            "brier": round(float(brier_score_loss(y_test, holdout_proba)), 4),
            "test_size": int(len(y_test)),
            "confusion_matrix": {
                "true_lucky": int(matrix[0][0]),
                "false_skilled": int(matrix[0][1]),
                "false_lucky": int(matrix[1][0]),
                "true_skilled": int(matrix[1][1]),
            },
        },
        "stress_test": {
            "accuracy": round(float(accuracy_score(y_test, stressed_pred)) * 100, 2),
            "scenario": "Noisy gameplay conditions with higher volatility, profit drift, and bet-size pressure.",
        },
        "cross_validation": {
            "folds": 5,
            "accuracy": _summarize_scores(cv_results["test_accuracy"]),
            "precision": _summarize_scores(cv_results["test_precision"]),
            "recall": _summarize_scores(cv_results["test_recall"]),
            "f1": _summarize_scores(cv_results["test_f1"]),
        },
        "calibration": {
            "noisy_data": {
                "predicted": [round(float(value) * 100, 2) for value in prob_pred],
                "observed": [round(float(value) * 100, 2) for value in prob_true],
                "note": "Calibration on noisy training data (same distribution as training)"
            },
            "clean_data": {
                "predicted": [round(float(value) * 100, 2) for value in prob_pred_clean],
                "observed": [round(float(value) * 100, 2) for value in prob_true_clean],
                "note": "Calibration on original clean data (true generalization test)"
            }
        },
        "overall": {
            "oof_accuracy": round(float(accuracy_score(y, oof_pred)) * 100, 2),
            "class_balance": {
                "skilled_pct": round(float(y.mean()) * 100, 2),
                "lucky_pct": round(float((1 - y.mean())) * 100, 2),
            },
            "dataset_rows": int(len(df)),
        },
        "limitations": limitations,
    }


def explain_prediction_drivers(player_data, prediction, dataset_path="data/dataset.csv"):
    df = load_engineered_dataset(dataset_path)
    label = prediction["label"]

    class_mask = df["is_skilled"] == (1 if label == "Skilled" else 0)
    opposite_mask = ~class_mask

    feature_map = {
        "num_games": "game volume",
        "win_rate": "win rate",
        "variance": "variance",
        "profit": "profit",
        "avg_bet": "average bet",
        "streak": "streak strength",
        "profit_per_game": "profit efficiency",
        "consistency_score": "consistency",
        "risk_index": "risk exposure",
    }

    player_feature_values = {
        "num_games": player_data["num_games"],
        "win_rate": player_data["win_rate"],
        "variance": player_data["variance"],
        "profit": player_data["profit"],
        "avg_bet": player_data["avg_bet"],
        "streak": player_data["streak"],
        "profit_per_game": player_data["profit"] / max(player_data["num_games"], 1),
        "consistency_score": 1 / (player_data["variance"] + 0.001),
        "risk_index": player_data["avg_bet"] * player_data["variance"],
    }

    drivers = []
    for feature, label_name in feature_map.items():
        class_mean = float(df.loc[class_mask, feature].mean())
        opposite_mean = float(df.loc[opposite_mask, feature].mean())
        midpoint = (class_mean + opposite_mean) / 2
        distance = abs(player_feature_values[feature] - midpoint)
        alignment = abs(player_feature_values[feature] - opposite_mean) - abs(
            player_feature_values[feature] - class_mean
        )

        drivers.append(
            {
                "feature": feature,
                "label": label_name,
                "value": player_feature_values[feature],
                "class_mean": class_mean,
                "opposite_mean": opposite_mean,
                "alignment": alignment,
                "strength": distance,
            }
        )

    supporting = sorted(
        [item for item in drivers if item["alignment"] > 0],
        key=lambda item: (item["alignment"], item["strength"]),
        reverse=True,
    )[:3]
    cautionary = sorted(
        [item for item in drivers if item["alignment"] <= 0],
        key=lambda item: item["strength"],
        reverse=True,
    )[:2]

    return {
        "supporting_factors": supporting,
        "cautionary_factors": cautionary,
    }
