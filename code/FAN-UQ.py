
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, Reshape, Conv1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import xgboost as xgb
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")



def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


@dataclass
class WaterTargets:
    df: pd.DataFrame
    feature_cols: List[str]
    y_wqi: np.ndarray
    y_wqc: np.ndarray
    class_names: List[str]


def load_and_build_targets(csv_path: str,
                           feature_cols: List[str],
                           rename_columns: bool = True) -> WaterTargets:

    df = pd.read_csv(csv_path, encoding_errors="ignore")

  
    if rename_columns:

        if df.shape[1] >= 12:
            df.columns = ['Temp', 'DO', 'PH', 'Cond',
                          'BOD', 'NO3', 'FC', 'TC', 'year'] + list(df.columns[12:])
       
    keep_cols = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")

    df = df.copy()
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    limits = {'DO': 5, 'PH': 8.5, 'BOD': 3, 'Cond': 1000, 'NO3': 45, 'FC': 500, 'TC': 5000}
    k = 1 / sum(1.0 / v for v in limits.values())
    weights = {p: k / v for p, v in limits.items()}

    def compute_wqi(row: pd.Series) -> float:
        q_sum, w_sum = 0.0, 0.0
        for p in limits.keys():
            val = row[p]
            if pd.isna(val):
                return np.nan

            if p == 'DO':
                q = ((val - 14.6) / (5.0 - 14.6)) * 100.0
            elif p == 'PH':
                q = ((val - 7.0) / (8.5 - 7.0)) * 100.0
            else:
              
                q = (val / limits[p]) * 100.0

            q_sum += q * weights[p]
            w_sum += weights[p]
        return q_sum / w_sum

    df["WQI"] = df.apply(compute_wqi, axis=1)

    def map_wqc(v: float) -> str:
        if v <= 25:
            return "Excellent"
        elif v <= 50:
            return "Good"
        elif v <= 75:
            return "Poor"
        elif v <= 100:
            return "Very Poor"
        else:
            return "Unfit"

    df["WQC"] = df["WQI"].apply(lambda x: map_wqc(x) if pd.notna(x) else np.nan)

    df = df.dropna(subset=["WQI", "WQC"]).reset_index(drop=True)

    df = df[df["WQC"] != "Excellent"].reset_index(drop=True)

    # Encode WQC labels
    le = LabelEncoder()
    y_wqc = le.fit_transform(df["WQC"].values)
    class_names = list(le.classes_)

    y_wqi = df["WQI"].values.astype(float)

    return WaterTargets(df=df, feature_cols=feature_cols, y_wqi=y_wqi, y_wqc=y_wqc, class_names=class_names)



@dataclass
class FoldPreprocessor:
    imputer: KNNImputer
    scaler: StandardScaler
    feature_cols_used: List[str]

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        Xt_imp = self.imputer.fit_transform(X_train)
        self.scaler.fit(Xt_imp)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled


def iqr_cap_fit(X_train: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    caps = {}
    for c in X_train.columns:
        q1 = X_train[c].quantile(0.25)
        q3 = X_train[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        caps[c] = (lo, hi)
    return caps


def iqr_cap_apply(X: pd.DataFrame, caps: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xc = X.copy()
    for c, (lo, hi) in caps.items():
        Xc[c] = Xc[c].clip(lower=lo, upper=hi)
    return Xc



class FeatureAttention(Layer):

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="att_w", shape=(d, d), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(d,), initializer="zeros", trainable=True)

    def call(self, x):
        u = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(u, axis=-1)
        return x * a, a


def build_fan_uq(input_dim: int, task: str, n_classes: int) -> Tuple[Model, Model]:

    inp = Input(shape=(input_dim,))
    x_att, alpha = FeatureAttention(name="feature_attention")(inp)

    x = Dense(64, activation="relu")(x_att)
    x = Dropout(0.30)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.20)(x)

    if task == "reg":
        out = Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        out = Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = Model(inp, out, name=f"FAN_UQ_{task}")
    model.compile(optimizer="adam", loss=loss)

    att_model = Model(inp, alpha, name="FAN_UQ_attention")
    return model, att_model


def build_keras_baseline(name: str, input_dim: int, task: str, n_classes: int) -> Model:
    inp = Input(shape=(input_dim,))

    if name == "MLP":
        x = Dense(64, activation="relu")(inp)
        x = Dropout(0.30)(x)
        x = Dense(32, activation="relu")(x)

    elif name == "1D-CNN":
        x = Reshape((input_dim, 1))(inp)
        x = Conv1D(32, 3, padding="same", activation="relu")(x)
        x = Conv1D(16, 3, padding="same", activation="relu")(x)
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)

    elif name == "LSTM":
        x = Reshape((input_dim, 1))(inp)
        x = LSTM(32, activation="tanh")(x)
        x = Dense(32, activation="relu")(x)

    else:
        raise ValueError(f"Unknown Keras baseline: {name}")

    if task == "reg":
        out = Dense(1, activation="linear")(x)
        loss = "mse"
    else:
        out = Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    m = Model(inp, out, name=f"{name}_{task}")
    m.compile(optimizer="adam", loss=loss)
    return m


def build_tree_model(name: str, task: str):
    if task == "reg":
        if name == "RF":
            return RandomForestRegressor(random_state=42, n_estimators=400, n_jobs=-1)
        if name == "XGBoost":
            return xgb.XGBRegressor(
                random_state=42, n_estimators=700, max_depth=6,
                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9
            )
    else:
        if name == "RF":
            return RandomForestClassifier(random_state=42, n_estimators=500, n_jobs=-1)
        if name == "XGBoost":
            return xgb.XGBClassifier(
                random_state=42, n_estimators=700, max_depth=6,
                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                eval_metric="mlogloss"
            )
    raise ValueError(f"Unknown tree model: {name} / task={task}")



SCENARIOS = {
    "S1_all": [],
    "S2_missing_FC": ["FC"],
    "S3_missing_pH_DO": ["PH", "DO"],
}


def cols_after_scenario(all_cols: List[str], missing_cols: List[str]) -> List[str]:
    return [c for c in all_cols if c not in missing_cols]



@dataclass
class CVResult:
    fold_scores: List[Dict[str, float]]
    y_true_all: np.ndarray
    y_pred_all: np.ndarray
    entropy_all: Optional[np.ndarray] = None
    attention_all: Optional[np.ndarray] = None  # mean attention per fold


def mc_dropout_predict_classification(model: Model, X: np.ndarray, n_passes: int = 50) -> Tuple[np.ndarray, np.ndarray]:
  
    probs = []
    for _ in range(n_passes):
        p = model(X, training=True).numpy()
        probs.append(p)
    mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
 
    mean_prob = mean_prob / np.clip(mean_prob.sum(axis=1, keepdims=True), 1e-12, None)
    ent = scipy_entropy(mean_prob.T)  
    return mean_prob, ent


def mc_dropout_predict_regression(model: Model, X: np.ndarray, n_passes: int = 50) -> np.ndarray:
    preds = []
    for _ in range(n_passes):
        p = model(X, training=True).numpy().ravel()
        preds.append(p)
    return np.mean(np.stack(preds, axis=0), axis=0)


def run_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
    task: str,
    model_name: str,
    scenario_missing: List[str],
    class_names: Optional[List[str]] = None,
    seed: int = 42
) -> CVResult:

    set_global_seed(seed)

    used_cols = cols_after_scenario(feature_cols, scenario_missing)
    X_full = df[used_cols].copy()

    
    if task == "cls":
        strat_labels = y
        n_classes = len(np.unique(y))
    else:
        
        strat_labels = np.zeros(len(y), dtype=int)
        n_classes = 1  # unused

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_scores = []
    y_true_all, y_pred_all = [], []
    entropy_all = []  
    attention_folds = []  

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, strat_labels), start=1):
        X_tr_raw = X_full.iloc[tr_idx].copy()
        X_va_raw = X_full.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

      
        if task == "reg":
            kb = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
            y_tr_bins = kb.fit_transform(y_tr.reshape(-1, 1)).astype(int).ravel()
  

        caps = iqr_cap_fit(X_tr_raw)
        X_tr_capped = iqr_cap_apply(X_tr_raw, caps)
        X_va_capped = iqr_cap_apply(X_va_raw, caps)

        pre = FoldPreprocessor(
            imputer=KNNImputer(n_neighbors=5),
            scaler=StandardScaler(),
            feature_cols_used=used_cols
        ).fit(X_tr_capped)

        X_tr = pre.transform(X_tr_capped)
        X_va = pre.transform(X_va_capped)

 
        if model_name == "FAN-UQ":
            if task == "reg":
                model, att_model = build_fan_uq(X_tr.shape[1], task="reg", n_classes=0)
                cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
                model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=200, batch_size=32, verbose=0, callbacks=cb)
                yhat = mc_dropout_predict_regression(model, X_va, n_passes=50)
   
                att = att_model.predict(X_va, verbose=0).mean(axis=0)  # (d,)
                attention_folds.append(att)

                r2 = r2_score(y_va, yhat)
                mae = mean_absolute_error(y_va, yhat)
                rmse = np.sqrt(mean_squared_error(y_va, yhat))
                fold_scores.append({"fold": fold_i, "R2": r2, "MAE": mae, "RMSE": rmse})

            else:
                assert class_names is not None
                model, att_model = build_fan_uq(X_tr.shape[1], task="cls", n_classes=len(class_names))
                cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
                model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=200, batch_size=32, verbose=0, callbacks=cb)

                mean_prob, ent = mc_dropout_predict_classification(model, X_va, n_passes=50)
                yhat = np.argmax(mean_prob, axis=1)
                entropy_all.append(ent)

                att = att_model.predict(X_va, verbose=0).mean(axis=0)
                attention_folds.append(att)

                acc = accuracy_score(y_va, yhat)
                prec = precision_score(y_va, yhat, average="macro", zero_division=0)
                rec = recall_score(y_va, yhat, average="macro", zero_division=0)
                f1 = f1_score(y_va, yhat, average="macro", zero_division=0)
                fold_scores.append({"fold": fold_i, "Accuracy": acc, "Precision": prec, "Recall": rec, "MacroF1": f1})

        elif model_name in ["MLP", "1D-CNN", "LSTM"]:
            if task == "reg":
                model = build_keras_baseline(model_name, X_tr.shape[1], task="reg", n_classes=0)
                cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
                model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=200, batch_size=32, verbose=0, callbacks=cb)
                yhat = model.predict(X_va, verbose=0).ravel()

                r2 = r2_score(y_va, yhat)
                mae = mean_absolute_error(y_va, yhat)
                rmse = np.sqrt(mean_squared_error(y_va, yhat))
                fold_scores.append({"fold": fold_i, "R2": r2, "MAE": mae, "RMSE": rmse})
            else:
                assert class_names is not None
                model = build_keras_baseline(model_name, X_tr.shape[1], task="cls", n_classes=len(class_names))
                cb = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
                model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=200, batch_size=32, verbose=0, callbacks=cb)
                prob = model.predict(X_va, verbose=0)
                yhat = np.argmax(prob, axis=1)

                acc = accuracy_score(y_va, yhat)
                prec = precision_score(y_va, yhat, average="macro", zero_division=0)
                rec = recall_score(y_va, yhat, average="macro", zero_division=0)
                f1 = f1_score(y_va, yhat, average="macro", zero_division=0)
                fold_scores.append({"fold": fold_i, "Accuracy": acc, "Precision": prec, "Recall": rec, "MacroF1": f1})

        else:
         
            if task == "reg":
                model = build_tree_model(model_name, task="reg")
                model.fit(X_tr, y_tr)
                yhat = model.predict(X_va)

                r2 = r2_score(y_va, yhat)
                mae = mean_absolute_error(y_va, yhat)
                rmse = np.sqrt(mean_squared_error(y_va, yhat))
                fold_scores.append({"fold": fold_i, "R2": r2, "MAE": mae, "RMSE": rmse})
            else:
                assert class_names is not None
                model = build_tree_model(model_name, task="cls")
                model.fit(X_tr, y_tr)
                yhat = model.predict(X_va)

                acc = accuracy_score(y_va, yhat)
                prec = precision_score(y_va, yhat, average="macro", zero_division=0)
                rec = recall_score(y_va, yhat, average="macro", zero_division=0)
                f1 = f1_score(y_va, yhat, average="macro", zero_division=0)
                fold_scores.append({"fold": fold_i, "Accuracy": acc, "Precision": prec, "Recall": rec, "MacroF1": f1})

        y_true_all.extend(list(y_va))
        y_pred_all.extend(list(yhat))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    ent_out = None
    if entropy_all:
        ent_out = np.concatenate(entropy_all, axis=0)

    att_out = None
    if attention_folds:
        att_out = np.vstack(attention_folds)  

    return CVResult(
        fold_scores=fold_scores,
        y_true_all=y_true_all,
        y_pred_all=y_pred_all,
        entropy_all=ent_out,
        attention_all=att_out
    )



def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_entropy_violin(entropy_vals: np.ndarray, correct_mask: np.ndarray, title: str, save_path: Optional[str] = None):

    data = [entropy_vals[correct_mask], entropy_vals[~correct_mask]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_ylabel("Predictive entropy")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_attention_radar(att_mean: np.ndarray, feat_cols: List[str], title: str, save_path: Optional[str] = None):
    n = len(feat_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    vals = att_mean.tolist()
    vals += vals[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_cols)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    set_global_seed(42)


    FEATURES = ["Temp", "DO", "PH", "Cond", "BOD", "NO3", "FC", "TC"]

    data = load_and_build_targets("water_data.csv", FEATURES, rename_columns=True)

    MODELS = ["FAN-UQ", "MLP", "1D-CNN", "LSTM", "RF", "XGBoost"]


    results_cls = {}
    for scen_name, missing in SCENARIOS.items():
        results_cls[scen_name] = {}
        for m in MODELS:
            res = run_cv(
                df=data.df,
                feature_cols=data.feature_cols,
                y=data.y_wqc,
                task="cls",
                model_name=m,
                scenario_missing=missing,
                class_names=data.class_names,
                seed=42
            )
            results_cls[scen_name][m] = res
            mean_acc = np.mean([d["Accuracy"] for d in res.fold_scores])
            mean_f1 = np.mean([d["MacroF1"] for d in res.fold_scores])
            print(f"[CLS] {scen_name:>14s} | {m:>7s} | mean Acc={mean_acc:.4f} | mean MacroF1={mean_f1:.4f}")

    fan_s1 = results_cls["S1_all"]["FAN-UQ"]
    cm = confusion_matrix(fan_s1.y_true_all, fan_s1.y_pred_all)
    plot_confusion_matrix(cm, data.class_names, "FAN-UQ confusion matrix (S1)")

    if fan_s1.entropy_all is not None:
        correct = (fan_s1.y_true_all == fan_s1.y_pred_all)
        plot_entropy_violin(fan_s1.entropy_all, correct, "FAN-UQ predictive entropy (S1)")

    if fan_s1.attention_all is not None:
        att_mean = fan_s1.attention_all.mean(axis=0)
        used_cols = cols_after_scenario(FEATURES, SCENARIOS["S1_all"])
        plot_attention_radar(att_mean, used_cols, "FAN-UQ global mean attention (S1)")
