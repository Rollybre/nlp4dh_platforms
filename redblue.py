# -*- coding: utf-8 -*-

"""
Complete script for:
1) spaCy text classification with group-safe cross-validation
2) logistic classification with out-of-fold AUC
3) more defensible p-values via cross-fit held-out inference
4) univariate tests with FDR correction
5) targeted ablations around first-person narration
6) ablation plotting

Relative paths still, hesitated to do pathlib but don't want to break things now, we are in a rush.
"""

import os
import shutil
import logging
import warnings
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train as spacy_train

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_val_predict
)
from sklearn.metrics import roc_auc_score, classification_report


# ======================================================
# PARAMETERS / RELATIVE PATHS
# ======================================================

CP_PATH = "/home/alexandre/phd/data/corpus_creepypastas.csv"
FEATURES_PATH = "/home/alexandre/phd/data/data_pronoms.csv"
SPACY_CONFIG_PATH = "/home/alexandre/phd/spacy_textcat/config_gpu.cfg"

CV_DIR = "cv"
OUTPUT_DIR = "output"

N_PER_SOURCE = 2000
MAX_TEXT_CHARS = 1_000_000
MAX_TOKENS = 512
OVERLAP = 50
N_SPLITS = 5
RANDOM_STATE = 42

TEXT_COL = "texte"
SOURCE_COL = "source"
VALID_SOURCES = ["reddit", "fandom"]

GPU = True

SAVE_TABLES = True
SAVE_PLOTS = True

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ======================================================
# GENERIC UTILITIES
# ======================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def winsorize_series(s, lower=0.01, upper=0.99):
    return s.clip(s.quantile(lower), s.quantile(upper))


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def flatten_spacy_reports(reports, labels):
    rows = []
    for report in reports:
        row = {
            "accuracy": report.get("accuracy", np.nan),
            "precision_macro": report.get("macro avg", {}).get("precision", np.nan),
            "recall_macro": report.get("macro avg", {}).get("recall", np.nan),
            "f1_macro": report.get("macro avg", {}).get("f1-score", np.nan),
            "precision_weighted": report.get("weighted avg", {}).get("precision", np.nan),
            "recall_weighted": report.get("weighted avg", {}).get("recall", np.nan),
            "f1_weighted": report.get("weighted avg", {}).get("f1-score", np.nan),
        }

        for label in labels:
            if label in report:
                row[f"precision_{label}"] = report[label].get("precision", np.nan)
                row[f"recall_{label}"] = report[label].get("recall", np.nan)
                row[f"f1_{label}"] = report[label].get("f1-score", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def save_df(df, filename):
    if SAVE_TABLES:
        ensure_dir(OUTPUT_DIR)
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


# ======================================================
# PART 1 — SPACY TEXT CLASSIFICATION (NO LEAKAGE)
# ======================================================

def load_story_corpus(cp_path):
    logger.info("Loading story corpus...")
    df_cp = pd.read_csv(cp_path)

    df = df_cp[df_cp[SOURCE_COL].isin(VALID_SOURCES)].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.slice(0, MAX_TEXT_CHARS)

    df = (
        df.groupby(SOURCE_COL, group_keys=False)
          .apply(lambda x: x.sample(n=min(N_PER_SOURCE, len(x)), random_state=RANDOM_STATE))
          .reset_index(drop=True)
    )

    df["story_id"] = np.arange(len(df))

    logger.info(f"Balanced corpus: {df.shape[0]} stories")
    logger.info(f"Distribution: {df[SOURCE_COL].value_counts().to_dict()}")

    return df


def make_docs_chunked(nlp, data, target_file, cats, max_tokens=512, overlap=50):
    """
    data = [(text, label), ...]
    Writes a .spacy file and returns chunk docs + labels in memory.
    """
    docs_bin = DocBin(store_user_data=True)
    class_counter = Counter()

    chunk_docs = []
    chunk_labels = []

    for text, label in data:
        if not isinstance(text, str):
            text = str(text)

        doc = nlp.make_doc(text)
        n_tokens = len(doc)

        if n_tokens == 0:
            continue

        start = 0
        while start < n_tokens:
            end = min(start + max_tokens, n_tokens)

            chunk_span = doc[start:end]
            chunk_doc = chunk_span.as_doc()

            for cat in cats:
                chunk_doc.cats[cat] = 1.0 if cat == label else 0.0

            docs_bin.add(chunk_doc)
            chunk_docs.append(chunk_doc)
            chunk_labels.append(label)
            class_counter[label] += 1

            if end == n_tokens:
                break

            start = end - overlap

    docs_bin.to_disk(target_file)

    logger.info(f"DocBin written: {target_file}")
    logger.info(f"Chunk distribution: {dict(class_counter)}")

    return chunk_docs, chunk_labels


def run_spacy_cv():
    logger.info("=== SPACY TEXT CLASSIFICATION ===")

    if not os.path.exists(SPACY_CONFIG_PATH):
        raise FileNotFoundError(f"spaCy config not found: {SPACY_CONFIG_PATH}")

    ensure_dir(CV_DIR)
    ensure_dir(OUTPUT_DIR)

    df = load_story_corpus(CP_PATH)

    X_text = df[TEXT_COL].tolist()
    y = df[SOURCE_COL].tolist()
    groups = df["story_id"].tolist()

    labels = sorted(set(y))
    logger.info(f"Labels: {labels}")

    # Tokenizer only: avoids unnecessary overhead
    nlp_tok = spacy.blank("en")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    reports = []

    for fold, (train_idx, dev_idx) in enumerate(cv.split(X_text, y, groups), start=1):
        logger.info(f"\n===== FOLD {fold}/{N_SPLITS} =====")

        fold_dir = os.path.join(CV_DIR, f"fold_{fold}")
        reset_dir(fold_dir)

        train_data = [(X_text[i], y[i]) for i in train_idx]
        dev_data = [(X_text[i], y[i]) for i in dev_idx]

        train_spacy = os.path.join(fold_dir, "train.spacy")
        dev_spacy = os.path.join(fold_dir, "dev.spacy")
        fold_config = os.path.join(fold_dir, "config.cfg")
        model_dir = os.path.join(fold_dir, "model")

        _, _ = make_docs_chunked(
            nlp=nlp_tok,
            data=train_data,
            target_file=train_spacy,
            cats=labels,
            max_tokens=MAX_TOKENS,
            overlap=OVERLAP
        )

        dev_docs, dev_labels = make_docs_chunked(
            nlp=nlp_tok,
            data=dev_data,
            target_file=dev_spacy,
            cats=labels,
            max_tokens=MAX_TOKENS,
            overlap=OVERLAP
        )

        shutil.copy(SPACY_CONFIG_PATH, fold_config)

        logger.info("Training spaCy model...")
        spacy_train(
            fold_config,
            output_path=model_dir,
            overrides={
                "paths.train": train_spacy,
                "paths.dev": dev_spacy,
            },
        )

        logger.info("Evaluating...")
        nlp_fold = spacy.load(os.path.join(model_dir, "model-best"))

        y_true = []
        y_pred = []

        for doc, true_label in zip(dev_docs, dev_labels):
            pred = nlp_fold(doc.text).cats
            pred_label = max(pred, key=pred.get)

            y_true.append(true_label)
            y_pred.append(pred_label)

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        reports.append(report)

        logger.info(f"Accuracy fold {fold}: {report['accuracy']:.4f}")

    results_df = flatten_spacy_reports(reports, labels)

    print("\n===== SPACY RESULTS BY FOLD =====")
    print(results_df)

    print("\n===== SPACY MEAN / STD =====")
    print(results_df.agg(["mean", "std"]))

    save_df(results_df, "spacy_cv_results.csv")

    return results_df


# ======================================================
# PART 2 — FEATURE DATA / PREPROCESSING
# ======================================================

def load_feature_data(features_path):
    logger.info("Loading feature data...")
    df = pd.read_csv(features_path)
    df = df[df[SOURCE_COL].isin(VALID_SOURCES)].copy()

    # reddit=0, fandom=1
    df[SOURCE_COL] = df[SOURCE_COL].apply(lambda x: 1 if x == "fandom" else 0)

    num_features = [
        "longueur_texte", "nombre_mots",
        "longueur_phrase", "Flesch Reading Ease", "Flesch-Kincaid Grade Level",
        "Gunning Fog Index", "SMOG Index", "Automated Readability Index",
        "Coleman-Liau Index", "Dale-Chall Readability Score", "Indice Lix",
        "Ratio Types/Tokens", "Hapax Legomena", "Densité Lexicale",
        "Indice Honore's R", "fear", "neutral", "disgust", "anger", "sadness",
        "surprise", "joy", "noun_verb_ratio", "noun_adj_verb_ratio",
        "passive_verb_ratio", "nll", "Ratio_P1_rest"
    ]

    available_features = [c for c in num_features if c in df.columns]
    df = df[available_features + [SOURCE_COL]].copy()

    logger.info(f"Available features: {len(available_features)}")
    return df, available_features


def preprocess_features(df):
    df_clean = df.copy()

    if "nombre_mots" in df_clean.columns:
        df_clean = df_clean[df_clean["nombre_mots"] > 30]

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()

    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != SOURCE_COL]

    for col in num_cols:
        df_clean[col] = winsorize_series(df_clean[col])

    heavy_cols = [
        "longueur_texte",
        "nombre_mots",
        "Hapax Legomena",
        "Indice Honore's R",
        "Indice Lix"
    ]

    for col in heavy_cols:
        if col in df_clean.columns:
            df_clean[f"log_{col}"] = np.log1p(df_clean[col])

    return df_clean


# ======================================================
# PART 3 — PREDICTIVE LOGIT BACKBONE
# ======================================================

def make_base_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            cv=5,
            scoring="roc_auc",
            max_iter=5000,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])


def cross_validated_auc(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    pipeline = make_base_pipeline()

    proba_oof = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    auc = roc_auc_score(y, proba_oof)
    return auc


# ======================================================
# PART 4 — CROSS-FIT HELD-OUT INFERENCE
# ======================================================

#To do more appropriate than p-values

def run_logit_with_crossfit_inference():
    """
    1) honest predictive AUC with out-of-fold probabilities
    2) more defensible p-values via sample-splitting / cross-fit inference:
       - variable selection on train folds only
       - GLM Binomial inference on held-out fold only
       - conservative aggregation of held-out p-values
    """
    logger.info("=== LOGIT + CROSS-FIT INFERENCE ===")

    df_raw, _ = load_feature_data(FEATURES_PATH)
    df_clean = preprocess_features(df_raw)

    print("Shape after cleaning:", df_clean.shape)

    y = df_clean[SOURCE_COL].astype(int).copy()
    X = df_clean.drop(columns=[SOURCE_COL]).copy()
    feature_names = X.columns.tolist()

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # 1) A quite honest predictive AUC
    pred_pipeline = make_base_pipeline()

    proba_oof = cross_val_predict(
        pred_pipeline,
        X,
        y,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    auc = roc_auc_score(y, proba_oof)

    print("\n=== Overall predictive performance ===")
    print("AUC out-of-fold:", round(auc, 4))

    # 2) Cross-fit inference
    stats = {
        feat: {
            "selected_count": 0,
            "tested_count": 0,
            "pvals": [],
            "coefs": [],
            "odds_ratios": []
        }
        for feat in feature_names
    }

    fold_summaries = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        scaler = StandardScaler()
        X_train_z = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_names,
            index=X_train.index
        )
        X_test_z = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_names,
            index=X_test.index
        )

        selector = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            cv=5,
            scoring="roc_auc",
            max_iter=5000,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        selector.fit(X_train_z, y_train)

        coef_train = pd.Series(selector.coef_[0], index=feature_names)
        selected_vars = coef_train[coef_train != 0].index.tolist()

        # a teensy tiny safeguard against unstable held-out GLM fits
        max_vars = max(1, min(len(selected_vars), int(len(y_test) / 15)))
        if len(selected_vars) > max_vars:
            selected_vars = (
                coef_train[selected_vars]
                .abs()
                .sort_values(ascending=False)
                .head(max_vars)
                .index
                .tolist()
            )

        print("Selected vars on training fold:", len(selected_vars))
        print(selected_vars)

        for feat in selected_vars:
            stats[feat]["selected_count"] += 1

        if len(selected_vars) == 0:
            fold_summaries.append({
                "fold": fold,
                "n_selected": 0,
                "tested_vars": ""
            })
            continue

        X_test_sm = sm.add_constant(X_test_z[selected_vars], has_constant="add")

        try:
            glm = sm.GLM(y_test, X_test_sm, family=sm.families.Binomial())
            res = glm.fit(cov_type="HC3")

            tested_here = []
            for feat in selected_vars:
                if feat in res.params.index:
                    pval = res.pvalues.get(feat, np.nan)
                    coef = res.params.get(feat, np.nan)

                    stats[feat]["tested_count"] += 1
                    tested_here.append(feat)

                    if pd.notnull(pval):
                        stats[feat]["pvals"].append(float(pval))
                    if pd.notnull(coef):
                        stats[feat]["coefs"].append(float(coef))
                        stats[feat]["odds_ratios"].append(float(np.exp(coef)))

            fold_summaries.append({
                "fold": fold,
                "n_selected": len(selected_vars),
                "tested_vars": ", ".join(tested_here)
            })

        except Exception as e:
            warnings.warn(f"Fold {fold}: held-out GLM inference failed ({e}).")
            fold_summaries.append({
                "fold": fold,
                "n_selected": len(selected_vars),
                "tested_vars": ""
            })
            continue

    # 3) Aggregate p-values conservatively
    rows = []

    for feat, d in stats.items():
        sel_count = d["selected_count"]
        test_count = d["tested_count"]
        pvals = d["pvals"]
        coefs = d["coefs"]
        ors = d["odds_ratios"]

        if sel_count == 0:
            continue

        selection_freq = sel_count / N_SPLITS

        if len(pvals) > 0:
            p_min = float(np.min(pvals))
            p_median = float(np.median(pvals))
            p_agg_bonf = min(1.0, len(pvals) * p_min)
        else:
            p_min = np.nan
            p_median = np.nan
            p_agg_bonf = np.nan

        if len(coefs) > 0:
            coef_median = float(np.median(coefs))
            coef_mean = float(np.mean(coefs))
            or_median = float(np.median(ors))
            sign_stability = float(np.mean(np.sign(coefs) == np.sign(np.median(coefs))))
        else:
            coef_median = np.nan
            coef_mean = np.nan
            or_median = np.nan
            sign_stability = np.nan

        rows.append({
            "feature": feat,
            "selection_freq": selection_freq,
            "n_selected_folds": sel_count,
            "n_tested_folds": test_count,
            "coef_median_heldout": coef_median,
            "coef_mean_heldout": coef_mean,
            "OR_median_heldout": or_median,
            "p_min_heldout": p_min,
            "p_median_heldout": p_median,
            "p_agg_bonf": p_agg_bonf,
            "sign_stability": sign_stability
        })

    inference_table = pd.DataFrame(rows)

    if not inference_table.empty:
        mask = inference_table["p_agg_bonf"].notna()
        if mask.sum() > 0:
            reject, p_fdr, _, _ = multipletests(
                inference_table.loc[mask, "p_agg_bonf"],
                method="fdr_bh"
            )
            inference_table.loc[mask, "p_fdr"] = p_fdr
            inference_table.loc[mask, "significant_fdr"] = reject

        inference_table = inference_table.sort_values(
            ["p_agg_bonf", "selection_freq"],
            ascending=[True, False]
        ).reset_index(drop=True)

    fold_summaries_df = pd.DataFrame(fold_summaries)

    print("\n=== Fold summaries ===")
    print(fold_summaries_df)

    print("\n=== Cross-fit inference table ===")
    print(inference_table)

    save_df(fold_summaries_df, "logit_crossfit_fold_summaries.csv")
    if not inference_table.empty:
        save_df(inference_table, "logit_crossfit_inference.csv")

    return {
        "auc_oof": auc,
        "fold_summaries": fold_summaries_df,
        "inference_table": inference_table
    }


# ======================================================
# PART 5 — UNIVARIATE TESTS + FDR
# ======================================================

def run_ttests():
    logger.info("=== UNIVARIATE TESTS ===")

    df_raw, available_features = load_feature_data(FEATURES_PATH)
    df_raw = df_raw.copy()
    df_raw["source_label"] = df_raw[SOURCE_COL].apply(lambda x: "fandom" if x == 1 else "reddit")

    metrics = [m for m in available_features]

    rows = []

    for metric in metrics:
        x = df_raw[df_raw["source_label"] == "fandom"][metric].replace([np.inf, -np.inf], np.nan).dropna()
        y = df_raw[df_raw["source_label"] == "reddit"][metric].replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) < 2 or len(y) < 2:
            continue

        stat, pval = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        d = cohen_d(x, y)

        rows.append({
            "metric": metric,
            "n_fandom": len(x),
            "n_reddit": len(y),
            "mean_fandom": float(np.mean(x)),
            "mean_reddit": float(np.mean(y)),
            "t_stat": float(stat),
            "p_value": float(pval),
            "cohen_d": float(d) if pd.notnull(d) else np.nan
        })

    if len(rows) == 0:
        print("No t-test results.")
        return None

    df_res = pd.DataFrame(rows)

    reject, pvals_corr, _, _ = multipletests(df_res["p_value"].values, method="fdr_bh")
    df_res["p_fdr"] = pvals_corr
    df_res["significant_fdr"] = reject

    df_res = df_res.sort_values("p_fdr").reset_index(drop=True)

    print("\n=== T-tests with FDR correction ===")
    print(df_res)

    save_df(df_res, "ttests_fdr.csv")

    return df_res


# ======================================================
# PART 6 — ABLATIONS
# ======================================================

def get_feature_blocks(df_clean):
    blocks = {
        "length": [
            "longueur_texte", "nombre_mots", "log_longueur_texte", "log_nombre_mots"
        ],
        "readability": [
            "longueur_phrase",
            "Flesch Reading Ease",
            "Flesch-Kincaid Grade Level",
            "Gunning Fog Index",
            "SMOG Index",
            "Automated Readability Index",
            "Coleman-Liau Index",
            "Dale-Chall Readability Score",
            "Indice Lix",
            "log_Indice Lix",
        ],
        "lexical": [
            "Ratio Types/Tokens",
            "Hapax Legomena",
            "Densité Lexicale",
            "Indice Honore's R",
            "log_Hapax Legomena",
            "log_Indice Honore's R",
        ],
        "emotion": [
            "fear", "neutral", "disgust", "anger", "sadness", "surprise", "joy"
        ],
        "syntax": [
            "noun_verb_ratio",
            "noun_adj_verb_ratio",
            "passive_verb_ratio",
            "Ratio_P1_rest",
        ],
        "lm": [
            "nll"
        ]
    }

    available = {}
    for block_name, cols in blocks.items():
        cols_here = [c for c in cols if c in df_clean.columns]
        if cols_here:
            available[block_name] = cols_here

    return available


def cross_validated_auc_residualized(df_clean, feature_cols, target_col="source",
                                     residualize_on="Ratio_P1_rest",
                                     n_splits=5, random_state=42):
    """
    For each outer fold, regress each predictor on residualize_on using TRAIN only,
    then classify using held-out residuals.
    The residualizing variable itself is removed from the predictor set.
    """
    if residualize_on not in df_clean.columns:
        raise ValueError(f"{residualize_on} not found in df_clean.")

    cols = [c for c in feature_cols if c != residualize_on]
    if len(cols) == 0:
        raise ValueError("No feature left after removing residualize_on.")

    y = df_clean[target_col].astype(int).copy()
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    oof_pred = np.zeros(len(df_clean), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(df_clean[cols], y), start=1):
        train_df = df_clean.iloc[train_idx].copy()
        test_df = df_clean.iloc[test_idx].copy()

        z_train = train_df[residualize_on].values
        z_test = test_df[residualize_on].values

        X_train_resid = pd.DataFrame(index=train_df.index)
        X_test_resid = pd.DataFrame(index=test_df.index)

        for col in cols:
            x_train = train_df[col].values
            A_train = np.column_stack([np.ones(len(z_train)), z_train])

            coef, _, _, _ = np.linalg.lstsq(A_train, x_train, rcond=None)
            a, b = coef

            xhat_train = a + b * z_train
            xhat_test = a + b * z_test

            X_train_resid[col] = x_train - xhat_train
            X_test_resid[col] = test_df[col].values - xhat_test

        y_train = y.iloc[train_idx]

        pipeline = make_base_pipeline()
        pipeline.fit(X_train_resid, y_train)
        oof_pred[test_idx] = pipeline.predict_proba(X_test_resid)[:, 1]

    auc = roc_auc_score(y, oof_pred)
    return auc


def run_ablations():
    logger.info("=== ABLATIONS ===")

    df_raw, _ = load_feature_data(FEATURES_PATH)
    df_clean = preprocess_features(df_raw)

    y = df_clean["source"].astype(int).copy()
    all_features = [c for c in df_clean.columns if c != "source"]

    blocks = get_feature_blocks(df_clean)

    if "Ratio_P1_rest" not in df_clean.columns:
        raise ValueError("Ratio_P1_rest is missing, cannot run the key ablations.")

    syntax_block = blocks.get("syntax", [])
    lexical_block = blocks.get("lexical", [])
    emotion_block = blocks.get("emotion", [])
    readability_block = blocks.get("readability", [])
    length_block = blocks.get("length", [])

    nominal_downstream = [
        c for c in [
            "Densité Lexicale",
            "noun_verb_ratio",
            "noun_adj_verb_ratio",
            "passive_verb_ratio",
            "Indice Honore's R",
            "Ratio Types/Tokens"
        ] if c in df_clean.columns
    ]

    experiments = {}

    experiments["full"] = all_features
    experiments["p1_only"] = ["Ratio_P1_rest"]
    experiments["no_p1"] = [c for c in all_features if c != "Ratio_P1_rest"]

    if len(syntax_block) > 0:
        experiments["no_syntax_block"] = [c for c in all_features if c not in syntax_block]

    if len(lexical_block) > 0:
        experiments["no_lexical_block"] = [c for c in all_features if c not in lexical_block]

    if len(emotion_block) > 0:
        experiments["no_emotion_block"] = [c for c in all_features if c not in emotion_block]

    if len(readability_block) > 0:
        experiments["no_readability_block"] = [c for c in all_features if c not in readability_block]

    if len(length_block) > 0:
        experiments["no_length_block"] = [c for c in all_features if c not in length_block]

    if len(nominal_downstream) > 0:
        experiments["no_nominal_lexical_downstream"] = [
            c for c in all_features if c not in nominal_downstream
        ]

    rows = []

    print("\n=== ABLATION RESULTS ===")
    for exp_name, feat_cols in experiments.items():
        if len(feat_cols) == 0:
            continue

        auc = cross_validated_auc(df_clean[feat_cols], y)

        rows.append({
            "experiment": exp_name,
            "n_features": len(feat_cols),
            "auc_oof": auc
        })

        print(f"{exp_name:35s} | n_features={len(feat_cols):2d} | AUC={auc:.4f}")

    if len(all_features) > 1:
        auc_resid = cross_validated_auc_residualized(
            df_clean=df_clean,
            feature_cols=all_features,
            target_col="source",
            residualize_on="Ratio_P1_rest",
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE
        )

        rows.append({
            "experiment": "residualized_without_p1",
            "n_features": len(all_features) - 1,
            "auc_oof": auc_resid
        })

        print(f"{'residualized_without_p1':35s} | n_features={len(all_features)-1:2d} | AUC={auc_resid:.4f}")

    results = pd.DataFrame(rows).sort_values("auc_oof", ascending=False).reset_index(drop=True)

    if "full" in results["experiment"].values:
        full_auc = results.loc[results["experiment"] == "full", "auc_oof"].iloc[0]
        results["delta_vs_full"] = results["auc_oof"] - full_auc

    print("\n=== ABLATION TABLE ===")
    print(results)

    save_df(results, "ablation_results.csv")

    return results


# ======================================================
# PART 7 — ABLATION PLOTTING
# ======================================================

def relabel_ablation_names(ablation_results):
    pretty_names = {
        "full": "Full model",
        "p1_only": "First-person ratio only",
        "no_p1": "Full model without first-person ratio",
        "no_syntax_block": "Full model without syntax block",
        "no_lexical_block": "Full model without lexical block",
        "no_emotion_block": "Full model without emotion block",
        "no_readability_block": "Full model without readability block",
        "no_length_block": "Full model without length block",
        "no_nominal_lexical_downstream": "Without nominal / lexical downstream features",
        "residualized_without_p1": "Residualized features (P1 removed)"
    }

    df = ablation_results.copy()
    df["experiment"] = df["experiment"].map(lambda x: pretty_names.get(x, x))
    return df


def plot_ablation_results(ablation_results, save_path=None, title=None, wrap_width=24):
    required_cols = {"experiment", "auc_oof"}
    missing = required_cols - set(ablation_results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = ablation_results.copy()
    df = df.sort_values("auc_oof", ascending=True).reset_index(drop=True)

    df["label_wrapped"] = df["experiment"].apply(
        lambda x: "\n".join(textwrap.wrap(str(x), width=wrap_width))
    )

    full_auc = None
    if "Full model" in df["experiment"].values:
        full_auc = df.loc[df["experiment"] == "Full model", "auc_oof"].iloc[0]
    elif "full" in df["experiment"].values:
        full_auc = df.loc[df["experiment"] == "full", "auc_oof"].iloc[0]

    fig_height = max(4, 0.65 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(df["label_wrapped"], df["auc_oof"])

    for i, (_, row) in enumerate(df.iterrows()):
        auc_txt = f"{row['auc_oof']:.3f}"

        if "delta_vs_full" in df.columns and pd.notnull(row.get("delta_vs_full", None)):
            delta = row["delta_vs_full"]
            if abs(delta) < 1e-12:
                delta_txt = " (ref)"
            else:
                delta_txt = f" ({delta:+.3f})"
        else:
            delta_txt = ""

        ax.text(
            row["auc_oof"] + 0.002,
            i,
            auc_txt + delta_txt,
            va="center",
            fontsize=10
        )

    if full_auc is not None:
        ax.axvline(full_auc, linestyle="--", linewidth=1)
        ax.text(
            full_auc,
            len(df) - 0.4,
            f" full = {full_auc:.3f}",
            ha="left",
            va="bottom",
            fontsize=10
        )

    xmin = max(0.45, df["auc_oof"].min() - 0.03)
    xmax = min(1.00, df["auc_oof"].max() + 0.06)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Out-of-fold AUC")
    ax.set_ylabel("Ablation")
    ax.set_title(title or "Ablation study: platform classification performance")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path is not None and SAVE_PLOTS:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("1) SPACY TEXT CLASSIFICATION")
    print("=" * 60)
    try:
        if GPU : 
            spacy.prefer_gpu()
        
        spacy_results = run_spacy_cv()
    except Exception as e:
        print(f"spaCy block failed: {e}")
        spacy_results = None

    print("\n" + "=" * 60)
    print("2) LOGIT + CROSS-FIT INFERENCE")
    print("=" * 60)
    try:
        logit_results = run_logit_with_crossfit_inference()
    except Exception as e:
        print(f"logit block failed: {e}")
        logit_results = None

    print("\n" + "=" * 60)
    print("3) T-TESTS")
    print("=" * 60)
    try:
        ttest_results = run_ttests()
    except Exception as e:
        print(f"t-tests block failed: {e}")
        ttest_results = None

    print("\n" + "=" * 60)
    print("4) ABLATIONS")
    print("=" * 60)
    try:
        ablation_results = run_ablations()

        pretty_ablation_results = relabel_ablation_names(ablation_results)
        plot_ablation_results(
            pretty_ablation_results,
            save_path=os.path.join(OUTPUT_DIR, "ablation_plot.png"),
            title="Ablation study for platform classification"
        )

        paper_rows = [
            "full",
            "p1_only",
            "no_p1",
            "no_nominal_lexical_downstream",
            "residualized_without_p1"
        ]

        paper_plot_df = ablation_results[
            ablation_results["experiment"].isin(paper_rows)
        ].copy()

        if not paper_plot_df.empty:
            paper_plot_df = relabel_ablation_names(paper_plot_df)
            plot_ablation_results(
                paper_plot_df,
                save_path=os.path.join(OUTPUT_DIR, "ablation_plot_paper.png"),
                title="Ablation study: testing the contribution of first-person narration"
            )

    except Exception as e:
        print(f"ablation block failed: {e}")
        ablation_results = None