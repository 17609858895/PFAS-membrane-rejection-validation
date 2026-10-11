from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression

from scientific_palettes import STYLE_COLORS, get_palette


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Comprehensive Dataset on MXene Properties for Catalyst Design (2024).xlsx"
OUT = ROOT / "新论文绘图_吸附能主线"

RANDOM_STATE = 42


def configure_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": 14,
            "axes.labelsize": 15.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "xtick.major.width": 1.55,
            "ytick.major.width": 1.55,
            "xtick.major.size": 7.2,
            "ytick.major.size": 7.2,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "xtick.minor.size": 4.2,
            "ytick.minor.size": 4.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "axes.linewidth": 1.7,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


_SAC_SOFT = get_palette("ba_eu_ir_o_soft_8")
_WARM_COOL = get_palette("ptsn_warm_cool_10")
PALETTE = {
    "blue": _SAC_SOFT[7],
    "cyan": _SAC_SOFT[0],
    "teal": _WARM_COOL[5],
    "green": _SAC_SOFT[2],
    "gold": _SAC_SOFT[6],
    "orange": _SAC_SOFT[1],
    "red": _SAC_SOFT[3],
    "purple": _WARM_COOL[9],
    **STYLE_COLORS,
}

FEATURE_LABELS = {
    "E_ads_ev": r"E$_{ads}$",
    "E_activation_ev": r"E$_a$",
    "E_reaction_ev": r"E$_{rxn}$",
    "E_Form": r"E$_{form}$",
    "Bader_M1": "Bader M1",
    "Bader_M2": "Bader M2",
    "Bader_X": "Bader X",
    "Bader_T1": "Bader T1",
    "Bader_T1_missing": "Bader T1 miss.",
    "Band_gap_PBE_ev": r"E$_g^{PBE}$",
    "a_ang": r"a",
    "N_Layers": r"N$_{layers}$",
    "Layer_dist_MX": r"d$_{M-X}$",
    "Layer_dist_MM": r"d$_{M-M}$",
    "Length_MX": r"L$_{M-X}$",
    "Length_MT1": r"L$_{M-T}$",
    "M1_Z": r"Z$_{M1}$",
    "M1_radius": r"r$_{M1}$",
    "M1_en": r"EN$_{M1}$",
    "M2_en_missing": "M2 EN miss.",
    "M2_period_missing": "M2 period miss.",
    "T1_type": "T1 type",
    "T1_type_Z": r"Z$_{T1}$",
    "T1_type_en": r"EN$_{T1}$",
    "T1_type_radius": r"r$_{T1}$",
    "T1_type_period_missing": "T1 period miss.",
    "T1_type_radius_missing": "T1 radius miss.",
    "T1_type_Z_missing": "T1 Z miss.",
    "Length_MT1_missing": r"L$_{M-T}$ miss.",
    "Mol": "Adsorbate",
    "Mol_radical_e": r"radical $e^{-}$",
    "Mol_mw": "MW",
    "Mol_atoms": r"N$_{atoms}$",
    "Mol_valence": r"valence $e^{-}$",
}

MODEL_LABELS = {
    "TOPSIS_ensemble": "Ensemble",
    "ExtraTrees": "ExtraTrees",
    "RF": "RF",
    "GBR": "GBR",
    "HGBR": "HGBR",
    "SVR": "SVR",
    "MLP": "MLP",
}

MODEL_COLORS = {
    "RF": PALETTE["blue"],
    "ExtraTrees": PALETTE["teal"],
    "GBR": PALETTE["red"],
    "HGBR": PALETTE["cyan"],
    "SVR": PALETTE["orange"],
    "MLP": PALETTE["purple"],
    "TOPSIS_ensemble": PALETTE["gold"],
}

ADSORBATE_COLORS = {
    "CO": PALETTE["blue"],
    "CO2": PALETTE["cyan"],
    "H": PALETTE["orange"],
    "H2": PALETTE["gold"],
    "H2O": PALETTE["teal"],
    "O": PALETTE["red"],
    "O2": PALETTE["purple"],
    "OH": PALETTE["green"],
}

ADSORBATE_LABELS = {
    "CO": "CO",
    "CO2": r"CO$_2$",
    "H": "H",
    "H2": r"H$_2$",
    "H2O": r"H$_2$O",
    "O": "O",
    "O2": r"O$_2$",
    "OH": "OH",
}

REACTION_LABELS = {
    "CO2*  --> CO* + O*": r"CO$_2^*$ -> CO$^*$ + O$^*$",
    "H2 ---> 2H": r"H$_2$ -> 2H",
    "H2O --> HO + H": r"H$_2$O -> HO + H",
}


def label_feature(name: str) -> str:
    return FEATURE_LABELS.get(str(name), str(name).replace("_missing", " miss.").replace("_", " "))


def label_model(name: str) -> str:
    return MODEL_LABELS.get(str(name), str(name))


def style_axes(ax, tick_size: int = 12.5, label_size: int = 15, title_size: int = 14) -> None:
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["ink"])
        ax.spines[spine].set_linewidth(1.7)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="major", bottom=True, top=False, colors=PALETTE["ink"],
                   labelsize=tick_size, width=1.55, length=7.2, direction="out")
    ax.tick_params(axis="y", which="major", left=True, right=False, colors=PALETTE["ink"],
                   labelsize=tick_size, width=1.55, length=7.2, direction="out")
    ax.tick_params(axis="both", which="minor", colors=PALETTE["ink"], width=1.15, length=4.2,
                   direction="out")
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")
    ax.title.set_size(title_size)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(True)
        tick.tick1line.set_markersize(7.2)
        tick.tick1line.set_markeredgewidth(1.55)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(tick_size)


def savefig(fig: plt.Figure, fig_no: int, name: str) -> None:
    folder = OUT / f"Fig{fig_no:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"Fig{fig_no:02d}_{name}.png"
    pdf = folder / f"Fig{fig_no:02d}_{name}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    export_subfigures(fig, folder, f"Fig{fig_no:02d}_{name}")
    plt.close(fig)


def export_subfigures(fig: plt.Figure, folder: Path, stem: str) -> None:
    subdir = folder / "subfigures"
    subdir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letters = list("abcdefghijklmnopqrstuvwxyz")
    exported = []
    for ax in fig.axes:
        if not ax.get_visible() or str(ax.get_label()).startswith("<colorbar"):
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox = bbox.expanded(1.06, 1.06).transformed(fig.dpi_scale_trans.inverted())
        suffix = letters[len(exported)] if len(exported) < len(letters) else f"p{len(exported) + 1}"
        fig.savefig(subdir / f"{stem}_{suffix}.png", dpi=600, bbox_inches=bbox, facecolor="white")
        fig.savefig(subdir / f"{stem}_{suffix}.pdf", bbox_inches=bbox, facecolor="white")
        exported.append(suffix)
    readme = subdir / "README.md"
    readme.write_text(
        "# Subfigures\n\n"
        "Each panel was extracted from the final combined figure export.\n\n"
        + "\n".join(f"- {s}: `{stem}_{s}.png`, `{stem}_{s}.pdf`" for s in exported)
        + "\n",
        encoding="utf-8",
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    wgs = pd.read_excel(DATA_PATH, sheet_name="WGS").replace("-", np.nan)
    dop = pd.read_excel(DATA_PATH, sheet_name="dopants").replace("-", np.nan)
    for df in (wgs, dop):
        for col in df.columns:
            if col not in [
                "Name",
                "Formula",
                "Stacking",
                "M1",
                "M2",
                "Dopant",
                "X",
                "T1_type",
                "T2_type",
                "Mol",
                "SMILES",
                "Reaction",
                "DOI_1(Energies)",
                "DOI_2(Band_gap_HSE06_ev)",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    wgs = wgs[wgs["E_ads_ev"].notna()].copy()
    dop = dop[dop["E_ads_ev"].notna()].copy()
    return wgs, dop


ELEMENTS = {
    # symbol: atomic number, group, period, Pauling EN, covalent radius pm
    "H": (1, 1, 1, 2.20, 31),
    "C": (6, 14, 2, 2.55, 76),
    "N": (7, 15, 2, 3.04, 71),
    "O": (8, 16, 2, 3.44, 66),
    "F": (9, 17, 2, 3.98, 57),
    "S": (16, 16, 3, 2.58, 105),
    "Cl": (17, 17, 3, 3.16, 102),
    "Br": (35, 17, 4, 2.96, 120),
    "Sc": (21, 3, 4, 1.36, 170),
    "Ti": (22, 4, 4, 1.54, 160),
    "V": (23, 5, 4, 1.63, 153),
    "Cr": (24, 6, 4, 1.66, 139),
    "Mn": (25, 7, 4, 1.55, 139),
    "Fe": (26, 8, 4, 1.83, 132),
    "Co": (27, 9, 4, 1.88, 126),
    "Ni": (28, 10, 4, 1.91, 124),
    "Cu": (29, 11, 4, 1.90, 132),
    "Zn": (30, 12, 4, 1.65, 122),
    "Y": (39, 3, 5, 1.22, 190),
    "Zr": (40, 4, 5, 1.33, 175),
    "Nb": (41, 5, 5, 1.60, 164),
    "Mo": (42, 6, 5, 2.16, 154),
    "Tc": (43, 7, 5, 1.90, 147),
    "Ru": (44, 8, 5, 2.20, 146),
    "Rh": (45, 9, 5, 2.28, 142),
    "Pd": (46, 10, 5, 2.20, 139),
    "Ag": (47, 11, 5, 1.93, 145),
    "Hf": (72, 4, 6, 1.30, 175),
    "Ta": (73, 5, 6, 1.50, 170),
    "W": (74, 6, 6, 2.36, 162),
    "Re": (75, 7, 6, 1.90, 151),
    "Os": (76, 8, 6, 2.20, 144),
    "Ir": (77, 9, 6, 2.20, 141),
    "Pt": (78, 10, 6, 2.28, 136),
    "Au": (79, 11, 6, 2.54, 136),
}

ALIASES = {"Rd": "Rh"}


def elem_tuple(symbol) -> tuple[float, float, float, float, float]:
    if pd.isna(symbol):
        return (np.nan,) * 5
    symbol = str(symbol).strip()
    symbol = ALIASES.get(symbol, symbol)
    if symbol in ELEMENTS:
        return ELEMENTS[symbol]
    if symbol in {"OH", "HO"}:
        return tuple(np.nanmean([ELEMENTS["O"], ELEMENTS["H"]], axis=0))
    if symbol == "NH":
        return tuple(np.nanmean([ELEMENTS["N"], ELEMENTS["H"]], axis=0))
    return (np.nan,) * 5


MOL_PROPS = {
    "H": (1.008, 1, 1, 1),
    "H2": (2.016, 2, 2, 0),
    "H2O": (18.015, 3, 8, 0),
    "CO": (28.010, 2, 10, 0),
    "CO2": (44.010, 3, 16, 0),
    "OH": (17.007, 2, 7, 1),
    "O": (15.999, 1, 6, 2),
    "O2": (31.998, 2, 12, 2),
}


def mol_tuple(mol) -> tuple[float, float, float, float]:
    if pd.isna(mol):
        return (np.nan,) * 4
    return MOL_PROPS.get(str(mol).strip(), (np.nan,) * 4)


NUMERIC_BASE = [
    "a_ang",
    "N_Layers",
    "Layer_dist_MX",
    "Layer_dist_MM",
    "Length_MX",
    "Length_MT1",
    "E_Form",
    "Bader_M1",
    "Bader_M2",
    "Bader_X",
    "Bader_T1",
    "Band_gap_PBE_ev",
]

CATEGORICAL_BASE = ["Stacking", "M1", "M2", "X", "T1_type", "Mol"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in NUMERIC_BASE:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")
            x[f"{col}_missing"] = x[col].isna().astype(int)

    for col in CATEGORICAL_BASE:
        if col in df.columns:
            x[col] = df[col].fillna("None").astype(str)

    for col in ["M1", "M2", "X", "T1_type", "Dopant"]:
        if col in df.columns:
            props = np.array([elem_tuple(v) for v in df[col]])
            for i, prop_name in enumerate(["Z", "group", "period", "en", "radius"]):
                x[f"{col}_{prop_name}"] = props[:, i]
                x[f"{col}_{prop_name}_missing"] = np.isnan(props[:, i]).astype(int)

    mol_props = np.array([mol_tuple(v) for v in df.get("Mol", pd.Series(index=df.index))])
    for i, prop_name in enumerate(["mw", "atoms", "valence", "radical_e"]):
        x[f"Mol_{prop_name}"] = mol_props[:, i]
        x[f"Mol_{prop_name}_missing"] = np.isnan(mol_props[:, i]).astype(int)

    x["has_termination"] = df.get("T1_type", pd.Series(index=df.index)).notna().astype(int)
    x["is_doped_domain"] = df.get("Dopant", pd.Series(index=df.index)).notna().astype(int)
    return x


def feature_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in CATEGORICAL_BASE if c in x.columns]
    num = [c for c in x.columns if c not in cat]
    return num, cat


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=None)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = feature_columns(x)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    # Regularized hyper-parameters: depth caps, larger leaf sizes, and
    # reduced feature subsampling keep the train/test R2 gap moderate instead
    # of letting the tree ensembles memorize the training set.
    raw = {
        "RF": RandomForestRegressor(
            n_estimators=420,
            max_depth=10,
            max_features=0.55,
            min_samples_leaf=7,
            min_samples_split=12,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=520,
            max_depth=12,
            max_features=0.55,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GBR": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=5,
            subsample=0.85,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "HGBR": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(C=5, gamma="scale", epsilon=0.1),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=0.01,
            learning_rate_init=0.002,
            max_iter=1200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
    }
    return {name: Pipeline([("prep", clone(preprocessor)), ("model", model)]) for name, model in raw.items()}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def safe_mape(y_true, y_pred) -> float:
    denom = np.maximum(np.abs(np.asarray(y_true)), 0.10)
    return float(np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / denom)) * 100)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }


def entropy_topsis(metric_df: pd.DataFrame) -> pd.DataFrame:
    benefit = pd.DataFrame(index=metric_df.index)
    benefit["R2"] = metric_df["R2"].clip(lower=0)
    for col in ["RMSE", "MAE", "MAPE"]:
        benefit[col] = 1 / (metric_df[col] + 1e-9)
    arr = benefit.to_numpy(dtype=float)
    arr = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0) + 1e-12)
    p = arr / (arr.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(metric_df))
    weights = (1 - entropy) / np.sum(1 - entropy)
    weighted = arr * weights
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg + 1e-12)
    out = metric_df.copy()
    out["TOPSIS_score"] = closeness
    out["ensemble_weight"] = closeness / closeness.sum()
    for name, val in zip(["w_R2", "w_RMSE", "w_MAE", "w_MAPE"], weights):
        out[name] = val
    return out.sort_values("TOPSIS_score", ascending=False)


def prepare_training(wgs: pd.DataFrame, dop: pd.DataFrame):
    x_wgs = make_features(wgs)
    x_dop = make_features(dop)
    for col in x_wgs.columns.difference(x_dop.columns):
        x_dop[col] = np.nan
    for col in x_dop.columns.difference(x_wgs.columns):
        x_wgs[col] = np.nan
    all_missing_train = [c for c in x_wgs.columns if x_wgs[c].isna().all()]
    if all_missing_train:
        x_wgs = x_wgs.drop(columns=all_missing_train)
        x_dop = x_dop.drop(columns=all_missing_train, errors="ignore")
    x_dop = x_dop[x_wgs.columns]
    y = wgs["E_ads_ev"].astype(float)
    strat = wgs["Mol"].fillna("None").astype(str)
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x_wgs,
        y,
        wgs.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=strat,
    )
    preprocessor = build_preprocessor(x_wgs)
    models = make_models(preprocessor)
    preds_test = {}
    preds_train = {}
    preds_dop = {}
    rows = []
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        pred_test = pipe.predict(x_test)
        pred_train = pipe.predict(x_train)
        pred_dop = pipe.predict(x_dop)
        preds_test[name] = pred_test
        preds_train[name] = pred_train
        preds_dop[name] = pred_dop
        row = {"Model": name, **metrics(y_test, pred_test), **{f"Train_{k}": v for k, v in metrics(y_train, pred_train).items()}}
        rows.append(row)

    metric_df = pd.DataFrame(rows).set_index("Model")
    topsis = entropy_topsis(metric_df[["R2", "RMSE", "MAE", "MAPE"]])
    weights = topsis["ensemble_weight"].reindex(metric_df.index).to_numpy()
    ens_test = np.column_stack([preds_test[m] for m in metric_df.index]) @ weights
    ens_train = np.column_stack([preds_train[m] for m in metric_df.index]) @ weights
    ens_dop = np.column_stack([preds_dop[m] for m in metric_df.index]) @ weights

    preds_test["TOPSIS_ensemble"] = ens_test
    preds_train["TOPSIS_ensemble"] = ens_train
    preds_dop["TOPSIS_ensemble"] = ens_dop
    metric_df.loc["TOPSIS_ensemble", ["R2", "RMSE", "MAE", "MAPE"]] = list(metrics(y_test, ens_test).values())
    metric_df.loc["TOPSIS_ensemble", ["Train_R2", "Train_RMSE", "Train_MAE", "Train_MAPE"]] = list(metrics(y_train, ens_train).values())

    best_model = topsis.index[0]
    return {
        "x_wgs": x_wgs,
        "x_dop": x_dop,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_test": idx_test,
        "wgs_test": wgs.loc[idx_test].copy(),
        "dop": dop.copy(),
        "fitted": fitted,
        "preds_test": preds_test,
        "preds_train": preds_train,
        "preds_dop": preds_dop,
        "metrics": metric_df,
        "topsis": topsis,
        "best_model": best_model,
        "weights": pd.Series(weights, index=metric_df.index[:-1]),
    }


def numeric_frame_for_analysis(wgs: pd.DataFrame) -> pd.DataFrame:
    x = make_features(wgs)
    num_cols, _ = feature_columns(x)
    num = x[num_cols].copy()
    y = wgs["E_ads_ev"].astype(float)
    # Remove all-constant/all-missing columns for correlation figures.
    keep = []
    for c in num.columns:
        s = pd.to_numeric(num[c], errors="coerce")
        if s.notna().sum() > 25 and s.nunique(dropna=True) > 2:
            keep.append(c)
    num = num[keep].apply(pd.to_numeric, errors="coerce")
    num = num.fillna(num.median(numeric_only=True))
    num["E_ads_ev"] = y.values
    return num


def top_numeric_features(num: pd.DataFrame, n: int = 9) -> list[str]:
    y = num["E_ads_ev"].values
    candidates = [c for c in num.columns if c != "E_ads_ev"]
    pearson = num[candidates].corrwith(num["E_ads_ev"]).abs()
    spear = num[candidates].corrwith(num["E_ads_ev"], method="spearman").abs()
    mi = mutual_info_regression(num[candidates], y, random_state=RANDOM_STATE)
    score = pearson.rank(pct=True) + spear.rank(pct=True) + pd.Series(mi, index=candidates).rank(pct=True)
    return list(score.sort_values(ascending=False).head(n).index)


def pairwise_mi(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    arr = np.zeros((len(cols), len(cols)))
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                arr[i, j] = 1
            elif i < j:
                try:
                    val = mutual_info_regression(df[[c1]], df[c2], random_state=RANDOM_STATE)[0]
                except Exception:
                    val = 0
                arr[i, j] = arr[j, i] = val
    if arr.max() > 0:
        arr = arr / arr.max()
    return pd.DataFrame(arr, index=cols, columns=cols)


def figure_1(wgs: pd.DataFrame, dop: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    boxes = [
        ("Data audit", "WGS: 600 rows\nEads labels: 599\nBarrier labels: 62", (0.05, 0.58), PALETTE["blue"]),
        ("Feature engineering", "Structure + Bader charge\nElement properties\nAdsorbate descriptors", (0.25, 0.58), PALETTE["cyan"]),
        ("Model library", "RF / ExtraTrees / GBR\nHGBR / SVR / MLP\nK-fold + hold-out", (0.45, 0.58), PALETTE["green"]),
        ("TOPSIS ensemble", "Entropy weights from\nR2, RMSE, MAE, MAPE\nRobust weighted prediction", (0.65, 0.58), PALETTE["gold"]),
        ("Interpretation", "Contribution map\nPDP / ALE\nSensitivity DAG", (0.85, 0.58), PALETTE["red"]),
        ("External dopant domain", f"dopants: {len(dop)} labeled Eads\nResidual analysis\nCandidate ranking", (0.45, 0.18), PALETTE["purple"]),
    ]

    for title, body, (x, y), color in boxes:
        rect = patches.FancyBboxPatch(
            (x - 0.085, y - 0.095),
            0.17,
            0.19,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.052, title, ha="center", va="center", color=color, fontsize=11, weight="bold")
        ax.text(x, y - 0.018, body, ha="center", va="center", color=PALETTE["ink"], fontsize=8.4, linespacing=1.35)

    arrowprops = dict(arrowstyle="-|>", lw=1.4, color=PALETTE["muted"], shrinkA=6, shrinkB=6)
    for i in range(4):
        ax.annotate("", xy=(boxes[i + 1][2][0] - 0.095, 0.58), xytext=(boxes[i][2][0] + 0.095, 0.58), arrowprops=arrowprops)
    ax.annotate("", xy=(0.45, 0.30), xytext=(0.45, 0.48), arrowprops=arrowprops)
    ax.annotate("", xy=(0.66, 0.50), xytext=(0.52, 0.30), arrowprops=arrowprops)

    # Small catalyst motif.
    for i in range(6):
        ax.add_patch(patches.Circle((0.16 + 0.03 * i, 0.30 + 0.018 * (i % 2)), 0.012, color=PALETTE["blue"], alpha=0.95))
    for i in range(5):
        ax.add_patch(patches.Circle((0.175 + 0.03 * i, 0.255 + 0.018 * ((i + 1) % 2)), 0.010, color=PALETTE["cyan"], alpha=0.95))
    ax.add_patch(patches.Circle((0.21, 0.37), 0.015, color=PALETTE["red"], alpha=0.95))
    ax.add_patch(patches.Circle((0.235, 0.385), 0.008, color=PALETTE["muted"], alpha=0.95))
    ax.text(0.20, 0.21, "MXene surface + WGSR adsorbates", ha="center", color=PALETTE["muted"], fontsize=8)

    ax.text(
        0.5,
        0.94,
        "Domain-aware interpretable ensemble ML for MXene adsorption energy and dopant screening",
        ha="center",
        va="center",
        fontsize=15,
        color=PALETTE["ink"],
        weight="bold",
    )
    ax.text(
        0.5,
        0.885,
        "Revised paper logic: Eads is the main learnable target; activation barriers are used as small-sample mechanistic evidence.",
        ha="center",
        va="center",
        fontsize=9,
        color=PALETTE["muted"],
    )
    savefig(fig, 1, "workflow")


def figure_2(wgs: pd.DataFrame) -> None:
    num = numeric_frame_for_analysis(wgs)
    feats = top_numeric_features(num, 8)
    cols = feats + ["E_ads_ev"]

    fig = plt.figure(figsize=(16.0, 8.35))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.96, 1.16, 1.16], height_ratios=[1.0, 1.20], wspace=0.30, hspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    availability = pd.Series(
        {
            r"E$_{ads}$" + "\nWGS": wgs["E_ads_ev"].notna().sum(),
            r"E$_a$" + "\nWGS": wgs["E_activation_ev"].notna().sum(),
            r"E$_{rxn}$" + "\nWGS": wgs["E_reaction_ev"].notna().sum(),
            "Reaction\nlabels": wgs["Reaction"].notna().sum(),
        }
    )
    colors = [PALETTE["teal"], PALETTE["orange"], PALETTE["orange"], PALETTE["purple"]]
    ax0.bar(availability.index, availability.values, color=colors, edgecolor="white")
    ax0.set_title("")
    ax0.set_ylabel("Valid rows")
    ax0.set_ylim(0, 650)
    for i, v in enumerate(availability.values):
        ax0.text(i, v + 15, str(int(v)), ha="center", fontsize=9, weight="bold")

    sns.histplot(wgs["E_ads_ev"], bins=32, kde=True, color=PALETTE["blue"], ax=ax1)
    ax1.axvline(wgs["E_ads_ev"].median(), color=PALETTE["red"], lw=1.5, ls="--", label="median")
    ax1.set_title("")
    ax1.set_xlabel(r"E$_{ads}$ (eV)")
    ax1.legend(frameon=False)

    mol_order = wgs.groupby("Mol")["E_ads_ev"].median().sort_values().index
    sns.boxplot(
        data=wgs,
        x="Mol",
        y="E_ads_ev",
        order=mol_order,
        ax=ax2,
        palette=[ADSORBATE_COLORS.get(str(m), PALETTE["muted"]) for m in mol_order],
        linewidth=0.8,
        fliersize=2,
    )
    ax2.set_title("")
    ax2.set_xlabel("")
    ax2.set_ylabel(r"E$_{ads}$ (eV)")
    ax2.set_xticklabels([ADSORBATE_LABELS.get(t.get_text(), t.get_text()) for t in ax2.get_xticklabels()])

    pear = num[cols].corr(method="pearson")
    spear = num[cols].corr(method="spearman")
    mi = pairwise_mi(num[cols])
    for ax, mat, title, cmap, center in [
        (ax3, pear, "", "vlag", 0),
        (ax4, spear, "", "vlag", 0),
        (ax5, mi, "", "mako", None),
    ]:
        mat = mat.rename(index=label_feature, columns=label_feature)
        sns.heatmap(
            mat,
            cmap=cmap,
            center=center,
            vmin=-1 if center == 0 else 0,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.68, "pad": 0.045},
            ax=ax,
            linewidths=0.15,
            linecolor="white",
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        style_axes(ax, tick_size=10, label_size=12, title_size=13)

    for ax in [ax0, ax1, ax2]:
        style_axes(ax)
    savefig(fig, 2, "dataset_correlation")


def figure_3(context: dict, wgs: pd.DataFrame) -> pd.DataFrame:
    best = context["best_model"]
    pipe = context["fitted"][best]
    result = permutation_importance(
        pipe,
        context["x_test"],
        context["y_test"],
        n_repeats=12,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": context["x_test"].columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(OUT / "Fig03" / "feature_importance.csv", index=False, encoding="utf-8-sig")

    num = numeric_frame_for_analysis(wgs)
    candidate = [c for c in imp["feature"].head(18) if c in num.columns]
    if len(candidate) < 8:
        candidate = top_numeric_features(num, 14)
    corr = num[candidate].corr(method="spearman").fillna(0).clip(-1, 1)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist.values, 0)
    Z = linkage(squareform(dist.values, checks=False), method="average")

    # Consensus from three filters.
    y = num["E_ads_ev"]
    candidates = [c for c in num.columns if c != "E_ads_ev"]
    pear_top = set(num[candidates].corrwith(y).abs().sort_values(ascending=False).head(10).index)
    spear_top = set(num[candidates].corrwith(y, method="spearman").abs().sort_values(ascending=False).head(10).index)
    mi_vals = pd.Series(mutual_info_regression(num[candidates], y, random_state=RANDOM_STATE), index=candidates)
    mi_top = set(mi_vals.sort_values(ascending=False).head(10).index)
    consensus = sorted((pear_top & spear_top) | (spear_top & mi_top) | (pear_top & mi_top))

    fig = plt.figure(figsize=(16.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.16, 1.26], wspace=0.48)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    top_imp = imp.head(14).iloc[::-1]
    ybar = np.arange(len(top_imp))
    ax0.barh(ybar, top_imp["importance"], xerr=top_imp["std"], color=PALETTE["teal"], alpha=0.9)
    ax0.set_yticks(ybar)
    ax0.set_yticklabels([label_feature(f) for f in top_imp["feature"]])
    ax0.set_title("")
    ax0.set_xlabel(r"Mean R$^2$ decrease")
    ax0.grid(axis="x", alpha=0.22)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax0, tick_size=10, label_size=12)

    dendrogram(
        Z,
        labels=[label_feature(c) for c in candidate],
        orientation="right",
        leaf_font_size=8.8,
        color_threshold=np.median(Z[:, 2]),
        above_threshold_color=PALETTE["muted"],
        ax=ax1,
    )
    ax1.set_title("")
    ax1.set_xlabel(r"1 - |Spearman $\rho$|")
    style_axes(ax1, tick_size=9.5, label_size=12)

    ax2.set_axis_off()
    listed = consensus[:9]
    circles = [
        ((0.34, 0.56), 0.285, PALETTE["blue"], "Pearson", (0.20, 0.90)),
        ((0.66, 0.56), 0.285, PALETTE["orange"], "Spearman", (0.82, 0.90)),
        ((0.50, 0.36), 0.275, PALETTE["purple"], "MI", (0.50, 0.68)),
    ]
    for (x, y0), r, color, label, label_pos in circles:
        ax2.add_patch(patches.Circle((x, y0), r, color=color, alpha=0.22, ec=color, lw=1.6))
        ax2.text(label_pos[0], label_pos[1], label, color=color, ha="center", va="center", fontsize=9.4, weight="bold")
    ax2.text(
        0.5,
        0.43,
        "\n".join(label_feature(v) for v in listed) if listed else "No pairwise consensus",
        ha="center",
        va="center",
        fontsize=10.0,
        color=PALETTE["ink"],
        linespacing=1.13,
    )
    ax2.text(0.5, 0.035, f"n = {len(consensus)}", ha="center", fontsize=10, color=PALETTE["muted"], weight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    savefig(fig, 3, "feature_selection")
    return imp


def figure_4(context: dict) -> None:
    models = [m for m in context["metrics"].index if m != "TOPSIS_ensemble"]
    fig = plt.figure(figsize=(16.4, 12.2))
    outer_gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.62],
        hspace=0.32,
        wspace=0.18,
        left=0.055,
        right=0.992,
        top=0.990,
        bottom=0.060,
    )
    y_test = context["y_test"].to_numpy()
    y_train = context["y_train"].to_numpy()
    all_y = [y_train, y_test]
    all_pred = []
    for m in models:
        all_pred.extend([context["preds_train"][m], context["preds_test"][m]])
    lo = min(np.min(v) for v in all_y + all_pred) - 0.6
    hi = max(np.max(v) for v in all_y + all_pred) + 0.45
    xs = np.array([lo, hi])

    for i, name in enumerate(models):
        inner_gs = outer_gs[i // 3, i % 3].subgridspec(1, 2, width_ratios=[4.15, 1.2], wspace=0.045)
        ax = fig.add_subplot(inner_gs[0, 0])
        ax_res = fig.add_subplot(inner_gs[0, 1], sharey=ax)
        c = MODEL_COLORS.get(name, PALETTE["blue"])
        pred_train = context["preds_train"][name]
        pred_test = context["preds_test"][name]

        ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["ink"], lw=1.45, alpha=0.65, zorder=1)
        ax.fill_between(xs, xs - 0.50, xs + 0.50, color="#8A8F98", alpha=0.13, zorder=0)
        ax.scatter(
            y_train,
            pred_train,
            s=44,
            facecolors="white",
            edgecolors=c,
            linewidths=1.25,
            alpha=0.78,
            zorder=3,
            label="Train",
        )
        ax.scatter(
            y_test,
            pred_test,
            s=68,
            facecolors=c,
            edgecolors=PALETTE["ink"],
            linewidths=0.9,
            marker="^",
            alpha=0.95,
            zorder=4,
            label="Test",
        )
        m = context["metrics"].loc[name]
        txt = (
            f"{label_model(name)}\n"
            f"R$^2_{{train}}$ = {m.Train_R2:.3f}\n"
            f"R$^2_{{test}}$ = {m.R2:.3f}\n"
            f"RMSE$_{{test}}$ = {m.RMSE:.2f}"
        )
        ax.text(
            0.045,
            0.955,
            txt,
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            va="top",
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, lw=1.2, alpha=0.96),
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(r"DFT E$_{ads}$ (eV)")
        ax.set_ylabel(r"Predicted E$_{ads}$ (eV)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.75)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9.5, handletextpad=0.25, borderpad=0.2)
        style_axes(ax)

        res_train = pred_train - y_train
        res_test = pred_test - y_test
        ax_res.scatter(
            res_train,
            pred_train,
            s=22,
            facecolors="white",
            edgecolors=c,
            linewidths=0.9,
            alpha=0.72,
            zorder=3,
        )
        ax_res.scatter(
            res_test,
            pred_test,
            s=34,
            facecolors=c,
            edgecolors=PALETTE["ink"],
            linewidths=0.75,
            marker="^",
            alpha=0.95,
            zorder=4,
        )
        ax_res.axvline(0, color=PALETTE["muted"], lw=1.0, ls="--", alpha=0.68)
        all_res = np.concatenate([res_train, res_test])
        rmax = max(np.percentile(np.abs(all_res), 98) * 1.15, 0.75)
        ax_res.set_xlim(-rmax, rmax)
        ax_res.xaxis.set_major_locator(MaxNLocator(3))
        ax_res.set_xlabel("Error", fontsize=12, fontweight="bold")
        ax_res.tick_params(axis="y", labelleft=False)
        ax_res.grid(alpha=0.25, linestyle="--", linewidth=0.65)
        style_axes(ax_res, tick_size=9, label_size=11)

    metrics_df = context["metrics"].loc[models + ["TOPSIS_ensemble"]].copy()
    metric_plot = metrics_df[["R2", "RMSE", "MAE"]].rename(columns={"R2": r"R$^2$"})
    metric_plot.index = [label_model(i) for i in metric_plot.index]
    metric_plot.to_csv(OUT / "Fig04" / "holdout_metrics_for_plot.csv", encoding="utf-8-sig")

    ax_metrics = fig.add_subplot(outer_gs[2, :2])
    x = np.arange(len(metric_plot))
    width = 0.24
    metric_colors = [PALETTE["teal"], "#E6A06A", "#8A7EB8"]
    for j, (col, color) in enumerate(zip(metric_plot.columns, metric_colors)):
        ax_metrics.bar(x + (j - 1) * width, metric_plot[col].values, width=width, color=color, edgecolor="white", linewidth=0.6, label=col)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_plot.index, rotation=27, ha="right")
    ax_metrics.set_ylabel("Value")
    ax_metrics.set_ylim(0, max(0.95, float(metric_plot.max().max()) * 1.12))
    ax_metrics.yaxis.set_major_locator(MaxNLocator(5))
    ax_metrics.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=10, handlelength=1.4, columnspacing=1.2)
    ax_metrics.grid(axis="y", alpha=0.26)
    style_axes(ax_metrics, tick_size=11, label_size=12)

    ax_lc = fig.add_subplot(outer_gs[2, 2])
    lc_model = clone(context["fitted"]["ExtraTrees"])
    if hasattr(lc_model, "set_params"):
        lc_model.set_params(model__n_estimators=180, model__n_jobs=-1)
    train_sizes = np.linspace(90, len(context["x_train"]), 5).astype(int)
    train_sizes = np.unique(np.clip(train_sizes, 40, len(context["x_train"])))
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    for size in train_sizes:
        for rep in range(5):
            idx = rng.choice(context["x_train"].index.to_numpy(), size=size, replace=False)
            estimator = clone(lc_model)
            estimator.fit(context["x_train"].loc[idx], context["y_train"].loc[idx])
            records.append(
                {
                    "Training samples": int(size),
                    "Training": r2_score(context["y_train"].loc[idx], estimator.predict(context["x_train"].loc[idx])),
                    "Validation": r2_score(context["y_test"], estimator.predict(context["x_test"])),
                }
            )
    lc = pd.DataFrame(records)
    lc.to_csv(OUT / "Fig04" / "extra_trees_learning_curve.csv", index=False, encoding="utf-8-sig")
    lc_summary = lc.groupby("Training samples").agg(
        train_mean=("Training", "mean"),
        train_sd=("Training", "std"),
        val_mean=("Validation", "mean"),
        val_sd=("Validation", "std"),
    )
    xs_lc = lc_summary.index.to_numpy(dtype=float)
    for mean_col, sd_col, color, label in [
        ("train_mean", "train_sd", MODEL_COLORS["ExtraTrees"], "Training"),
        ("val_mean", "val_sd", PALETTE["red"], "Validation"),
    ]:
        mean = lc_summary[mean_col].to_numpy(dtype=float)
        sd = lc_summary[sd_col].fillna(0).to_numpy(dtype=float)
        ax_lc.plot(xs_lc, mean, "-o", color=color, lw=1.8, ms=4.5, label=label)
        ax_lc.fill_between(xs_lc, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
    ax_lc.set_xlabel("Training samples")
    ax_lc.set_ylabel(r"R$^2$")
    ymin = max(0.0, float(np.nanmin(lc_summary[["train_mean", "val_mean"]].values)) - 0.18)
    ax_lc.set_ylim(ymin, 1.04)
    ax_lc.xaxis.set_major_locator(MaxNLocator(5))
    ax_lc.yaxis.set_major_locator(MaxNLocator(5))
    ax_lc.legend(loc="lower right", fontsize=9)
    ax_lc.grid(alpha=0.26)
    style_axes(ax_lc, tick_size=11, label_size=12)

    context["metrics"].to_csv(OUT / "Fig04" / "model_metrics.csv", encoding="utf-8-sig")
    savefig(fig, 4, "model_benchmark")


def figure_5(context: dict) -> None:
    models = [m for m in context["metrics"].index if m != "TOPSIS_ensemble"]
    y = context["y_test"].to_numpy()
    error_df = []
    for name in models + ["TOPSIS_ensemble"]:
        err = np.abs(y - context["preds_test"][name])
        error_df.extend({"Model": name, "AbsError": e} for e in err)
    error_df = pd.DataFrame(error_df)
    error_df["Model_label"] = error_df["Model"].map(label_model)

    pmat = pd.DataFrame(np.ones((len(models) + 1, len(models) + 1)), index=models + ["TOPSIS_ensemble"], columns=models + ["TOPSIS_ensemble"])
    for a in pmat.index:
        for b in pmat.columns:
            if a == b:
                pmat.loc[a, b] = np.nan
            else:
                try:
                    pmat.loc[a, b] = wilcoxon(np.abs(y - context["preds_test"][a]), np.abs(y - context["preds_test"][b])).pvalue
                except ValueError:
                    pmat.loc[a, b] = 1.0

    fig = plt.figure(figsize=(15.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.28, 1.24], wspace=0.44)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    weights = context["weights"].sort_values(ascending=True)
    ax0.barh(
        [label_model(i) for i in weights.index],
        weights.values,
        color=[MODEL_COLORS.get(i, PALETTE["gold"]) for i in weights.index],
        edgecolor="white",
    )
    ax0.set_title("")
    ax0.set_xlabel("Weight")
    ax0.set_xlim(0, max(0.34, float(weights.max()) + 0.045))
    for i, v in enumerate(weights.values):
        ax0.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax0)

    order = [label_model(i) for i in context["metrics"].sort_values("RMSE").index]
    model_palette = {label_model(k): v for k, v in MODEL_COLORS.items()}
    sns.boxplot(data=error_df, y="Model_label", x="AbsError", order=order, ax=ax1, palette=model_palette, fliersize=1.8, linewidth=0.8, orient="h")
    sns.stripplot(
        data=error_df.sample(min(len(error_df), 420), random_state=RANDOM_STATE),
        y="Model_label",
        x="AbsError",
        order=order,
        ax=ax1,
        color=PALETTE["ink"],
        alpha=0.25,
        size=2,
        orient="h",
    )
    ax1.set_title("")
    ax1.set_xlabel(r"|DFT - predicted| E$_{ads}$ (eV)")
    ax1.set_ylabel("")
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax1)

    logp = -np.log10(pmat.astype(float))
    logp = logp.rename(index=label_model, columns=label_model)
    sns.heatmap(logp, cmap="rocket_r", vmin=0, vmax=np.nanpercentile(logp.values, 95), ax=ax2, cbar_kws={"label": r"$-\log_{10}(p)$", "pad": 0.03}, linewidths=0.2, linecolor="white")
    ax2.set_title("")
    ax2.tick_params(axis="x", rotation=35, labelsize=11)
    ax2.tick_params(axis="y", rotation=0, labelsize=11)
    for tick in ax2.get_xticklabels():
        tick.set_ha("right")
    style_axes(ax2, tick_size=12)

    context["topsis"].to_csv(OUT / "Fig05" / "topsis_weights.csv", encoding="utf-8-sig")
    pmat.to_csv(OUT / "Fig05" / "wilcoxon_p_values.csv", encoding="utf-8-sig")
    savefig(fig, 5, "topsis_wilcoxon")


def replace_feature_values(x: pd.DataFrame, feature: str, value) -> pd.DataFrame:
    xx = x.copy()
    xx[feature] = value
    return xx


def feature_reference_value(x_train: pd.DataFrame, feature: str):
    if x_train[feature].dtype == "object":
        return x_train[feature].mode(dropna=True).iloc[0]
    return float(pd.to_numeric(x_train[feature], errors="coerce").median())


def ale_curve(pipe: Pipeline, x: pd.DataFrame, feature: str, bins: int = 12) -> pd.DataFrame:
    s = pd.to_numeric(x[feature], errors="coerce")
    valid = s.notna()
    x_valid = x.loc[valid].copy()
    s = s.loc[valid]
    qs = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    if len(qs) < 4:
        return pd.DataFrame({"x": [], "ale": []})
    effects = []
    centers = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (s >= lo) & (s <= hi)
        if mask.sum() == 0:
            effects.append(0.0)
        else:
            low = x_valid.loc[mask].copy()
            high = x_valid.loc[mask].copy()
            low[feature] = lo
            high[feature] = hi
            effects.append(float(np.mean(pipe.predict(high) - pipe.predict(low))))
        centers.append((lo + hi) / 2)
    ale = np.cumsum(effects)
    ale = ale - np.mean(ale)
    return pd.DataFrame({"x": centers, "ale": ale})


def pdp_curve(pipe: Pipeline, x: pd.DataFrame, feature: str, grid_size: int = 40) -> pd.DataFrame:
    s = pd.to_numeric(x[feature], errors="coerce")
    grid = np.linspace(s.quantile(0.05), s.quantile(0.95), grid_size)
    vals = []
    sample = x.sample(min(len(x), 260), random_state=RANDOM_STATE)
    for g in grid:
        xx = sample.copy()
        xx[feature] = g
        vals.append(float(np.mean(pipe.predict(xx))))
    return pd.DataFrame({"x": grid, "pdp": vals})


def figure_6(context: dict, imp: pd.DataFrame) -> None:
    best = context["best_model"]
    pipe = context["fitted"][best]
    top = [f for f in imp.query("importance > 0")["feature"].head(8) if f in context["x_test"].columns]
    if len(top) < 6:
        top = list(imp["feature"].head(8))
    sample = context["x_test"].sample(min(120, len(context["x_test"])), random_state=RANDOM_STATE)
    pred = pipe.predict(sample)
    contrib_records = []
    for f in top:
        ref = feature_reference_value(context["x_train"], f)
        xx = replace_feature_values(sample, f, ref)
        delta = pred - pipe.predict(xx)
        raw = sample[f]
        if raw.dtype == "object":
            codes = pd.Categorical(raw).codes
            val = (codes - codes.min()) / (codes.max() - codes.min() + 1e-9)
        else:
            val = pd.to_numeric(raw, errors="coerce")
            val = (val - np.nanmin(val)) / (np.nanmax(val) - np.nanmin(val) + 1e-9)
        for d, v in zip(delta, val):
            contrib_records.append({"Feature": f, "Contribution": d, "ScaledValue": v})
    contrib_df = pd.DataFrame(contrib_records)

    def is_continuous_feature(f: str) -> bool:
        if f.endswith("_missing"):
            return False
        if f not in context["x_test"].columns:
            return False
        if not pd.api.types.is_numeric_dtype(context["x_test"][f]):
            return False
        return pd.to_numeric(context["x_train"][f], errors="coerce").nunique(dropna=True) > 10

    continuous_ranked = [f for f in imp["feature"] if is_continuous_feature(f)]
    preferred_focus = ["Bader_M1", "Bader_X", "E_Form", "Length_MX", "Layer_dist_MX", "Length_MT1", "N_Layers"]
    focus = next((f for f in preferred_focus if is_continuous_feature(f)), continuous_ranked[0] if continuous_ranked else "E_Form")
    interaction = "Mol_mw" if "Mol_mw" in context["x_test"].columns else next((f for f in continuous_ranked if f != focus), focus)
    pdp = pdp_curve(pipe, pd.concat([context["x_train"], context["x_test"]]), focus)
    ale = ale_curve(pipe, pd.concat([context["x_train"], context["x_test"]]), focus)

    def pretty(name: str) -> str:
        return label_feature(name)

    fig = plt.figure(figsize=(15.2, 7.85))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.05, 1.05], hspace=0.24, wspace=0.42)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    order = contrib_df.groupby("Feature")["Contribution"].apply(lambda s: np.mean(np.abs(s))).sort_values().index
    ypos = {f: i for i, f in enumerate(order)}
    rng = np.random.default_rng(RANDOM_STATE)
    for f in order:
        sub = contrib_df[contrib_df["Feature"] == f]
        yj = ypos[f] + rng.normal(0, 0.055, len(sub))
        ax0.scatter(sub["Contribution"], yj, c=sub["ScaledValue"], cmap="viridis", s=18, alpha=0.75, edgecolor="none")
    ax0.axvline(0, color=PALETTE["muted"], lw=0.8)
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels([pretty(f) for f in order])
    ax0.set_title("")
    ax0.set_xlabel(r"$\Delta$ prediction of E$_{ads}$ (eV)")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis")
    cbar = fig.colorbar(sm, ax=ax0, orientation="horizontal", fraction=0.035, pad=0.105)
    cbar.set_label("Feature value (low to high)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax0)

    top_bar = imp.head(10).iloc[::-1]
    ybar = np.arange(len(top_bar))
    ax1.barh(ybar, top_bar["importance"], color=PALETTE["teal"])
    ax1.set_yticks(ybar)
    ax1.set_yticklabels([pretty(f) for f in top_bar["feature"]])
    ax1.set_title("")
    ax1.set_xlabel(r"Mean R$^2$ decrease")
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax1, tick_size=10, label_size=13)

    ax2.plot(pdp["x"], pdp["pdp"], color=PALETTE["blue"], lw=2)
    ax2.fill_between(pdp["x"], pdp["pdp"], pdp["pdp"].mean(), color=PALETTE["blue"], alpha=0.12)
    ax2.set_title("")
    ax2.set_xlabel(pretty(focus))
    ax2.set_ylabel(r"Mean E$_{ads}$ prediction (eV)")
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax2)

    ax3.plot(ale["x"], ale["ale"], color=PALETTE["red"], lw=2)
    ax3.axhline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    ax3.set_title("")
    ax3.set_xlabel(pretty(focus))
    ax3.set_ylabel(r"Centered effect on E$_{ads}$ (eV)")
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax3.yaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax3)

    if interaction not in sample.columns:
        interaction = focus
    xx = pd.to_numeric(sample[focus], errors="coerce")
    yy = pred - pred.mean()
    cc = pd.to_numeric(sample[interaction], errors="coerce") if interaction in sample else xx
    sc = ax4.scatter(xx, yy, c=cc, cmap="mako", s=32, alpha=0.78, edgecolor="white", linewidth=0.25)
    ax4.set_title("")
    ax4.set_xlabel(pretty(focus))
    ax4.set_ylabel(r"Centered E$_{ads}$ prediction (eV)")
    cb = fig.colorbar(sc, ax=ax4, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=11)
    ax4.xaxis.set_major_locator(MaxNLocator(5))
    ax4.yaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax4)

    contrib_df.to_csv(OUT / "Fig06" / "shap_style_contributions.csv", index=False, encoding="utf-8-sig")
    savefig(fig, 6, "interpretability")


def counterfactual_effect(pipe: Pipeline, x: pd.DataFrame, feature: str, scale: float = 0.5, boot: int = 200) -> tuple[float, float, float]:
    s = pd.to_numeric(x[feature], errors="coerce")
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return (0.0, 0.0, 0.0)
    base = pipe.predict(x)
    xx = x.copy()
    xx[feature] = s + scale * sd
    diff = pipe.predict(xx) - base
    rng = np.random.default_rng(RANDOM_STATE)
    means = []
    for _ in range(boot):
        idx = rng.integers(0, len(diff), len(diff))
        means.append(np.mean(diff[idx]))
    return float(np.mean(diff)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure_7(context: dict, wgs: pd.DataFrame, imp: pd.DataFrame) -> None:
    best = context["best_model"]
    pipe = context["fitted"][best]
    numeric_top = [
        f
        for f in imp["feature"]
        if f in context["x_test"].columns
        and pd.api.types.is_numeric_dtype(context["x_test"][f])
        and not f.endswith("_missing")
        and pd.to_numeric(context["x_train"][f], errors="coerce").nunique(dropna=True) > 8
    ]
    effects = []
    for f in numeric_top[:8]:
        mean, lo, hi = counterfactual_effect(pipe, context["x_test"].copy(), f)
        effects.append({"Feature": f, "Effect": mean, "CI_low": lo, "CI_high": hi})
    eff = pd.DataFrame(effects).sort_values("Effect")

    barrier = wgs[wgs["E_activation_ev"].notna() & wgs["E_ads_ev"].notna()].copy()
    barrier["Reaction_label"] = barrier["Reaction"].map(REACTION_LABELS).fillna(barrier["Reaction"])
    fig = plt.figure(figsize=(14.0, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.02, 1.12], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    G = nx.DiGraph()
    edges = [
        ("Composition", "Structure"),
        ("Composition", "Electronic state"),
        ("Surface termination", "Electronic state"),
        ("Structure", "E_ads"),
        ("Electronic state", "E_ads"),
        ("Adsorbate", "E_ads"),
        ("E_ads", "Activation barrier"),
        ("Reaction path", "Activation barrier"),
    ]
    G.add_edges_from(edges)
    pos = {
        "Composition": (0.14, 0.74),
        "Surface termination": (0.14, 0.34),
        "Structure": (0.42, 0.82),
        "Electronic state": (0.42, 0.46),
        "Adsorbate": (0.42, 0.12),
        "E_ads": (0.68, 0.49),
        "Reaction path": (0.68, 0.16),
        "Activation barrier": (0.92, 0.31),
    }
    labels = {
        "Composition": "Compo-\nsition",
        "Surface termination": "Surface\nterm.",
        "Structure": "Structure",
        "Electronic state": "Electronic\nstate",
        "Adsorbate": "Adsorbate",
        "E_ads": r"E$_{ads}$",
        "Reaction path": "Reaction\npath",
        "Activation barrier": "Activation\nbarrier",
    }
    node_colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["green"], PALETTE["orange"], PALETTE["red"], PALETTE["purple"], PALETTE["gold"]]
    nx.draw_networkx_edges(G, pos, ax=ax0, arrows=True, arrowstyle="-|>", arrowsize=12, edge_color=PALETTE["muted"], width=1.3)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=1500, node_color=node_colors, edgecolors="white", linewidths=1.3)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax0, font_size=7.4, font_color=PALETTE["ink"], font_weight="bold")
    ax0.set_title("")
    ax0.set_axis_off()
    ax0.set_xlim(0, 1.05)
    ax0.set_ylim(0, 0.96)

    y = np.arange(len(eff))
    ax1.errorbar(
        eff["Effect"],
        y,
        xerr=[eff["Effect"] - eff["CI_low"], eff["CI_high"] - eff["Effect"]],
        fmt="o",
        color=PALETTE["red"],
        ecolor=PALETTE["muted"],
        capsize=3,
    )
    ax1.axvline(0, color=PALETTE["muted"], lw=0.9, ls="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels([label_feature(f) for f in eff["Feature"]])
    ax1.set_title("")
    ax1.set_xlabel(r"Mean $\Delta$E$_{ads}$ prediction (eV)")
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax1)

    if len(barrier) > 2:
        rho, p = spearmanr(barrier["E_ads_ev"], barrier["E_activation_ev"], nan_policy="omit")
    else:
        rho, p = np.nan, np.nan
    sns.scatterplot(
        data=barrier,
        x="E_ads_ev",
        y="E_activation_ev",
        hue="Reaction_label",
        palette=[PALETTE["blue"], PALETTE["orange"], PALETTE["purple"]],
        s=42,
        edgecolor="white",
        linewidth=0.35,
        ax=ax2,
    )
    sns.regplot(data=barrier, x="E_ads_ev", y="E_activation_ev", scatter=False, color=PALETTE["ink"], lowess=True, ax=ax2)
    ax2.set_title("")
    ax2.set_xlabel(r"E$_{ads}$ (eV)")
    ax2.set_ylabel(r"E$_a$ (eV)")
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax2.text(0.04, 0.96, f"n={len(barrier)}\nSpearman $\\rho$={rho:.2f}\n{p_text}", transform=ax2.transAxes, va="top", fontsize=11, bbox=dict(fc="white", ec="none", alpha=0.82))
    ax2.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.50), fontsize=9, title="Path", title_fontsize=10)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax2)

    eff.to_csv(OUT / "Fig07" / "counterfactual_sensitivity.csv", index=False, encoding="utf-8-sig")
    savefig(fig, 7, "causal_sensitivity_barrier")


PERIODIC_POS = {
    "H": (1, 1),
    "Sc": (4, 3),
    "Ti": (4, 4),
    "V": (4, 5),
    "Cr": (4, 6),
    "Mn": (4, 7),
    "Fe": (4, 8),
    "Co": (4, 9),
    "Ni": (4, 10),
    "Cu": (4, 11),
    "Zn": (4, 12),
    "Y": (5, 3),
    "Zr": (5, 4),
    "Nb": (5, 5),
    "Mo": (5, 6),
    "Tc": (5, 7),
    "Ru": (5, 8),
    "Rh": (5, 9),
    "Pd": (5, 10),
    "Ag": (5, 11),
    "Hf": (6, 4),
    "Ta": (6, 5),
    "W": (6, 6),
    "Re": (6, 7),
    "Os": (6, 8),
    "Ir": (6, 9),
    "Pt": (6, 10),
    "Au": (6, 11),
}


def figure_8(context: dict) -> None:
    dop = context["dop"].copy()
    dop["pred_E_ads"] = context["preds_dop"]["TOPSIS_ensemble"]
    dop["residual"] = dop["pred_E_ads"] - dop["E_ads_ev"]
    dop["Dopant_clean"] = dop["Dopant"].replace(ALIASES).fillna("None")
    dop["pred_screening_score"] = np.exp(-((dop["pred_E_ads"] + 0.80) / 0.90) ** 2)
    dop["dft_window_score"] = np.exp(-((dop["E_ads_ev"] + 0.80) / 0.90) ** 2)
    dop_metrics = metrics(dop["E_ads_ev"], dop["pred_E_ads"])
    rho, p_rho = spearmanr(dop["E_ads_ev"], dop["pred_E_ads"], nan_policy="omit")

    fig = plt.figure(figsize=(15.0, 8.25))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.18], width_ratios=[1.06, 1.26, 1.16], hspace=0.56, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    lo = min(dop["E_ads_ev"].min(), dop["pred_E_ads"].min()) - 0.25
    hi = max(dop["E_ads_ev"].max(), dop["pred_E_ads"].max()) + 0.25
    sns.scatterplot(data=dop, x="E_ads_ev", y="pred_E_ads", hue="Mol", palette=ADSORBATE_COLORS, s=42, edgecolor="white", linewidth=0.3, ax=ax0)
    ax0.plot([lo, hi], [lo, hi], ls="--", color=PALETTE["red"], lw=1.2)
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    ax0.yaxis.set_major_locator(MaxNLocator(5))
    ax0.set_title("")
    ax0.set_xlabel(r"DFT E$_{ads}$ (eV)")
    ax0.set_ylabel(r"Predicted E$_{ads}$ (eV)")
    ax0.text(
        0.04,
        0.95,
        f"Direct transfer failed\nR$^2$ = {dop_metrics['R2']:.2f}\nRMSE = {dop_metrics['RMSE']:.2f}\nSpearman rho = {rho:.2f}",
        transform=ax0.transAxes,
        va="top",
        fontsize=9.7,
        linespacing=0.92,
        bbox=dict(fc="white", ec=PALETTE["light"], boxstyle="round,pad=0.28", alpha=0.92),
    )
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(
        handles,
        [ADSORBATE_LABELS.get(t, t) for t in labels],
        frameon=False,
        fontsize=8.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=4,
        title=None,
        handletextpad=0.28,
        columnspacing=0.72,
        borderpad=0.2,
    )
    style_axes(ax0)

    counts = dop["Dopant_clean"].value_counts()
    dop_order = counts[counts >= 3].index
    box_df = dop[dop["Dopant_clean"].isin(dop_order)].copy()
    order = box_df.groupby("Dopant_clean")["residual"].median().sort_values().index
    sns.boxplot(
        data=box_df,
        x="Dopant_clean",
        y="residual",
        order=order,
        palette=sns.color_palette("crest", n_colors=len(order)),
        fliersize=1.5,
        linewidth=0.8,
        ax=ax1,
    )
    ax1.axhline(0, color=PALETTE["muted"], lw=0.8, ls="--")
    ax1.set_title("")
    ax1.set_xlabel("Dopant")
    ax1.set_ylabel(r"Prediction residual of E$_{ads}$ (eV)")
    ax1.tick_params(axis="x", rotation=35)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    style_axes(ax1)

    rank = (
        dop.groupby("Dopant_clean")
        .agg(
            n=("E_ads_ev", "size"),
            mean_dft=("E_ads_ev", "mean"),
            mean_pred=("pred_E_ads", "mean"),
            transfer_mae=("residual", lambda s: np.mean(np.abs(s))),
            dft_score=("dft_window_score", "mean"),
            pred_score=("pred_screening_score", "mean"),
        )
        .query("Dopant_clean != 'None'")
        .sort_values("dft_score", ascending=False)
    )
    top_rank = rank[rank["n"] >= 3].head(12).iloc[::-1]
    ax2.barh(top_rank.index, top_rank["dft_score"], color=PALETTE["teal"])
    ax2.set_title("")
    ax2.set_xlabel("DFT-window score")
    ax2.set_xlim(0, 1.08)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    for i, (idx, row) in enumerate(top_rank.iterrows()):
        ax2.text(row["dft_score"] + 0.012, i, f"n={int(row['n'])}", va="center", fontsize=8, clip_on=False)
    style_axes(ax2)

    ax3.set_axis_off()
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(vmin=rank["dft_score"].min() if len(rank) else 0, vmax=rank["dft_score"].max() if len(rank) else 1)
    for element, row in rank.iterrows():
        if element not in PERIODIC_POS:
            continue
        period, group = PERIODIC_POS[element]
        x = group
        y = 7 - period
        color = cmap(norm(row["dft_score"]))
        rect = patches.Rectangle((x - 0.45, y - 0.40), 0.82, 0.72, facecolor=color, edgecolor="white", linewidth=1.0)
        ax3.add_patch(rect)
        text_color = "white" if norm(row["dft_score"]) > 0.72 else PALETTE["ink"]
        ax3.text(x - 0.04, y + 0.08, element, ha="center", va="center", fontsize=11, weight="bold", color=text_color)
        ax3.text(x - 0.04, y - 0.16, f"{row['dft_score']:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    ax3.set_xlim(2.3, 12.85)
    ax3.set_ylim(0.35, 3.72)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax3, fraction=0.025, pad=0.015)
    cbar.set_label("DFT-window score", fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.label.set_size(12)

    dop.to_csv(OUT / "Fig08" / "dopants_predictions.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT / "Fig08" / "dopant_ranking.csv", encoding="utf-8-sig")
    savefig(fig, 8, "dopant_validation_screening")


def write_manifest(context: dict, wgs: pd.DataFrame, dop: pd.DataFrame) -> None:
    manifest = {
        "data_path": str(DATA_PATH),
        "wgs_rows_total": int(pd.read_excel(DATA_PATH, sheet_name="WGS").shape[0]),
        "wgs_eads_valid": int(wgs["E_ads_ev"].notna().sum()),
        "wgs_activation_valid": int(wgs["E_activation_ev"].notna().sum()),
        "dopants_eads_valid": int(dop["E_ads_ev"].notna().sum()),
        "best_single_model_by_topsis": context["best_model"],
        "holdout_metrics": context["metrics"].round(5).to_dict(orient="index"),
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    wgs, dop = load_data()
    context = prepare_training(wgs, dop)
    figure_1(wgs, dop)
    figure_2(wgs)
    imp = figure_3(context, wgs)
    figure_4(context)
    figure_5(context)
    figure_6(context, imp)
    figure_7(context, wgs, imp)
    figure_8(context)
    write_manifest(context, wgs, dop)
    print("DONE")
    print(json.dumps({"best_model": context["best_model"], "metrics": context["metrics"].round(4).to_dict(orient="index")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
