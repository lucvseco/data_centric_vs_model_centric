"""
viz.py — Visualization layer for BRACIS2 forecasting experiments.

Usage:
    import viz
    viz.set_figure_dir("outputs/figures")
    viz.plot_series(series, title="My Series")
    viz.plot_forecast_comparison(test, preds_dict, train)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

FIGURE_DIR: Optional[str] = None

APPROACH_COLORS = {
    "Data-Centric": "#2196F3",   # blue
    "Model-Centric": "#FF9800",  # orange
    "Baseline":      "#4CAF50",  # green
    "Reference":     "#9E9E9E",  # grey
}

MODEL_APPROACH_MAP = {
    "DC_SARIMA":   "Data-Centric",
    "DC_XGB":      "Data-Centric",
    "DC_LSTM":     "Data-Centric",
    "SARIMA-MC":   "Model-Centric",
    "XGBoost-MC":  "Model-Centric",
    "LSTM-MC":     "Model-Centric",
    "SARIMA_MC":   "Model-Centric",
    "XGB_MC":      "Model-Centric",
    "LSTM_MC":     "Model-Centric",
    "Prophet":     "Baseline",
    "Theta":       "Baseline",
    "ETS":         "Baseline",
    "SNaive":      "Reference",
}

# Line styles per model (within approach, for disambiguation)
_MODEL_LINESTYLES = {
    "DC_SARIMA":   "-",
    "DC_XGB":      "--",
    "DC_LSTM":     "-.",
    "SARIMA-MC":   "-",
    "XGBoost-MC":  "--",
    "LSTM-MC":     "-.",
    "SARIMA_MC":   "-",
    "XGB_MC":      "--",
    "LSTM_MC":     "-.",
    "Prophet":     "-",
    "Theta":       "--",
    "ETS":         "-.",
    "SNaive":      ":",
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def set_figure_dir(path: str) -> None:
    """Set the directory where figures are saved. Creates it if needed."""
    global FIGURE_DIR
    FIGURE_DIR = path
    os.makedirs(path, exist_ok=True)


def set_paper_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "lines.linewidth": 1.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _save(fig: plt.Figure, fname: str) -> str | None:
    """Save figure to FIGURE_DIR/fname if FIGURE_DIR is set. Returns saved path or None."""
    if FIGURE_DIR is None:
        return None
    path = os.path.join(FIGURE_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    return path


def _model_color(model: str, approach_map: dict | None = None) -> str:
    amap = approach_map or MODEL_APPROACH_MAP
    approach = amap.get(model, "Reference")
    return APPROACH_COLORS.get(approach, "#607D8B")


def _approach_legend_handles(models_used: list[str], approach_map: dict | None = None) -> list:
    amap = approach_map or MODEL_APPROACH_MAP
    seen = {}
    for m in models_used:
        ap = amap.get(m, "Reference")
        if ap not in seen:
            seen[ap] = APPROACH_COLORS.get(ap, "#607D8B")
    return [mpatches.Patch(color=c, label=ap) for ap, c in seen.items()]


# ---------------------------------------------------------------------------
# 1. Series overview
# ---------------------------------------------------------------------------

def plot_series(
    series: pd.Series,
    title: str = "Original Series",
    ylabel: str = "Value",
    train: pd.Series | None = None,
    test: pd.Series | None = None,
    fname: str = "01_series.png",
) -> plt.Figure:
    """
    Plot the full time series, optionally highlighting train/test split.

    Parameters
    ----------
    series : pd.Series with DatetimeIndex
    train, test : optional splits (colored differently if supplied)
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if train is not None and test is not None:
        ax.plot(train.index, train.values, color="#1565C0", label="Train", linewidth=1.8)
        ax.plot(test.index, test.values, color="#E53935", label="Test", linewidth=1.8)
        ax.axvline(test.index[0], color="black", linestyle="--", alpha=0.5, linewidth=1)
    else:
        ax.plot(series.index, series.values, color="#1565C0", linewidth=1.8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    if train is not None:
        ax.legend()
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 2. STL decomposition
# ---------------------------------------------------------------------------

def plot_stl_decomposition(
    series: pd.Series,
    period: int = 12,
    title: str = "STL Decomposition",
    fname: str = "02_stl_decomposition.png",
) -> plt.Figure:
    """
    Run STL decomposition (robust=True) and plot trend / seasonal / residual.
    Requires statsmodels.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        raise ImportError("statsmodels is required for STL decomposition.")

    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    components = [
        (series.values, "Observed", "#1565C0"),
        (res.trend, "Trend", "#FF6F00"),
        (res.seasonal, "Seasonal", "#2E7D32"),
        (res.resid, "Residual", "#6A1B9A"),
    ]
    for ax, (data, label, color) in zip(axes, components):
        ax.plot(series.index, data, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=10)
        if label == "Residual":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 3. Forecast comparison (all models vs actuals)
# ---------------------------------------------------------------------------

def plot_forecast_comparison(
    test: pd.Series,
    preds_dict: dict[str, np.ndarray],
    train: pd.Series | None = None,
    n_train_context: int = 24,
    title: str = "Forecast Comparison",
    ylabel: str = "Value",
    approach_map: dict | None = None,
    fname: str = "03_forecast_comparison.png",
) -> plt.Figure:
    """
    Plot test actuals against predictions from all models.

    Parameters
    ----------
    test : pd.Series (index = forecast dates)
    preds_dict : {model_label: np.ndarray of length len(test)}
    train : optional; last `n_train_context` points shown as context
    approach_map : override MODEL_APPROACH_MAP
    """
    amap = approach_map or MODEL_APPROACH_MAP
    fig, ax = plt.subplots(figsize=(13, 5))

    if train is not None and n_train_context > 0:
        ctx = train.iloc[-n_train_context:]
        ax.plot(ctx.index, ctx.values, color="#90A4AE", linewidth=1.5,
                label="Train (context)", zorder=1)

    ax.plot(test.index, test.values, color="black", linewidth=2.2,
            label="Actual", zorder=10)

    for model, preds in preds_dict.items():
        color = _model_color(model, amap)
        ls = _MODEL_LINESTYLES.get(model, "-")
        ax.plot(test.index, preds, color=color, linestyle=ls,
                linewidth=1.5, alpha=0.85, label=model, zorder=5)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 4. Metric bar chart (ranked models)
# ---------------------------------------------------------------------------

def plot_metric_bar(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
    exclude_reference: bool = False,
    approach_map: dict | None = None,
    fname: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart: models ranked by `metric` (ascending = better).

    Parameters
    ----------
    results_df : output of build_holdout_results(); must have `metric` column and
                 optionally `approach` column.
    """
    amap = approach_map or MODEL_APPROACH_MAP

    df = results_df.copy()
    if exclude_reference and "approach" in df.columns:
        df = df[df["approach"] != "Reference"]

    df = df[[metric]].dropna().sort_values(metric, ascending=True)

    colors = [_model_color(m, amap) for m in df.index]
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.45)))
    bars = ax.barh(df.index[::-1], df[metric].values[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Models ranked by {metric}")

    for bar, val in zip(bars, df[metric].values[::-1]):
        ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Approach legend
    handles = _approach_legend_handles(list(df.index), amap)
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, fname or f"04_bar_{metric.lower()}.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Horizon band metrics
# ---------------------------------------------------------------------------

def plot_horizon_bands(
    band_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
    approach_map: dict | None = None,
    fname: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: metric by horizon band for each model.

    Parameters
    ----------
    band_df : output of compute_horizon_band_metrics(); columns include
              {metric}_{band} e.g. RMSE_h01-h04, RMSE_h05-h08, RMSE_h09-h12
    """
    amap = approach_map or MODEL_APPROACH_MAP
    band_cols = [c for c in band_df.columns if c.startswith(f"{metric}_")]
    if not band_cols:
        raise ValueError(f"No columns matching '{metric}_*' in band_df. "
                         f"Available: {list(band_df.columns)}")

    bands = [c.split(f"{metric}_", 1)[1] for c in band_cols]
    models = band_df.index.tolist()
    x = np.arange(len(bands))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [band_df.loc[model, c] if c in band_df.columns else np.nan for c in band_cols]
        color = _model_color(model, amap)
        ls = _MODEL_LINESTYLES.get(model, "-")
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=color, alpha=0.85,
               label=model, hatch="" if ls == "-" else "//")

    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel(metric)
    ax.set_xlabel("Horizon band")
    ax.set_title(title or f"{metric} by Horizon Band")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, fname or f"05_horizon_bands_{metric.lower()}.png")
    return fig


# ---------------------------------------------------------------------------
# 6. CV fold results (line per model across folds)
# ---------------------------------------------------------------------------

def plot_cv_fold_results(
    cv_results_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
    approach_map: dict | None = None,
    fname: str | None = None,
) -> plt.Figure:
    """
    Line chart showing `metric` per fold for each model.

    Parameters
    ----------
    cv_results_df : output of run_temporal_cv() — rows are (model, fold) combinations;
                    must have columns: model, fold_idx, and `metric`.
    """
    amap = approach_map or MODEL_APPROACH_MAP
    if "fold" in cv_results_df.columns and "fold_idx" not in cv_results_df.columns:
        cv_results_df = cv_results_df.rename(columns={"fold": "fold_idx"})
    required = {"model", "fold_idx", metric}
    missing = required - set(cv_results_df.columns)
    if missing:
        raise ValueError(f"cv_results_df missing columns: {missing}")

    df = cv_results_df[cv_results_df[metric].notna()].copy()
    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        sub = df[df["model"] == model].sort_values("fold_idx")
        color = _model_color(model, amap)
        ls = _MODEL_LINESTYLES.get(model, "-")
        ax.plot(sub["fold_idx"], sub[metric], color=color, linestyle=ls,
                linewidth=1.6, marker="o", markersize=5, label=model)

    ax.set_xlabel("Fold index")
    ax.set_ylabel(metric)
    ax.set_title(title or f"CV {metric} per Fold")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save(fig, fname or f"06_cv_fold_{metric.lower()}.png")
    return fig


# ---------------------------------------------------------------------------
# 7. CV summary boxplot
# ---------------------------------------------------------------------------

def plot_cv_summary_boxplot(
    cv_results_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
    approach_map: dict | None = None,
    fname: str | None = None,
) -> plt.Figure:
    """
    Boxplot of `metric` distribution across folds, one box per model.

    Parameters
    ----------
    cv_results_df : same format as plot_cv_fold_results input.
    """
    amap = approach_map or MODEL_APPROACH_MAP
    df = cv_results_df[cv_results_df[metric].notna()].copy()
    models = sorted(df["model"].unique(),
                    key=lambda m: (amap.get(m, "z"), m))

    data = [df[df["model"] == m][metric].values for m in models]
    colors = [_model_color(m, amap) for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.1), 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"CV {metric} Distribution")

    handles = _approach_legend_handles(models, amap)
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    _save(fig, fname or f"07_cv_boxplot_{metric.lower()}.png")
    return fig


# ---------------------------------------------------------------------------
# 8. Approach comparison (mean metrics per approach)
# ---------------------------------------------------------------------------

def plot_approach_comparison(
    approach_summary: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Mean Metrics by Approach",
    fname: str = "08_approach_comparison.png",
) -> plt.Figure:
    """
    Grouped bar chart comparing approaches across multiple metrics.

    Parameters
    ----------
    approach_summary : output of summarize_by_approach(); index = approach names.
    metrics : list of metric columns to show (default: MAE, RMSE, sMAPE).
    """
    if metrics is None:
        metrics = [m for m in ["MAE", "RMSE", "sMAPE"] if m in approach_summary.columns]

    df = approach_summary[metrics].copy()
    approaches = df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, approach in enumerate(approaches):
        vals = df.loc[approach].values
        color = APPROACH_COLORS.get(approach, "#607D8B")
        offset = (i - len(approaches) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=color, alpha=0.85, label=approach)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 9. All forecasts heatmap / small multiples (optional helper)
# ---------------------------------------------------------------------------

def plot_all_forecasts(
    test: pd.Series,
    preds_dict: dict[str, np.ndarray],
    approach_map: dict | None = None,
    title: str = "Individual Model Forecasts",
    fname: str = "09_all_forecasts.png",
) -> plt.Figure:
    """
    Small-multiples grid: one subplot per model showing actual vs forecast.
    """
    amap = approach_map or MODEL_APPROACH_MAP
    models = list(preds_dict.keys())
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, model in zip(axes_flat, models):
        color = _model_color(model, amap)
        ax.plot(test.index, test.values, color="black", linewidth=1.8, label="Actual")
        ax.plot(test.index, preds_dict[model], color=color, linewidth=1.6,
                linestyle="--", label=model)
        ax.set_title(model, fontsize=10)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 10. Metrics heatmap across models × metrics
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(
    results_df: pd.DataFrame,
    metric_cols: list[str] | None = None,
    title: str = "Metrics Heatmap (lower = better)",
    fname: str = "10_metrics_heatmap.png",
) -> plt.Figure:
    """
    Colour-coded heatmap of normalised metric values.
    Each column (metric) is normalised to [0,1] so colours are comparable.

    Parameters
    ----------
    results_df : output of build_holdout_results().
    """
    if metric_cols is None:
        metric_cols = [c for c in ["MAE", "RMSE", "MAPE", "sMAPE", "U2", "MASE"]
                       if c in results_df.columns]

    df = results_df[metric_cols].copy().dropna(how="all")
    # Normalise per column (min-max, lower = better → lower = lighter)
    normed = df.apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-12), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(metric_cols) * 1.2), max(4, len(df) * 0.45)))
    im = ax.imshow(normed.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)

    # Annotate cells with original values
    for i in range(len(df)):
        for j, col in enumerate(metric_cols):
            val = df.iloc[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color="black")

    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised (0=best, 1=worst)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 11. Pareto: Accuracy vs Computational Cost
# ---------------------------------------------------------------------------

def plot_pareto_cost_accuracy(
    cost_benefit_df: pd.DataFrame,
    x_col: str = "t_total (s)",
    y_col: str = "RMSE",
    approach_col: str = "Abordagem",
    title: str | None = None,
    fname: str = "11_pareto_cost_accuracy.png",
) -> plt.Figure:
    """
    Scatter plot de acurácia (y_col) vs custo computacional (x_col).

    Cada ponto é um modelo; a linha tracejada conecta os modelos na frente
    de Pareto (menor custo *e* menor erro — dominância bi-critério).

    Parameters
    ----------
    cost_benefit_df : DataFrame produzido pela célula de instrumentação;
                      index = nome do modelo.
    x_col     : coluna de custo (default "t_total (s)").
    y_col     : coluna de acurácia, menor = melhor (default "RMSE").
    approach_col : coluna que identifica a abordagem (default "Abordagem").
    title     : título do gráfico (auto se None).
    fname     : nome do arquivo para salvar (dentro de FIGURE_DIR).
    """
    APPROACH_COLOR_MAP = {
        "Data-Centric":  APPROACH_COLORS.get("Data-Centric",  "#2196F3"),
        "Model-Centric": APPROACH_COLORS.get("Model-Centric", "#FF9800"),
    }
    APPROACH_MARKER_MAP = {
        "Data-Centric":  "o",
        "Model-Centric": "s",
    }

    model_col = cost_benefit_df.index.name or "Modelo"
    df = cost_benefit_df.reset_index().dropna(subset=[x_col, y_col])
    if df.empty:
        raise ValueError(
            "cost_benefit_df não contém linhas com valores válidos "
            f"em '{x_col}' e '{y_col}'."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    for approach, grp in df.groupby(approach_col):
        color  = APPROACH_COLOR_MAP.get(str(approach), "#607D8B")
        marker = APPROACH_MARKER_MAP.get(str(approach), "D")
        ax.scatter(
            grp[x_col], grp[y_col],
            c=color, marker=marker, s=110,
            label=approach, zorder=5,
            edgecolors="white", linewidths=1.2,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                str(row[model_col]),
                (row[x_col], row[y_col]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8.5, color=color,
            )

    # Frente de Pareto: varrer da esquerda para a direita e manter
    # apenas os pontos que melhoram y (menor RMSE).
    pareto_df = df[[x_col, y_col]].sort_values(x_col).reset_index(drop=True)
    best_y = float("inf")
    pareto_pts: list[tuple[float, float]] = []
    for _, r in pareto_df.iterrows():
        if r[y_col] < best_y:
            best_y = r[y_col]
            pareto_pts.append((float(r[x_col]), float(r[y_col])))

    if len(pareto_pts) > 1:
        px, py = zip(*pareto_pts)
        ax.step(
            px, py, where="post",
            color="#555555", linestyle="--", linewidth=1.2,
            label="Frente de Pareto", zorder=3,
        )

    _title = title or (
        f"Pareto — {y_col} × {x_col}\n"
        "(canto inferior-esquerdo = mais eficiente)"
    )
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(_title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 12. Stacked Bar: time decomposition (Feature Eng | Tuning | Training)
# ---------------------------------------------------------------------------

def plot_stacked_time_bar(
    cost_benefit_df: pd.DataFrame,
    time_cols: list[str] | None = None,
    segment_labels: list[str] | None = None,
    segment_colors: list[str] | None = None,
    title: str = "Decomposição do Tempo por Etapa",
    fname: str = "12_stacked_time_bar.png",
) -> plt.Figure:
    """
    Stacked horizontal bar chart decompondo o tempo total de cada modelo
    em etapas: Feature Engineering, Tuning e Treino (+ Forecast opcional).

    Parameters
    ----------
    cost_benefit_df : DataFrame com index = Modelo e colunas de tempo.
    time_cols   : colunas a empilhar (default: t_features, t_tuning, t_treino, t_forecast).
    segment_labels : rótulos exibidos na legenda (mesma ordem que time_cols).
    segment_colors : cores hex por segmento.
    """
    if time_cols is None:
        time_cols = ["t_features (s)", "t_tuning (s)", "t_treino (s)", "t_forecast (s)"]
        # Usar apenas colunas que existem no DataFrame
        time_cols = [c for c in time_cols if c in cost_benefit_df.columns]

    if segment_labels is None:
        _label_map = {
            "t_features (s)":  "Feature Eng.",
            "t_tuning (s)":    "Tuning",
            "t_treino (s)":    "Treino",
            "t_forecast (s)":  "Forecast",
        }
        segment_labels = [_label_map.get(c, c) for c in time_cols]

    if segment_colors is None:
        segment_colors = ["#42A5F5", "#FFA726", "#66BB6A", "#AB47BC"]
        segment_colors = segment_colors[:len(time_cols)]

    df = cost_benefit_df[time_cols].copy().fillna(0.0)
    models = df.index.tolist()
    approach_col = "Abordagem" if "Abordagem" in cost_benefit_df.columns else None

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.65)))

    bar_height = 0.55
    lefts = np.zeros(len(models))
    y_pos = np.arange(len(models))

    for col, label, color in zip(time_cols, segment_labels, segment_colors):
        vals = df[col].values
        ax.barh(y_pos, vals, left=lefts, height=bar_height,
                color=color, label=label, alpha=0.88)
        # Anotar segmentos maiores que 2% do max
        total_max = df.sum(axis=1).max()
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 0.02 * total_max:
                ax.text(l + v / 2, y_pos[i], f"{v:.1f}s",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
        lefts += vals

    # Anotar total à direita
    totals = df.sum(axis=1).values
    for i, tot in enumerate(totals):
        ax.text(tot * 1.01, y_pos[i], f"{tot:.1f}s",
                va="center", fontsize=8.5, color="#333333")

    # Colorir rótulos do eixo Y por abordagem
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    if approach_col:
        for ytick, model in zip(ax.get_yticklabels(), models):
            approach = cost_benefit_df.loc[model, approach_col]
            ytick.set_color(APPROACH_COLORS.get(str(approach), "#333333"))

    ax.set_xlabel("Tempo (segundos)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 13. Radar Chart: trade-off multidimensional (DC vs MC)
# ---------------------------------------------------------------------------

def plot_radar_tradeoff(
    cost_benefit_df: pd.DataFrame,
    axes_cfg: list[dict] | None = None,
    approach_col: str = "Abordagem",
    title: str = "Radar — Trade-off DC vs MC",
    fname: str = "13_radar_tradeoff.png",
) -> plt.Figure:
    """
    Radar (spider) chart com médias por abordagem em quatro dimensões:
    Acurácia, Eficiência de Tempo, #Features e #Parâmetros Tunados.

    Todos os eixos são normalizados para [0, 1] e orientados de forma que
    *mais externo = melhor* — exceto onde indicado por ``invert=True``.

    Parameters
    ----------
    cost_benefit_df : DataFrame com index = Modelo.
    axes_cfg : lista de dicts com chaves:
        col    – coluna do DataFrame
        label  – rótulo exibido no radar
        invert – se True, normaliza invertido (menor = melhor → maior no radar)
    approach_col : coluna que identifica a abordagem.
    """
    if axes_cfg is None:
        axes_cfg = [
            {"col": "RMSE",               "label": "Acurácia\n(1-norm RMSE)", "invert": True},
            {"col": "t_total (s)",         "label": "Efic. Tempo\n(1-norm t)",  "invert": True},
            {"col": "n_features",          "label": "#Features",               "invert": False},
            {"col": "n_params_tunados",    "label": "#Params\nTunados",         "invert": False},
        ]

    # Filtrar eixos cujas colunas existem
    axes_cfg = [a for a in axes_cfg if a["col"] in cost_benefit_df.columns]
    n_axes = len(axes_cfg)
    if n_axes < 3:
        raise ValueError("São necessários ao menos 3 eixos para o radar chart.")

    approaches = cost_benefit_df[approach_col].unique()
    approach_means: dict[str, list[float]] = {}
    raw_means: dict[str, list[float]] = {}

    for ap in approaches:
        sub = cost_benefit_df[cost_benefit_df[approach_col] == ap]
        approach_means[ap] = [sub[a["col"]].mean() for a in axes_cfg]

    # Normalização global (min/max sobre todos os modelos)
    all_vals = np.array([[cost_benefit_df[a["col"]].mean() for a in axes_cfg]
                         for _ in [None]])  # placeholder

    col_min = np.array([cost_benefit_df[a["col"]].min() for a in axes_cfg], dtype=float)
    col_max = np.array([cost_benefit_df[a["col"]].max() for a in axes_cfg], dtype=float)
    col_range = np.where(col_max - col_min > 1e-12, col_max - col_min, 1.0)

    norm_means: dict[str, np.ndarray] = {}
    for ap, vals in approach_means.items():
        v = (np.array(vals, dtype=float) - col_min) / col_range
        for i, a in enumerate(axes_cfg):
            if a.get("invert", False):
                v[i] = 1.0 - v[i]
        norm_means[ap] = v

    # Ângulos do radar (fechando o polígono)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a["label"] for a in axes_cfg], fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", alpha=0.4)

    plotted_approaches = []
    for ap, v in norm_means.items():
        values = v.tolist() + v[:1].tolist()
        color = APPROACH_COLORS.get(str(ap), "#607D8B")
        ax.plot(angles, values, color=color, linewidth=2.2, label=ap)
        ax.fill(angles, values, color=color, alpha=0.18)
        plotted_approaches.append(ap)

    ax.set_title(title, y=1.12, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# 14. Bubble Plot: Tuning Time × Accuracy Gain × #Parameters
# ---------------------------------------------------------------------------

def plot_bubble_tuning(
    cost_benefit_df: pd.DataFrame,
    x_col: str = "t_tuning (s)",
    y_col: str = "RMSE",
    size_col: str = "n_params_tunados",
    approach_col: str = "Abordagem",
    baseline_rmse: float | None = None,
    title: str = "Bubble — Tempo de Tuning × Ganho de Acurácia",
    fname: str = "14_bubble_tuning.png",
) -> plt.Figure:
    """
    Bubble plot onde:
        Eixo X  = tempo de tuning (s)
        Eixo Y  = ganho de acurácia em relação ao baseline DC médio
                  (positivo = melhor que o baseline)
        Tamanho = n_params_tunados (escala proporcional à área do círculo)

    Parameters
    ----------
    cost_benefit_df : DataFrame com index = Modelo.
    baseline_rmse   : RMSE de referência para calcular o ganho.
                      Se None, usa a média RMSE dos modelos Data-Centric.
    """
    df = cost_benefit_df.reset_index().copy()
    model_col = cost_benefit_df.index.name or "Modelo"

    # Baseline: média RMSE dos modelos DC
    if baseline_rmse is None:
        dc_mask = df[approach_col] == "Data-Centric"
        if dc_mask.any():
            baseline_rmse = df.loc[dc_mask, y_col].mean()
        else:
            baseline_rmse = df[y_col].mean()

    df["ganho_rmse"] = baseline_rmse - df[y_col]   # positivo = melhor que baseline

    # Escala de tamanho dos bubbles
    raw_sizes = df[size_col].fillna(1).values.astype(float)
    # Mínimo visível; área proporcional ao valor
    raw_sizes = np.where(raw_sizes <= 0, 0.5, raw_sizes)
    bubble_sizes = (raw_sizes / raw_sizes.max()) * 1200 + 150

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in df.iterrows():
        approach = str(row[approach_col])
        color = APPROACH_COLORS.get(approach, "#607D8B")
        marker = "o" if approach == "Data-Centric" else "s"
        idx = df.index.get_loc(_)
        ax.scatter(
            row[x_col], row["ganho_rmse"],
            s=bubble_sizes[idx],
            c=color, marker=marker,
            alpha=0.78, edgecolors="white", linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            str(row[model_col]),
            (row[x_col], row["ganho_rmse"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8.5, color=color,
        )

    # Linha de referência y=0 (baseline DC médio)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7,
               label=f"Baseline DC (RMSE={baseline_rmse:.4f})")

    # Legenda de abordagem
    handles = _approach_legend_handles(df[model_col].tolist())

    # Legenda de tamanho de bubble
    _size_vals = sorted(set(raw_sizes.astype(int)))
    _size_sample = [_size_vals[0], _size_vals[len(_size_vals) // 2], _size_vals[-1]]
    _size_sample_scaled = [(v / raw_sizes.max()) * 1200 + 150 for v in _size_sample]
    size_handles = [
        plt.scatter([], [], s=s, c="#888888", alpha=0.6, label=f"{int(v)} params")
        for v, s in zip(_size_sample, _size_sample_scaled)
    ]

    ax.legend(
        handles=handles + size_handles,
        loc="lower right", fontsize=8.5, framealpha=0.88,
    )

    ax.set_xlabel(f"Tempo de Tuning — {x_col}", fontsize=10)
    ax.set_ylabel("Ganho de Acurácia (RMSE baseline − RMSE modelo)\n"
                  "positivo = melhor que baseline DC", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Anotação do tamanho como lenda interna
    ax.text(
        0.02, 0.98,
        f"Tamanho ∝ {size_col}",
        transform=ax.transAxes, fontsize=8, va="top",
        color="#555555", style="italic",
    )

    fig.tight_layout()
    _save(fig, fname)
    return fig


# ---------------------------------------------------------------------------
# Apply paper style on import
# ---------------------------------------------------------------------------

set_paper_style()
