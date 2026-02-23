"""Plotly Dash dashboard for Customer Churn Probability Forecasting.

Sections:
  1. KPI cards        – ROC-AUC, PR-AUC, Brier Score, ECE
  2. Probability distribution histogram
  3. Risk segmentation bar chart
  4. ROC curve (ML vs naïve baseline)
  5. Calibration plot
  6. Feature importance bar chart
  7. Guardrail status table

Run locally:
    uv run python app.py

Docker / HuggingFace Spaces:
    The app listens on 0.0.0.0:7860 (env PORT overrides the port).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
PREDICTIONS_PATH = ROOT_DIR / "predictions" / "churn_predictions.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_json(name: str) -> dict | list | None:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH)


def load_all() -> dict:
    return {
        "predictions": _load_predictions(),
        "metrics": _load_json("eval_metrics.json"),
        "calibration": _load_json("calibration.json"),
        "feature_importance": _load_json("feature_importance.json"),
        "roc_curve": _load_json("roc_curve.json"),
        "guardrails": _load_json("guardrails.json"),
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

_PALETTE = {
    "high": "#e74c3c",
    "medium": "#f39c12",
    "low": "#2ecc71",
    "primary": "#2c3e50",
    "grid": "#ecf0f1",
}


def _no_data_fig(title: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="Run <b>make train</b> and <b>make predict</b> first",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#7f8c8d"),
    )
    fig.update_layout(title=title, paper_bgcolor="white", plot_bgcolor="white")
    return fig


def fig_prob_distribution(predictions: pd.DataFrame | None) -> go.Figure:
    if predictions is None:
        return _no_data_fig("Churn Probability Distribution")
    fig = go.Figure(
        go.Histogram(
            x=predictions["churn_probability"],
            nbinsx=40,
            marker_color=_PALETTE["primary"],
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="Churn Probability Distribution",
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        bargap=0.05,
        paper_bgcolor="white",
        plot_bgcolor=_PALETTE["grid"],
    )
    return fig


def fig_risk_segmentation(predictions: pd.DataFrame | None) -> go.Figure:
    if predictions is None:
        return _no_data_fig("Risk Segmentation")
    counts = (
        predictions["risk_bucket"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    )
    colors = [_PALETTE["high"], _PALETTE["medium"], _PALETTE["low"]]
    fig = go.Figure(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=colors,
            text=counts.values.tolist(),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Customers by Risk Bucket",
        xaxis_title="Risk Bucket",
        yaxis_title="Number of Customers",
        paper_bgcolor="white",
        plot_bgcolor=_PALETTE["grid"],
    )
    return fig


def fig_roc_curve(roc_data: dict | None, metrics: dict | None) -> go.Figure:
    fig = go.Figure()
    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Random",
        )
    )
    if roc_data is None:
        fig.add_annotation(
            text="Run <b>make train</b> first",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="#7f8c8d"),
        )
    else:
        auc_label = f"XGBoost (AUC={metrics['roc_auc']:.3f})" if metrics else "XGBoost"
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                line=dict(color=_PALETTE["primary"], width=2),
                name=auc_label,
            )
        )
        if metrics and "baseline_roc_auc" in metrics and metrics["baseline_roc_auc"]:
            fig.add_annotation(
                text=f"Baseline AUC≈{metrics['baseline_roc_auc']:.3f}",
                xref="paper",
                yref="paper",
                x=0.60,
                y=0.10,
                showarrow=False,
                font=dict(size=11),
            )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        paper_bgcolor="white",
        plot_bgcolor=_PALETTE["grid"],
        legend=dict(x=0.60, y=0.05),
    )
    return fig


def fig_calibration(calibration: dict | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect calibration",
        )
    )
    if calibration is None:
        fig.add_annotation(
            text="Run <b>make train</b> first",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="#7f8c8d"),
        )
    else:
        ece_label = f"Model (ECE={calibration['ece']:.3f})"
        fig.add_trace(
            go.Scatter(
                x=calibration["mean_predicted_value"],
                y=calibration["fraction_of_positives"],
                mode="lines+markers",
                line=dict(color=_PALETTE["primary"], width=2),
                marker=dict(size=7),
                name=ece_label,
            )
        )
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="white",
        plot_bgcolor=_PALETTE["grid"],
    )
    return fig


def fig_feature_importance(fi: dict | None) -> go.Figure:
    if fi is None:
        return _no_data_fig("Feature Importance")
    # Top 15
    items = list(fi.items())[:15]
    features = [i[0] for i in items][::-1]
    scores = [i[1] for i in items][::-1]
    fig = go.Figure(
        go.Bar(
            y=features,
            x=scores,
            orientation="h",
            marker_color=_PALETTE["primary"],
        )
    )
    fig.update_layout(
        title="Top 15 Feature Importances (XGBoost gain)",
        xaxis_title="Importance",
        paper_bgcolor="white",
        plot_bgcolor=_PALETTE["grid"],
        height=420,
    )
    return fig


def _kpi_card(label: str, value: str, color: str = "#2c3e50") -> html.Div:
    return html.Div(
        [
            html.P(label, style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d"}),
            html.H3(value, style={"margin": "4px 0 0", "color": color, "fontSize": "24px"}),
        ],
        style={
            "background": "white",
            "borderRadius": "8px",
            "padding": "16px 24px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
            "minWidth": "160px",
            "textAlign": "center",
        },
    )


def _guardrail_table(guardrails: list[dict] | None) -> html.Div:
    if guardrails is None:
        return html.P("Run make train to see guardrail results.", style={"color": "#7f8c8d"})

    rows = []
    for g in guardrails:
        rows.append(
            {
                "Check": g["check"],
                "Value": f"{g['value']:.4f}",
                "Threshold": f"{g['threshold']:.4f}",
                "Status": "PASS" if g["passed"] else "FAIL",
            }
        )

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0]],
        style_data_conditional=[
            {
                "if": {"filter_query": '{Status} = "FAIL"', "column_id": "Status"},
                "backgroundColor": "#fdecea",
                "color": "#e74c3c",
                "fontWeight": "bold",
            },
            {
                "if": {"filter_query": '{Status} = "PASS"', "column_id": "Status"},
                "backgroundColor": "#eafaf1",
                "color": "#2ecc71",
                "fontWeight": "bold",
            },
        ],
        style_header={
            "backgroundColor": _PALETTE["primary"],
            "color": "white",
            "fontWeight": "bold",
        },
        style_cell={"textAlign": "center", "padding": "8px"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_layout(data: dict) -> html.Div:
    metrics = data.get("metrics")
    predictions = data.get("predictions")

    # KPI values
    def _fmt(val: float | None, fmt: str = ".4f") -> str:
        return f"{val:{fmt}}" if val is not None else "—"

    roc = _fmt(metrics.get("roc_auc") if metrics else None)
    pr = _fmt(metrics.get("pr_auc") if metrics else None)
    brier = _fmt(metrics.get("brier_score") if metrics else None)
    ece = _fmt(data["calibration"]["ece"] if data.get("calibration") else None)
    n_cust = f"{len(predictions):,}" if predictions is not None else "—"

    kpis = html.Div(
        [
            _kpi_card("Customers Scored", n_cust),
            _kpi_card("ROC-AUC", roc, "#2980b9"),
            _kpi_card("PR-AUC", pr, "#8e44ad"),
            _kpi_card("Brier Score", brier, "#e67e22"),
            _kpi_card("Calib. ECE", ece, "#27ae60"),
        ],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
    )

    layout = html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1(
                        "Customer Churn Probability Dashboard",
                        style={"color": "white", "margin": "0", "fontSize": "22px"},
                    ),
                    html.P(
                        "Production-ready churn forecasting system",
                        style={"color": "#bdc3c7", "margin": "4px 0 0", "fontSize": "13px"},
                    ),
                ],
                style={
                    "background": _PALETTE["primary"],
                    "padding": "20px 32px",
                    "marginBottom": "24px",
                },
            ),
            # Body
            html.Div(
                [
                    kpis,
                    # Row 1: distribution + risk buckets
                    html.Div(
                        [
                            dcc.Graph(
                                figure=fig_prob_distribution(predictions), style={"flex": "1"}
                            ),
                            dcc.Graph(
                                figure=fig_risk_segmentation(predictions), style={"flex": "1"}
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    # Row 2: ROC + Calibration
                    html.Div(
                        [
                            dcc.Graph(
                                figure=fig_roc_curve(data.get("roc_curve"), metrics),
                                style={"flex": "1"},
                            ),
                            dcc.Graph(
                                figure=fig_calibration(data.get("calibration")), style={"flex": "1"}
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    # Feature importance (full width)
                    dcc.Graph(
                        figure=fig_feature_importance(data.get("feature_importance")),
                        style={"marginBottom": "16px"},
                    ),
                    # Guardrails table
                    html.Div(
                        [
                            html.H3(
                                "Guardrail Checks",
                                style={"marginBottom": "12px", "color": _PALETTE["primary"]},
                            ),
                            _guardrail_table(data.get("guardrails")),
                        ],
                        style={
                            "background": "white",
                            "borderRadius": "8px",
                            "padding": "20px",
                            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                            "marginBottom": "24px",
                        },
                    ),
                ],
                style={"padding": "0 32px 32px"},
            ),
        ],
        style={
            "fontFamily": "Inter, Arial, sans-serif",
            "background": "#f5f6fa",
            "minHeight": "100vh",
        },
    )
    return layout


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Churn Dashboard")
app.layout = build_layout(load_all())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
