"""
Predictive Maintenance: Turbofan Engine Health Monitor
Interactive Streamlit app built on top of the CMAPSS_RUL_Prediction.ipynb pipeline.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Turbofan Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
)

DATA_PATH = "train_FD001.txt"

LOW_VARIANCE_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                         "sensor_16", "sensor_18", "sensor_19"]

# RUL thresholds for health status — tune these for your narrative
CRITICAL_THRESHOLD = 20   # cycles remaining
WARNING_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Data + model pipeline — mirrors CMAPSS_RUL_Prediction.ipynb exactly
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    columns = ["engine_id", "cycle", "op_setting1", "op_setting2", "op_setting3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = columns

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.rename(columns={"cycle": "max_cycle"}, inplace=True)
    df = df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    return df


@st.cache_resource
def train_models(df: pd.DataFrame, _seed: int = 42):
    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
    selected_sensors = [s for s in sensor_columns if s not in LOW_VARIANCE_SENSORS]
    features = ["cycle", "op_setting1", "op_setting2", "op_setting3"] + selected_sensors
    target = "RUL"

    engine_ids = df["engine_id"].unique()
    train_ids, test_ids = train_test_split(engine_ids, test_size=0.2, random_state=_seed)

    X_train = df[df["engine_id"].isin(train_ids)][features]
    y_train = df[df["engine_id"].isin(train_ids)][target]
    X_test = df[df["engine_id"].isin(test_ids)][features]
    y_test = df[df["engine_id"].isin(test_ids)][target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=_seed, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    metrics = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_rf))],
        "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf)],
        "R2": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf)],
    }).round(3)

    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

    return {
        "features": features,
        "scaler": scaler,
        "rf_model": rf_model,
        "lr_model": lr_model,
        "metrics": metrics,
        "importances": importances,
        "y_test": y_test.reset_index(drop=True),
        "y_pred_rf": y_pred_rf,
        "test_ids": test_ids,
    }


def health_status(predicted_rul: float):
    if predicted_rul <= CRITICAL_THRESHOLD:
        return "🔴 Critical", "red"
    elif predicted_rul <= WARNING_THRESHOLD:
        return "🟡 Warning", "orange"
    else:
        return "🟢 Healthy", "green"


# ---------------------------------------------------------------------------
# Load + train
# ---------------------------------------------------------------------------

with st.spinner("Loading data and training models..."):
    df = load_data(DATA_PATH)
    artifacts = train_models(df)

st.title("✈️ Turbofan Engine Health Monitor")
st.caption(
    "Predicting Remaining Useful Life (RUL) on NASA's C-MAPSS FD001 dataset — "
    "Random Forest vs. Linear Regression. Built on top of the original notebook analysis."
)

tab_fleet, tab_engine, tab_model = st.tabs(["Fleet Overview", "Engine Inspector", "Model Performance"])

# ---------------------------------------------------------------------------
# Tab 1 — Fleet overview
# ---------------------------------------------------------------------------

with tab_fleet:
    st.subheader("Fleet health snapshot")
    st.caption(
        "Simulated 'current' snapshot at 80% through each engine's recorded life, scored with the "
        "trained Random Forest model. (Using each engine's very last recorded cycle instead would show "
        "100% critical — the training data runs every engine to failure, so its last cycle is always the "
        "failure point by definition.)"
    )

    snapshot_pct = st.slider(
        "Simulate fleet at what % of each engine's recorded life?",
        min_value=10, max_value=100, value=80, step=5,
        help="100% = each engine's actual failure point (all will show Critical, since RUL=0 by definition there).",
    )

    def pick_snapshot_row(engine_df: pd.DataFrame) -> pd.Series:
        target_cycle = max(1, int(np.ceil(engine_df["cycle"].max() * snapshot_pct / 100)))
        idx = (engine_df["cycle"] - target_cycle).abs().idxmin()
        return engine_df.loc[idx]

    latest = df.groupby("engine_id", group_keys=False).apply(pick_snapshot_row).copy()
    X_latest = latest[artifacts["features"]]
    X_latest_scaled = artifacts["scaler"].transform(X_latest)
    latest["predicted_RUL"] = artifacts["rf_model"].predict(X_latest_scaled)
    latest["status"] = latest["predicted_RUL"].apply(lambda r: health_status(r)[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total engines", latest["engine_id"].nunique())
    c2.metric("🔴 Critical", (latest["predicted_RUL"] <= CRITICAL_THRESHOLD).sum())
    c3.metric("🟡 Warning", ((latest["predicted_RUL"] > CRITICAL_THRESHOLD) & (latest["predicted_RUL"] <= WARNING_THRESHOLD)).sum())
    c4.metric("🟢 Healthy", (latest["predicted_RUL"] > WARNING_THRESHOLD).sum())

    fig = px.bar(
        latest.sort_values("predicted_RUL"),
        x="engine_id", y="predicted_RUL", color="status",
        color_discrete_map={"🔴 Critical": "#d62728", "🟡 Warning": "#ff9800", "🟢 Healthy": "#2ca02c"},
        title="Predicted Remaining Useful Life by engine (most recent cycle)",
        labels={"predicted_RUL": "Predicted RUL (cycles)", "engine_id": "Engine ID"},
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("At-risk engines")
    at_risk = latest[latest["predicted_RUL"] <= WARNING_THRESHOLD].sort_values("predicted_RUL")
    if not at_risk.empty:
        st.dataframe(
            at_risk[["engine_id", "cycle", "predicted_RUL", "status"]].rename(
                columns={"cycle": "current_cycle"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No engines currently in warning or critical range.")

# ---------------------------------------------------------------------------
# Tab 2 — Engine inspector
# ---------------------------------------------------------------------------

with tab_engine:
    st.subheader("Inspect a single engine over its lifetime")

    engine_ids_sorted = sorted(df["engine_id"].unique())
    selected_engine = st.selectbox("Select engine ID", engine_ids_sorted)

    engine_df = df[df["engine_id"] == selected_engine].sort_values("cycle").reset_index(drop=True)
    max_cycle_val = int(engine_df["cycle"].max())

    selected_cycle = st.slider("Simulate current operating cycle", 1, max_cycle_val, max_cycle_val)
    snapshot = engine_df[engine_df["cycle"] == selected_cycle].iloc[[0]]

    X_snap = snapshot[artifacts["features"]]
    X_snap_scaled = artifacts["scaler"].transform(X_snap)
    predicted_rul = float(artifacts["rf_model"].predict(X_snap_scaled)[0])
    actual_rul = float(snapshot["RUL"].iloc[0])
    status_label, status_color = health_status(predicted_rul)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL", f"{predicted_rul:.0f} cycles")
    col2.metric("Actual RUL (ground truth)", f"{actual_rul:.0f} cycles")
    col3.markdown(f"### Status: {status_label}")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        title={"text": f"Engine {selected_engine} — Predicted RUL"},
        gauge={
            "axis": {"range": [0, max(150, predicted_rul + 20)]},
            "bar": {"color": status_color},
            "steps": [
                {"range": [0, CRITICAL_THRESHOLD], "color": "#f8d7da"},
                {"range": [CRITICAL_THRESHOLD, WARNING_THRESHOLD], "color": "#fff3cd"},
                {"range": [WARNING_THRESHOLD, max(150, predicted_rul + 20)], "color": "#d4edda"},
            ],
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("Sensor trends up to the selected cycle")
    default_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]
    sensor_options = [c for c in engine_df.columns if c.startswith("sensor_")]
    chosen_sensors = st.multiselect("Sensors to plot", sensor_options, default=default_sensors)

    if chosen_sensors:
        plot_df = engine_df[engine_df["cycle"] <= selected_cycle]
        fig_ts = go.Figure()
        for s in chosen_sensors:
            fig_ts.add_trace(go.Scatter(x=plot_df["cycle"], y=plot_df[s], mode="lines", name=s))
        fig_ts.add_vline(x=selected_cycle, line_dash="dash", line_color="gray")
        fig_ts.update_layout(title=f"Engine {selected_engine} sensor readings", xaxis_title="Cycle", yaxis_title="Sensor value")
        st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Predicted vs. actual RUL across this engine's full life")
    engine_X = engine_df[artifacts["features"]]
    engine_X_scaled = artifacts["scaler"].transform(engine_X)
    engine_df["predicted_RUL_curve"] = artifacts["rf_model"].predict(engine_X_scaled)

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=engine_df["cycle"], y=engine_df["RUL"], mode="lines", name="Actual RUL"))
    fig_curve.add_trace(go.Scatter(x=engine_df["cycle"], y=engine_df["predicted_RUL_curve"], mode="lines", name="Predicted RUL"))
    fig_curve.add_vline(x=selected_cycle, line_dash="dash", line_color="gray")
    fig_curve.update_layout(title="Actual vs. predicted RUL over engine lifetime", xaxis_title="Cycle", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig_curve, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 — Model performance
# ---------------------------------------------------------------------------

with tab_model:
    st.subheader("Model comparison")
    st.dataframe(artifacts["metrics"], use_container_width=True, hide_index=True)

    fig_bar = px.bar(artifacts["metrics"], x="Model", y="RMSE", color="Model",
                      title="Model Comparison — RMSE (lower is better)",
                      color_discrete_sequence=["#8ca6b8", "#1a3c5e"])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Predicted vs. actual RUL (test set, Random Forest)")
    scatter_df = pd.DataFrame({"actual": artifacts["y_test"], "predicted": artifacts["y_pred_rf"]})
    fig_scatter = px.scatter(scatter_df, x="actual", y="predicted", opacity=0.3,
                              title="Random Forest: Predicted vs Actual RUL")
    fig_scatter.add_trace(go.Scatter(
        x=[scatter_df["actual"].min(), scatter_df["actual"].max()],
        y=[scatter_df["actual"].min(), scatter_df["actual"].max()],
        mode="lines", name="Perfect prediction", line=dict(dash="dash", color="red"),
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Top predictive features (Random Forest)")
    top_features = artifacts["importances"].head(10).reset_index()
    top_features.columns = ["feature", "importance"]
    fig_importance = px.bar(top_features.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title="Top 10 Most Important Features")
    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")
st.caption(
    "Data: NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001) · "
    "Model: Random Forest (n_estimators=100, max_depth=10) · "
    "github.com/Pramila6578/predictive-maintenance-rul-cmapss"
)
