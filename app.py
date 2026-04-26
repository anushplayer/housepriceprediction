"""Streamlit app for House Price Prediction."""

import os
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
DATA_PATH = "data/house_prices.csv"
NUMERIC_FEATURES = [
    "size_sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "garage_spaces",
    "floors",
    "lot_size_sqft",
    "has_pool",
]
CATEGORICAL_FEATURES = ["neighborhood"]
NEIGHBORHOODS = ["Downtown", "Suburbs", "Uptown", "Rural", "Waterfront"]


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models() -> dict[str, Pipeline]:
    paths = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    models = {}
    for p in sorted(paths):
        name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
        models[name] = joblib.load(p)
    return models


@st.cache_data(show_spinner="Loading dataset…")
def load_results() -> pd.DataFrame | None:
    path = os.path.join(MODELS_DIR, "results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Model")
        return df
    return None


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame | None:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def make_input_df(
    size_sqft, bedrooms, bathrooms, year_built,
    garage_spaces, floors, lot_size_sqft, has_pool, neighborhood
) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "garage_spaces": garage_spaces,
            "floors": floors,
            "lot_size_sqft": lot_size_sqft,
            "has_pool": float(has_pool),
            "neighborhood": neighborhood,
        }]
    )


def feature_importance_chart(model_name: str, pipeline: Pipeline) -> go.Figure | None:
    """Return a Plotly bar chart of feature importances (tree-based models only)."""
    regressor = pipeline.named_steps.get("model")
    if not hasattr(regressor, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    num_names = NUMERIC_FEATURES
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    feature_names = num_names + cat_names
    importances = regressor.feature_importances_

    df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(15)
    )
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=f"Feature Importances – {model_name}",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, yaxis_title=None, height=450)
    return fig


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 House Price Prediction")
st.markdown(
    "Predict house prices using multiple machine-learning models trained on "
    "synthetic real-estate data."
)

# ── sidebar – check models are trained ───────────────────────────────────────

if not os.path.isdir(MODELS_DIR) or not glob.glob(os.path.join(MODELS_DIR, "*.pkl")):
    st.error(
        "⚠️ No trained models found. "
        "Please run `python train_models.py` first to train and save the models."
    )
    st.stop()

models = load_models()
results_df = load_results()
data_df = load_data()

# ── tabs ─────────────────────────────────────────────────────────────────────

tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Comparison", "📁 Dataset", "ℹ️ About"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Predict
# ════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.subheader("Enter House Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        size_sqft = st.number_input("House Size (sq ft)", 300, 10_000, 1_800, step=50)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.select_slider(
            "Bathrooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], value=2.0
        )

    with col2:
        year_built = st.number_input("Year Built", 1900, 2024, 2000, step=1)
        garage_spaces = st.slider("Garage Spaces", 0, 5, 2)
        floors = st.slider("Floors", 1, 4, 1)

    with col3:
        lot_size_sqft = st.number_input("Lot Size (sq ft)", 1_000, 50_000, 6_000, step=500)
        has_pool = st.checkbox("Has Swimming Pool", value=False)
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=1)

    st.divider()
    model_choice = st.selectbox("Select Model", list(models.keys()))

    if st.button("💰 Predict Price", type="primary", use_container_width=True):
        input_df = make_input_df(
            size_sqft, bedrooms, bathrooms, year_built,
            garage_spaces, floors, lot_size_sqft, has_pool, neighborhood
        )
        prediction = models[model_choice].predict(input_df)[0]

        st.success(f"### Estimated Price: **${prediction:,.0f}**")

        # Show all model predictions
        with st.expander("📋 Predictions from all models"):
            all_preds = {
                name: f"${pipeline.predict(input_df)[0]:,.0f}"
                for name, pipeline in models.items()
            }
            pred_df = pd.DataFrame(
                list(all_preds.items()), columns=["Model", "Predicted Price"]
            )
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Feature importance for selected model
        fig = feature_importance_chart(model_choice, models[model_choice])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Comparison
# ════════════════════════════════════════════════════════════════════════════

with tab_compare:
    if results_df is None:
        st.warning("results.csv not found. Run `python train_models.py` to generate it.")
    else:
        st.subheader("Model Performance on Test Set")
        st.dataframe(
            results_df.style.format(
                {"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.4f}", "MAPE": "{:.2f}%"}
            ).background_gradient(subset=["R2"], cmap="Greens")
            .background_gradient(subset=["RMSE"], cmap="Reds_r"),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_r2 = px.bar(
                results_df.sort_values("R2"),
                x="R2", y=results_df.sort_values("R2").index,
                orientation="h", title="R² Score (higher is better)",
                color="R2", color_continuous_scale="Greens",
                text_auto=".4f",
            )
            fig_r2.update_layout(showlegend=False, yaxis_title=None, height=420)
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_b:
            fig_rmse = px.bar(
                results_df.sort_values("RMSE", ascending=False),
                x="RMSE", y=results_df.sort_values("RMSE", ascending=False).index,
                orientation="h", title="RMSE – Root Mean Squared Error (lower is better)",
                color="RMSE", color_continuous_scale="Reds_r",
                text_auto=",.0f",
            )
            fig_rmse.update_layout(showlegend=False, yaxis_title=None, height=420)
            st.plotly_chart(fig_rmse, use_container_width=True)

        # Radar chart
        st.subheader("Radar Comparison (normalised metrics)")
        metrics_norm = results_df[["R2", "MAE", "RMSE", "MAPE"]].copy()
        for col in ["MAE", "RMSE", "MAPE"]:
            metrics_norm[col] = 1 - (metrics_norm[col] - metrics_norm[col].min()) / (
                metrics_norm[col].max() - metrics_norm[col].min() + 1e-9
            )
        categories = list(metrics_norm.columns)
        fig_radar = go.Figure()
        for model_name, row in metrics_norm.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=row.tolist() + [row.tolist()[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=model_name,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalised Model Performance (higher = better for all axes)",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Dataset
# ════════════════════════════════════════════════════════════════════════════

with tab_data:
    if data_df is None:
        st.warning("Dataset not found. Run `python generate_data.py` or `python train_models.py`.")
    else:
        st.subheader(f"Dataset Preview  ({len(data_df):,} rows)")
        st.dataframe(data_df.head(100), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(data_df.describe().round(2), use_container_width=True)

        st.subheader("Feature Distributions")
        num_cols = ["size_sqft", "bedrooms", "bathrooms", "year_built", "garage_spaces",
                    "lot_size_sqft", "price"]
        selected_col = st.selectbox("Select feature", num_cols, index=0)
        fig_hist = px.histogram(
            data_df, x=selected_col, nbins=50,
            title=f"Distribution of {selected_col}",
            color_discrete_sequence=["steelblue"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Price by Neighborhood")
        fig_box = px.box(
            data_df, x="neighborhood", y="price",
            color="neighborhood",
            title="House Price Distribution by Neighborhood",
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = data_df[num_cols].corr().round(3)
        fig_heat = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – About
# ════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.subheader("About This App")
    st.markdown(
        """
        ### 🏠 House Price Prediction

        This application demonstrates multiple machine-learning regression models
        trained to predict house prices based on key property features.

        #### Features used
        | Feature | Description |
        |---|---|
        | `size_sqft` | Total living area in square feet |
        | `bedrooms` | Number of bedrooms |
        | `bathrooms` | Number of bathrooms (0.5 increments) |
        | `year_built` | Year the house was constructed |
        | `garage_spaces` | Number of garage parking spaces |
        | `floors` | Number of floors |
        | `lot_size_sqft` | Total lot size in square feet |
        | `has_pool` | Whether the property has a swimming pool |
        | `neighborhood` | Neighborhood category |

        #### Models trained
        - **Linear Regression** – Ordinary least squares baseline
        - **Ridge** – L2-regularised linear regression
        - **Lasso** – L1-regularised linear regression (feature selection)
        - **ElasticNet** – Combined L1 + L2 regularisation
        - **Decision Tree** – Non-linear single-tree regressor
        - **Random Forest** – Ensemble of decision trees (bagging)
        - **Extra Trees** – Extremely randomised tree ensemble
        - **Gradient Boosting** – Sequential boosting ensemble (sklearn)
        - **XGBoost** – Extreme Gradient Boosting
        - **KNN** – K-Nearest Neighbours regressor
        - **SVR** – Support Vector Regressor (RBF kernel)

        #### How to run
        ```bash
        pip install -r requirements.txt
        python train_models.py          # train & save all models
        streamlit run app.py            # launch the web app
        ```
        """
    )
