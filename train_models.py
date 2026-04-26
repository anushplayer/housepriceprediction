"""Train all house-price regression models and persist them to models/."""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from generate_data import generate_dataset

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
DATA_PATH = "data/house_prices.csv"
SEED = 42

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
TARGET = "price"

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=100.0, max_iter=10_000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=SEED),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=SEED
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=SEED
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    ),
    "KNN": KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    "SVR": SVR(kernel="rbf", C=100_000, epsilon=5_000),
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load or generate data
    if not os.path.exists(DATA_PATH):
        print("Generating dataset…")
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"Dataset: {len(df)} rows, columns: {list(df.columns)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor()
    results = {}

    for name, regressor in MODELS.items():
        print(f"  Training {name}…", end=" ", flush=True)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", regressor)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics

        # Persist
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
        print(f"R²={metrics['R2']:.4f}  RMSE=${metrics['RMSE']:,.0f}  MAE=${metrics['MAE']:,.0f}")

    # Save results summary
    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "Model"
    results_df.to_csv(os.path.join(MODELS_DIR, "results.csv"))
    print("\n=== Model Results Summary ===")
    print(results_df.sort_values("R2", ascending=False).to_string())
    print(f"\nModels saved to '{MODELS_DIR}/'")


if __name__ == "__main__":
    main()
