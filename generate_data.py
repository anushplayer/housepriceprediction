"""Generate a synthetic house price dataset and save it to data/house_prices.csv."""

import numpy as np
import pandas as pd
import os

SEED = 42
N_SAMPLES = 2000

NEIGHBORHOOD_MULTIPLIERS = {
    "Downtown": 1.40,
    "Suburbs": 1.00,
    "Uptown": 1.25,
    "Rural": 0.70,
    "Waterfront": 1.60,
}


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Core features
    size_sqft = rng.integers(600, 5001, size=n_samples).astype(float)
    bedrooms = rng.integers(1, 7, size=n_samples).astype(float)
    bathrooms = np.clip(
        rng.integers(1, 5, size=n_samples) + rng.choice([0, 0.5], size=n_samples),
        1,
        4.5,
    ).astype(float)
    year_built = rng.integers(1950, 2024, size=n_samples).astype(float)
    garage_spaces = rng.integers(0, 4, size=n_samples).astype(float)
    floors = rng.integers(1, 4, size=n_samples).astype(float)
    lot_size_sqft = rng.integers(2000, 20001, size=n_samples).astype(float)
    has_pool = rng.choice([0, 1], size=n_samples, p=[0.75, 0.25]).astype(float)
    neighborhoods = rng.choice(
        list(NEIGHBORHOOD_MULTIPLIERS.keys()), size=n_samples
    )

    # Price formula (with noise)
    base_price = (
        50_000
        + size_sqft * 120
        + bedrooms * 8_000
        + bathrooms * 6_000
        + (2023 - year_built) * (-400)
        + garage_spaces * 10_000
        + floors * 5_000
        + lot_size_sqft * 5
        + has_pool * 20_000
    )

    neighborhood_mult = np.array(
        [NEIGHBORHOOD_MULTIPLIERS[n] for n in neighborhoods]
    )
    noise = rng.normal(0, 25_000, size=n_samples)
    price = np.maximum(base_price * neighborhood_mult + noise, 50_000)

    df = pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "garage_spaces": garage_spaces,
            "floors": floors,
            "lot_size_sqft": lot_size_sqft,
            "has_pool": has_pool,
            "neighborhood": neighborhoods,
            "price": price.round(2),
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/house_prices.csv", index=False)
    print(f"Dataset saved to data/house_prices.csv  ({len(df)} rows)")
