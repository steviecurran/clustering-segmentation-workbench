"""
Generates a fully synthetic customer segmentation dataset, matching the
column structure (not the values) of the Kaggle "Customer Clustering"
dataset by dev0914sharma, so the app's bundled-example demo works without
redistributing that dataset's unlicensed file.

Column coding follows the same style as the original for consistency with
the app's existing labels:
    Sex: 0 = male, 1 = female
    Marital status: 0 = single, 1 = married/partnered
    Education: 0 = other/unknown, 1 = high school, 2 = university, 3 = graduate
    Occupation: 0 = unemployed/unskilled, 1 = skilled/official, 2 = management
    Settlement size: 0 = small city, 1 = mid-size city, 2 = big city

Safe for public use — entirely synthetic, not derived from any real
individual's data.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
N = 2000


def generate():
    ids = np.arange(100000001, 100000001 + N)

    sex = RNG.binomial(1, 0.5, size=N)
    age = np.clip(RNG.normal(35, 12, size=N).round().astype(int), 18, 75)

    # Education loosely increases with age (more time to attain it), then flattens
    education_probs_by_age = np.where(age < 25, 0, np.where(age < 40, 1, 2))
    education = np.clip(
        education_probs_by_age + RNG.integers(-1, 2, size=N), 0, 3
    )

    marital_status = RNG.binomial(1, np.clip(0.2 + 0.01 * (age - 18), 0, 0.85))

    # Occupation loosely tied to education
    occupation = np.clip(
        (education >= 2).astype(int) + RNG.binomial(1, 0.3, size=N), 0, 2
    )

    settlement_size = RNG.integers(0, 3, size=N)

    # Income driven by education, occupation, age (experience), settlement size,
    # plus noise -- gives the clustering task genuine, learnable structure.
    base_income = (
        30000
        + 15000 * education
        + 20000 * occupation
        + 800 * (age - 18)
        + 8000 * settlement_size
    )
    income = np.clip(
        base_income + RNG.normal(0, 15000, size=N), 20000, 320000
    ).round().astype(int)

    df = pd.DataFrame({
        "ID": ids,
        "Sex": sex,
        "Marital status": marital_status,
        "Age": age,
        "Education": education,
        "Income": income,
        "Occupation": occupation,
        "Settlement size": settlement_size,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("customer-segmentation-synthetic.csv", index=False)
    print(f"Generated {len(df):,} rows")
    print(df.head())
    print()
    print(df.describe())
