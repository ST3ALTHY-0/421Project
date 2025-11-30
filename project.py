import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, KBinsDiscretizer


CSV_PATH = r".\Tech_Use_Stress_Wellness.csv"
OUT_PATH = r".\processed2.csv"

NORMALIZE_COLS = [
    "daily_screen_time_hours",
    "phone_usage_hours",
    "laptop_usage_hours",
    "tablet_usage_hours",
    "tv_usage_hours",
    "social_media_hours",
    "work_related_hours",
    "entertainment_hours",
    "gaming_hours",
    "sleep_duration_hours",
    "sleep_quality",
    "mood_rating",
    "stress_level",
    "physical_activity_hours_per_week",
    "mental_health_score",
    "caffeine_intake_mg_per_day",
    "weekly_anxiety_score",
    "mindfulness_minutes_per_day",
]

ENCODE_COLS = [
    "gender",
    "location_type",
    "uses_wellness_apps",
    "eats_healthy",
    "weekly_depression_bin",  # our binned column we will also want encoded
    "weekly_stress_bin",
]


def bin_weekly_depression_score(df):

    #can edit bin values to see what works best
    bins = [-np.inf, 4, 9, np.inf]
    labels = ["Low", "Moderate", "High"]

    df["weekly_depression_bin"] = pd.cut(
        df["weekly_depression_score"],
        bins=bins,
        labels=labels
    )

    # also create numeric codes (0..n-1) so CSV contains both label and code
    try:
        df["weekly_depression_bin_code"] = pd.cut(
            df["weekly_depression_score"],
            bins=bins,
            labels=False,
            include_lowest=True,
        )
    except Exception:
        # fallback: map labels to codes if cut with labels=False fails
        mapping = {lab: i for i, lab in enumerate(labels)}
        df["weekly_depression_bin_code"] = df["weekly_depression_bin"].map(mapping)

    return df


def bin_weekly_stress_score(df):
    # find the stress-like column in this CSV (try several common names)
    candidates = ["weekly_stress_score", "stress_level", "stress_score"]
    col = next((c for c in candidates if c in df.columns), None)

    if col is None:
        # no stress column available in this CSV — do not crash, just return
        print("Warning: no stress column found (tried: weekly_stress_score, stress_level, stress_score). Skipping stress binning.")
        return df

    df = df.copy()
    # coerce to numeric (KBinsDiscretizer expects numeric values)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    # select non-null rows for fitting
    non_null_idx = df[col].dropna().index
    if len(non_null_idx) == 0:
        print("Warning: stress column exists but contains no numeric values. Skipping stress binning.")
        return df

    X = df.loc[non_null_idx, col].to_numpy().reshape(-1, 1)
    disc = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    codes = disc.fit_transform(X).astype(int).ravel()

    # place codes back into full-length series (leave NaN where original was NaN)
    full_codes = pd.Series(index=df.index, dtype=float)
    full_codes.loc[non_null_idx] = codes
    df["weekly_stress_bin_code"] = full_codes
    df["weekly_stress_bin"] = df["weekly_stress_bin_code"].map(lambda i: ("Low","Moderate","High")[int(i)] if pd.notna(i) else np.nan)
    return df


def normalize_columns(df, columns):
    df = df.copy()

    numeric_cols = [c for c in columns if c in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_numeric = [c for c in numeric_cols if not df[c].isna().all()]

    if not valid_numeric:
        return df

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[valid_numeric])

    normalized_df = pd.DataFrame(
        scaled,
        index=df.index,
        columns=[f"{c} (Normalized)" for c in valid_numeric]
    )

    return pd.concat([df, normalized_df], axis=1)


def encode_columns(df, columns, method="auto"):
    # For simplicity (single CSV), always label-encode the categorical columns
    for col in columns:
        df[f"{col} (Encoded)"] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def process(df):
    # create binned columns first so they can be encoded later
    df = bin_weekly_depression_score(df)
    df = bin_weekly_stress_score(df)
    df = normalize_columns(df, NORMALIZE_COLS)
    df = encode_columns(df, ENCODE_COLS, method="auto")
    return df



def main():
    df = pd.read_csv(CSV_PATH)
    processed = process(df)
    processed.to_csv(OUT_PATH, index=False)
    print(f"Saved processed CSV: {OUT_PATH}")

main()
