import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load and preprocess the CSV
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Convert ts_event to datetime
    df["ts_event"] = pd.to_datetime(df["ts_event"])

    # Set index for easier grouping
    df.set_index("ts_event", inplace=True)
    return df


# Compute Best-Level OFI (OFI¹) for a single stock in a time window
def compute_best_level_ofi(df_stock, time_window="1min"):
    """
    Computes OFI¹ based on changes in best bid/ask prices and sizes.
    Formula from paper (Page 4): OFI = sum of contributions from bid/ask updates.
    """
    # Resample to time_window (e.g., 1-minute) and get LOB snapshots
    # (forward fill to handle missing data)
    df_resampled = (
        df_stock[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]]
        .resample(time_window)
        .last()
        .ffill()
    )

    # Initialize OFI
    ofi = []

    # Iterate through resampled data
    for i in range(1, len(df_resampled)):
        prev = df_resampled.iloc[i - 1]
        curr = df_resampled.iloc[i]

        # Bid side contribution
        bid_contrib = 0  # default (bid price decreases)
        if curr["bid_px_00"] > prev["bid_px_00"]:  # Bid price increases
            bid_contrib += curr["bid_sz_00"]
        elif curr["bid_px_00"] == prev["bid_px_00"]:  # Same price
            bid_contrib += curr["bid_sz_00"] - prev["bid_sz_00"]

        # Ask side contribution
        ask_contrib = 0  # default (ask price increases)
        if curr["ask_px_00"] < prev["ask_px_00"]:  # Ask price decreases
            ask_contrib += curr["ask_sz_00"]
        elif curr["ask_px_00"] == prev["ask_px_00"]:  # Same price
            ask_contrib += prev["ask_sz_00"] - curr["ask_sz_00"]

        # Total OFI for this step
        ofi_t = bid_contrib - ask_contrib
        ofi.append(ofi_t)

    # Create result DataFrame
    result = pd.DataFrame({"best_level_ofi": ofi}, index=df_resampled.index[1:])

    return result


# Compute Multi-Level OFI for a single stock
def compute_multi_level_ofi(df_stock, levels=10, time_window="1min"):
    """
    Computes OFI for each level (0 to levels-1) of the LOB.
    Returns a DataFrame with OFI for each level.
    """
    # Columns for each level
    bid_px_cols = [f"bid_px_{i:02d}" for i in range(levels)]
    ask_px_cols = [f"ask_px_{i:02d}" for i in range(levels)]
    bid_sz_cols = [f"bid_sz_{i:02d}" for i in range(levels)]
    ask_sz_cols = [f"ask_sz_{i:02d}" for i in range(levels)]

    # Resample to time_window
    cols = bid_px_cols + ask_px_cols + bid_sz_cols + ask_sz_cols
    df_resampled = df_stock[cols].resample(time_window).last().ffill()

    # Initialize Multi-Level OFI
    multi_ofi = {f"ofi_level_{i}": [] for i in range(levels)}
    index = df_resampled.index[1:]

    # Compute OFI for each level
    for i in range(1, len(df_resampled)):
        prev = df_resampled.iloc[i - 1]
        curr = df_resampled.iloc[i]

        for lvl in range(levels):
            bid_px_col = f"bid_px_{lvl:02d}"
            ask_px_col = f"ask_px_{lvl:02d}"
            bid_sz_col = f"bid_sz_{lvl:02d}"
            ask_sz_col = f"ask_sz_{lvl:02d}"

            # Bid contribution
            bid_contrib = 0
            if curr[bid_px_col] > prev[bid_px_col]:
                bid_contrib += curr[bid_sz_col]
            elif curr[bid_px_col] == prev[bid_px_col]:
                bid_contrib += curr[bid_sz_col] - prev[bid_sz_col]

            # Ask contribution
            ask_contrib = 0
            if curr[ask_px_col] < prev[ask_px_col]:
                ask_contrib += curr[ask_sz_col]
            elif curr[ask_px_col] == prev[ask_px_col]:
                ask_contrib += prev[ask_sz_col] - curr[ask_sz_col]

            # OFI for this level
            multi_ofi[f"ofi_level_{lvl}"].append(bid_contrib - ask_contrib)

    # Create result DataFrame
    result = pd.DataFrame(multi_ofi, index=index)
    return result


# Compute Integrated OFI using PCA
def compute_integrated_ofi(multi_level_ofi):
    """
    Applies PCA to Multi-Level OFI to get Integrated OFI (OFIᴵ).
    Returns zeros if input is empty, has one row, or all values are identical.
    """
    # Check if the multi-level OFI is:
    # empty, has only one unique value, or if all values are identical
    if (
        multi_level_ofi.empty
        or len(multi_level_ofi) <= 1
        or multi_level_ofi.nunique().eq(1).all()
    ):
        return pd.DataFrame(
            {"integrated_ofi": [0.0] * len(multi_level_ofi.index)},
            index=multi_level_ofi.index,
        )

    # Scale the data (mean=0, std=1) before PCA
    scaler = StandardScaler()
    scaled_ofi = scaler.fit_transform(multi_level_ofi)

    # Apply PCA to reduce to 1 component (Integrated OFI)
    # PCA will find the direction of maximum variance in the data
    pca = PCA(n_components=1)
    integrated_ofi = pca.fit_transform(scaled_ofi)

    return pd.DataFrame(
        {"integrated_ofi": integrated_ofi.flatten()}, index=multi_level_ofi.index
    )


# Compute all OFI features for all stocks
def compute_all_ofi_features(file_path, time_window="1min", levels=10):
    """
    Main function to compute all OFI features: Best-Level, Multi-Level, Integrated, Cross-Asset.
    """
    # Load data
    df = load_data(file_path)

    # Group by symbol (stock)
    stocks = df["symbol"].unique()
    all_features = []

    for stock in stocks:
        df_stock = df[df["symbol"] == stock]

        # Compute Best-Level OFI
        best_level_ofi = compute_best_level_ofi(df_stock, time_window)

        # Compute Multi-Level OFI
        multi_level_ofi = compute_multi_level_ofi(df_stock, levels, time_window)

        # Compute Integrated OFI
        integrated_ofi = compute_integrated_ofi(multi_level_ofi)

        # Combine features for this stock
        features = pd.concat([best_level_ofi, multi_level_ofi, integrated_ofi], axis=1)
        features["symbol"] = stock

        all_features.append(features)

    # Combine all stocks
    result = pd.concat(all_features)

    # Compute Cross-Asset OFI
    cross_asset_ofi = result.pivot(
        columns="symbol", values=["best_level_ofi", "integrated_ofi"]
    )
    cross_asset_ofi.columns = [f"{col[0]}_{col[1]}" for col in cross_asset_ofi.columns]

    # Merge with main result
    result = result.join(cross_asset_ofi, how="left")
    result.reset_index(inplace=True)

    return result


if __name__ == "__main__":
    # Ensure consistent file path naming
    file_path = "first_25000_rows.csv"

    # Compute all OFI features
    features_df = compute_all_ofi_features(file_path)

    # Preview the result
    print(features_df.head())

    # Save to CSV
    features_df.to_csv("ofi_features.csv", index=False)
