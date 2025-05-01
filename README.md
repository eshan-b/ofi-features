# Order Flow Imbalance (OFI) Feature Computation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This Python script computes Order Flow Imbalance (OFI) features from limit order book (LOB) data, as described in the paper ["Cross-impact of order flow imbalance in equity markets"](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2023.2236159) by Rama Cont, Mihai Cucuringu, and Chao Zhang (Quantitative Finance, 2023).

The script processes tick-by-tick LOB data to generate four types of OFI features: Best-Level OFI, Multi-Level OFI, Integrated OFI, and Cross-Asset OFI. These features are designed for analyzing price impact and cross-impact in equity markets, particularly for quantitative finance applications.

## Features Computed

The script generates the following OFI features for each stock and time window (default: 1-minute):

- **Best-Level OFI (OFI¹)**: Measures buy/sell imbalance at the best bid and ask prices (Level 0) of the LOB (Page 4 of the paper).

- **Multi-Level OFI**: Computes OFI for each of the top 10 LOB levels (Levels 0-9), capturing deeper market dynamics (Page 4).

- **Integrated OFI (OFIᴵ)**: Combines Multi-Level OFI using Principal Component Analysis (PCA) to produce a single, robust OFI metric (Page 4).

- **Cross-Asset OFI**: Includes OFI features from other stocks as predictors for cross-impact analysis (Page 6). For a single stock, these columns duplicate the stock’s own OFI.

## Input Data

The script expects a CSV file with tick-by-tick LOB data, similar to Nasdaq ITCH format. The dataset should include:

- **Columns**:

  - `ts_event`: Timestamp of the order book update (e.g., `2024-10-21T11:54:29.221064336Z`).

  - `symbol`: Stock ticker (e.g., `AAPL`).

  - `bid_px_00` to `bid_px_09`, `ask_px_00` to `ask_px_09`: Bid and ask prices for LOB levels 0-9.

  - `bid_sz_00` to `bid_sz_09`, `ask_sz_00` to `ask_sz_09`: Bid and ask sizes for LOB levels 0-9.

- **Current Dataset**: The provided dataset contains ~5,000 rows for a single stock (AAPL). The script is scalable to handle multiple stocks and larger datasets (e.g., 25,000+ rows).

## Prerequisites

- **Python Version**: Python 3.8+

- **Dependencies**: `pandas`, `numpy`, `scikit-learn`

## Installation

1. Clone or download this repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your LOB data CSV (e.g., `first_25000_rows.csv`) in the same directory as the script.

## Usage

1. Ensure your CSV file is named `first_25000_rows.csv` or update the `file_path` in the script.

2. Run the script:

   ```bash
   python compute_ofi_features.py
   ```

3. The script outputs a CSV file (`ofi_features.csv`) with the computed OFI features.

### Example Command

```bash
python compute_ofi_features.py
```

### Output

The output `ofi_features.csv` contains:

- **Columns**:

  - `ts_event`: Timestamp of the 1-minute window.

  - `symbol`: Stock ticker (e.g., `AAPL`).

  - `best_level_ofi`: Best-Level OFI for the stock.

  - `ofi_level_0` to `ofi_level_9`: Multi-Level OFI for each LOB level.

  - `integrated_ofi`: Integrated OFI from PCA.

  - `best_level_ofi_<TICKER>`, `integrated_ofi_<TICKER>`: Cross-Asset OFI for each stock (duplicates own OFI for single-stock data).

- **Example** (for AAPL-only):

  | ts_event                  | symbol | best_level_ofi | ofi_level_0 | ... | ofi_level_9 | integrated_ofi | best_level_ofi_AAPL | integrated_ofi_AAPL |
  | ------------------------- | ------ | -------------- | ----------- | --- | ----------- | -------------- | ------------------- | ------------------- |
  | 2024-10-21 11:55:00+00:00 | AAPL   | 0.0            | 0.0         | ... | 10.0        | 0.033090       | 0.0                 | 0.033090            |
  | 2024-10-21 11:56:00+00:00 | AAPL   | -173.0         | -173.0      | ... | 5.0         | 0.261467       | -173.0              | 0.261467            |

- **Rows**: Approximately one row per minute of trading data (e.g., ~300-400 rows for a trading day).

## Script Details

- **File**: `compute_ofi_features.py`

- **Functions**:

  - `load_data`: Loads and preprocesses the CSV, parsing timestamps.

  - `compute_best_level_ofi`: Computes Best-Level OFI using the paper’s formula (Page 4).

  - `compute_multi_level_ofi`: Computes Multi-Level OFI for 10 LOB levels.

  - `compute_integrated_ofi`: Applies PCA to Multi-Level OFI for Integrated OFI.

  - `compute_all_ofi_features`: Orchestrates feature computation for all stocks.

- **Parameters**:

  - `time_window`: Aggregation window (default: `1min`). Adjustable (e.g., `'30s'`).

  - `levels`: Number of LOB levels (default: 10, matching `bid_px_00` to `bid_px_09`).

- **Scalability**: Handles single-stock (e.g., AAPL) or multi-stock datasets. Processes 5,000 rows efficiently and scales to larger datasets.

## Testing

- **File**: `test_compute_ofi_features.py`

- **Example Command**:
  ```bash
  python -m unittest test_compute_ofi_features.py -v
  ```

## Notes

- **Single-Stock Dataset**: For AAPL-only data, Cross-Asset OFI columns (`best_level_ofi_AAPL`, `integrated_ofi_AAPL`) duplicate `best_level_ofi` and `integrated_ofi`. When multiple stocks are added, these columns will include other stocks’ OFI for cross-impact analysis.

- **Data Assumptions**: Assumes LOB columns (`bid_px_00` to `ask_sz_09`) are present and non-null. Missing data is forward-filled during resampling.

- **Customization**: Adjust `time_window` or `levels` in `compute_all_ofi_features` for different analysis needs.

- **Debugging**: If OFI values seem off (e.g., mostly zeros), check LOB updates in the CSV for sufficient price/size changes.

## Applications

- **Price Impact Models**: Use `best_level_ofi` or `integrated_ofi` to predict stock price changes (PI¹, PIᴵ models, Page 6).

- **Cross-Impact Analysis**: With multiple stocks, use Cross-Asset OFI for CI¹, CIᴵ models (Page 6).

- **Trading Strategies**: Leverage OFI for short-term return predictions (1-3 minutes, Page 10).

- **Further Analysis**: Combine with LASSO regression to identify significant cross-impact terms, as in the paper (Page 7).

## References

- Cont, R., Cucuringu, M., & Zhang, C. (2023). "Cross-impact of order flow imbalance in equity markets." _Quantitative Finance_. [doi: 10.1080/14697688.2023.2236159](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2023.2236159).

- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events." _Journal of Financial Econometrics_. [doi: 10.48550/arXiv.1011.6402](https://arxiv.org/abs/1011.6402)

## License

Custom license details can be found [here](LICENSE).
