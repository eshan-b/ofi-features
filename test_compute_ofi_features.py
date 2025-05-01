import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from compute_ofi_features import (
    load_data,
    compute_best_level_ofi,
    compute_multi_level_ofi,
    compute_integrated_ofi,
    compute_all_ofi_features,
)
import io


class TestOFIFeatures(unittest.TestCase):
    def setUp(self):
        # Create mock LOB data with 3 minutes for AAPL and MSFT
        data = {
            "ts_event": [
                "2024-10-21T11:54:29.221064336Z",  # AAPL, minute 11:54
                "2024-10-21T11:54:29.223769812Z",
                "2024-10-21T11:55:00.000000000Z",  # AAPL, minute 11:55
                "2024-10-21T11:55:01.000000000Z",
                "2024-10-21T11:56:00.000000000Z",  # AAPL, minute 11:56
                "2024-10-21T11:54:29.221064336Z",  # MSFT, minute 11:54
                "2024-10-21T11:55:00.000000000Z",  # MSFT, minute 11:55
                "2024-10-21T11:56:00.000000000Z",  # MSFT, minute 11:56
            ],
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "bid_px_00": [
                233.67,
                233.67,
                233.68,
                233.68,
                233.66,
                420.50,
                420.51,
                420.50,
            ],
            "ask_px_00": [
                233.74,
                233.74,
                233.74,
                233.74,
                233.76,
                420.60,
                420.61,
                420.62,
            ],
            "bid_sz_00": [100, 120, 150, 150, 100, 200, 220, 210],
            "ask_sz_00": [200, 200, 180, 180, 230, 150, 140, 145],
            "bid_px_01": [
                233.60,
                233.60,
                233.61,
                233.61,
                233.59,
                420.40,
                420.41,
                420.40,
            ],
            "ask_px_01": [
                233.75,
                233.75,
                233.75,
                233.75,
                233.77,
                420.61,
                420.62,
                420.63,
            ],
            "bid_sz_01": [50, 60, 70, 70, 50, 100, 110, 105],
            "ask_sz_01": [100, 100, 90, 90, 115, 80, 75, 78],
            # Fill remaining levels with dummy data
            **{
                f"bid_px_{i:02d}": [233.67 - i * 0.1 for _ in range(8)]
                for i in range(2, 10)
            },
            **{
                f"ask_px_{i:02d}": [233.74 + i * 0.1 for _ in range(8)]
                for i in range(2, 10)
            },
            **{f"bid_sz_{i:02d}": [50 for _ in range(8)] for i in range(2, 10)},
            **{f"ask_sz_{i:02d}": [100 for _ in range(8)] for i in range(2, 10)},
        }
        self.mock_data = pd.DataFrame(data)

        # Save to temporary CSV for load_data test
        self.csv_content = self.mock_data.to_csv(index=False)
        self.csv_file = io.StringIO(self.csv_content)

    def test_load_data(self):
        # Test loading and preprocessing
        df = load_data(self.csv_file)
        self.assertEqual(len(df), 8, "Should load 8 rows")
        self.assertTrue(
            isinstance(df.index, pd.DatetimeIndex), "Index should be datetime"
        )
        self.assertEqual(df["symbol"].nunique(), 2, "Should have 2 unique symbols")
        expected_cols = ["symbol"] + [
            f"{side}_{metric}_{i:02d}"
            for side in ["bid", "ask"]
            for metric in ["px", "sz"]
            for i in range(10)
        ]
        self.assertTrue(
            all(col in df.columns for col in expected_cols), "Missing expected columns"
        )

    def test_compute_best_level_ofi(self):
        # Test Best-Level OFI for AAPL
        df_aapl = self.mock_data[self.mock_data["symbol"] == "AAPL"].copy()
        df_aapl["ts_event"] = pd.to_datetime(df_aapl["ts_event"])
        df_aapl.set_index("ts_event", inplace=True)

        result = compute_best_level_ofi(df_aapl, time_window="1min")

        # Expected OFI for 11:55:00:
        # From 11:54:29 (last: 233.67,120/233.74,200) to 11:55:00 (last: 233.68,150/233.74,180):
        # Bid: price 233.67->233.68 (increases, contrib=150)
        # Ask: price 233.74->233.74 (same, contrib=200-180=20)
        # OFI = 150 - 20 = 130
        self.assertEqual(len(result), 2, "Should have 2 time windows")
        self.assertAlmostEqual(
            result["best_level_ofi"].iloc[0],
            130.0,
            places=5,
            msg="OFI should be 130 for 11:55",
        )

    def test_compute_multi_level_ofi(self):
        # Test Multi-Level OFI for AAPL
        df_aapl = self.mock_data[self.mock_data["symbol"] == "AAPL"].copy()
        df_aapl["ts_event"] = pd.to_datetime(df_aapl["ts_event"])
        df_aapl.set_index("ts_event", inplace=True)

        # Compute Multi-Level OFI for 2 levels
        result = compute_multi_level_ofi(df_aapl, levels=2, time_window="1min")

        # Check columns
        self.assertEqual(
            list(result.columns),
            ["ofi_level_0", "ofi_level_1"],
            "Should have OFI for 2 levels",
        )

        # Level 0 should match Best-Level OFI (130.0 for 11:55)
        self.assertAlmostEqual(
            result["ofi_level_0"].iloc[0],
            130.0,
            places=5,
            msg="Level 0 OFI should match Best-Level OFI",
        )

        # Level 1: bid price 233.60->233.61, ask price 233.75->233.75, contrib=70-10=60
        self.assertAlmostEqual(
            result["ofi_level_1"].iloc[0],
            60.0,
            places=5,
            msg="Level 1 OFI should be 60 for 11:55",
        )

    def test_compute_integrated_ofi(self):
        # Test Integrated OFI with mock Multi-Level OFI
        multi_level_ofi = pd.DataFrame(
            {
                "ofi_level_0": [130.0, -173.0, -499.0],
                "ofi_level_1": [60.0, -100.0, -200.0],
                "ofi_level_2": [0.0, -50.0, -100.0],
            },
            index=pd.date_range("2024-10-21 11:55:00", periods=3, freq="1min"),
        )

        result = compute_integrated_ofi(multi_level_ofi)

        self.assertEqual(result.shape, (3, 1), "Should have 3 rows, 1 column")
        self.assertTrue(
            "integrated_ofi" in result.columns, "Should have integrated_ofi column"
        )
        # Check if values are centered (mean ~0 after standardization)
        self.assertAlmostEqual(
            result["integrated_ofi"].mean(),
            0.0,
            places=5,
            msg="Integrated OFI should be centered",
        )

    def test_compute_all_ofi_features(self):
        # Test end-to-end pipeline with mock CSV
        self.csv_file.seek(0)  # Reset StringIO cursor
        result = compute_all_ofi_features(self.csv_file)

        # Expected columns
        expected_cols = (
            ["ts_event", "best_level_ofi"]
            + [f"ofi_level_{i}" for i in range(10)]
            + ["integrated_ofi", "symbol"]
            + [
                "best_level_ofi_AAPL",
                "best_level_ofi_MSFT",
                "integrated_ofi_AAPL",
                "integrated_ofi_MSFT",
            ]
        )

        # Check columns
        self.assertTrue(
            all(col in result.columns for col in expected_cols),
            "Missing expected columns",
        )

        # Check row count (2 stocks, 2 windows each: 11:55, 11:56)
        self.assertEqual(len(result), 4, "Should have 4 rows (2 per stock, 2 windows)")

        # Check AAPL Best-Level OFI for 11:55
        aapl_row = result[
            (result["symbol"] == "AAPL") & (result["ts_event"] == "2024-10-21 11:55:00")
        ]
        self.assertAlmostEqual(
            aapl_row["best_level_ofi"].iloc[0],
            130.0,
            places=5,
            msg="AAPL Best-Level OFI should be 130 for 11:55",
        )

        # Check Cross-Asset OFI
        self.assertTrue(
            "best_level_ofi_MSFT" in result.columns,
            "Should include MSFT Cross-Asset OFI",
        )
        self.assertFalse(
            result["best_level_ofi_MSFT"].isna().all(),
            "MSFT Cross-Asset OFI should have values",
        )


if __name__ == "__main__":
    unittest.main()
