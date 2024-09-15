import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trend_forecast.covariate_getters import get_lat_long, get_mean_temp


class TestCovariateFunctions(unittest.TestCase):

    @patch("your_module.pd.read_csv")
    def test_get_lat_long(self, mock_read_csv):
        # Mock the CSV data
        mock_df = pd.DataFrame(
            {
                "location": ["01", "02"],
                "latitude": [34.05, 36.16],
                "longitude": [-118.24, -115.15],
            }
        )
        mock_read_csv.return_value = mock_df

        # Test the function
        lat, long = get_lat_long("01")
        self.assertEqual(lat, 34.05)
        self.assertEqual(long, -118.24)

    @patch("your_module.requests.get")
    @patch("your_module.get_lat_long")
    def test_get_mean_temp(self, mock_get_lat_long, mock_requests_get):
        # Mock the latitude and longitude
        mock_get_lat_long.return_value = (34.05, -118.24)

        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "daily": {
                "time": ["2023-01-01", "2023-01-02"],
                "temperature_2m_max": [20, 22],
                "temperature_2m_min": [10, 12],
            }
        }
        mock_requests_get.return_value = mock_response

        # Test the function
        target_date = "2023-01-02"
        result = get_mean_temp("01", target_date)
        expected_dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
        expected_temps = pd.Series([15.0, 17.0], index=expected_dates, name="mean_temp")

        pd.testing.assert_series_equal(result, expected_temps)


if __name__ == "__main__":
    unittest.main()
