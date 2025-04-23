import os
import pandas as pd
import yfinance as yf

def fetch_data(symbol: str,
               start: str,
               cache_file: str = "cached_data.csv",
               force_refresh: bool = False) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo (auto_adjust), cache to CSV, and
    return a DataFrame with an 'adjusted_close' column.
    """
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
    else:
        print("Fetching data from Yahoo Finance...")
        df = yf.download(symbol, start=start, auto_adjust=True)
        df.columns = ["Close", "High", "Low", "Open", "Volume"]
        df = df.rename_axis("Date").reset_index()
        df["adjusted_close"] = df["Close"]
        df.to_csv(cache_file, index=False)
    return df
