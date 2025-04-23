import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION ===
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Please set the ALPHA_VANTAGE_API_KEY environment variable.")
SYMBOL = "AAPL"
SMA_PERIOD = 50


def fetch_data(symbol, api_key, cache_file="cached_data.csv", force_refresh=False):
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"]).set_index("timestamp")
        return df

    print("Fetching data from AlphaVantage...")
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching data.")

    df = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"]).set_index(
        "timestamp"
    )
    df = df.sort_index()
    df.to_csv(cache_file)
    print(f"Data saved to {cache_file}")
    return df


def sma_strategy(df, sma_period):
    df["SMA"] = df["adjusted_close"].rolling(window=sma_period).mean()
    df["Signal"] = 0
    df.loc[df["adjusted_close"] > df["SMA"], "Signal"] = 1
    df.loc[df["adjusted_close"] < df["SMA"], "Signal"] = -1
    return df


def backtest(df):
    df["Return"] = df["adjusted_close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]
    df.dropna(inplace=True)
    df["Equity"] = (1 + df["Strategy"]).cumprod()
    df["BuyHold"] = (1 + df["Return"]).cumprod()
    return df


def plot_results(df):
    df[["Equity", "BuyHold"]].plot(figsize=(12, 6), title="SMA Strategy vs Buy & Hold")
    plt.grid(True)
    plt.ylabel("Cumulative Return")
    plt.show()


if __name__ == "__main__":
    df = fetch_data(SYMBOL, ALPHA_VANTAGE_API_KEY)
    df = sma_strategy(df, SMA_PERIOD)
    df = backtest(df)
    plot_results(df)
