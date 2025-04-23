import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_equity_vs_buyhold(df: pd.DataFrame, short: int, long: int, symbol: str):
    df[["Equity","BuyHold"]].plot(
        figsize=(12,6),
        title=f"SMA {short}×{long} vs Buy&Hold: {symbol}"
    )
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()


def plot_drawdown(df: pd.DataFrame):
    df["Peak"]     = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] / df["Peak"] - 1
    plt.figure(figsize=(12,4))
    plt.fill_between(df.index, df["Drawdown"], alpha=0.4)
    plt.title("Drawdown")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()


def plot_holding_periods(df: pd.DataFrame):
    df["SignalChange"]   = df["Signal"].diff().abs().fillna(0)
    df["SignalBlockID"]  = df["SignalChange"].cumsum()
    holding_periods = df.groupby("SignalBlockID").size()
    holding_periods.hist(
        bins=range(1, holding_periods.max()+1),
        figsize=(8,4)
    )
    plt.title("Distribution of Signal Holding Periods")
    plt.xlabel("Bars Held")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
    print("Number of signal changes:", int(df["SignalChange"].sum()))


def plot_top_trades(df: pd.DataFrame, n: int = 10):
    """
    Plots top n losses on left (red bars extending left)
    and top n gains on right (green bars extending right),
    with two subplots side by side.
    """
    losses = df["Strategy"].nsmallest(n)
    gains  = df["Strategy"].nlargest(n)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        sharey=False,
        figsize=(12,6),
        gridspec_kw={'width_ratios':[1,1]}
    )

    # Left: losses (negative values extend left, zero is at right)
    ax1.barh(losses.index.astype(str), losses.values, color='red')
    ax1.set_title(f"Top {n} Losses")
    ax1.set_xlabel("Return")
    ax1.set_ylabel("Date")
    ax1.grid(True)

    # Right: gains
    ax2.barh(gains.index.astype(str), gains.values, color='green')
    ax2.set_title(f"Top {n} Gains")
    ax2.set_xlabel("Return")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Date")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_cumulative_profit_per_trade(df: pd.DataFrame):
    profits = df["Strategy"][df["Signal"].shift(1) != 0]
    profits.cumsum().plot(
        figsize=(10,4),
        title="Cumulative Profit Per Trade (Approx)"
    )
    plt.grid(True)
    plt.show()


def plot_rolling_sharpe(df: pd.DataFrame, window: int = 60):
    """
    Plot rolling Sharpe ratio of the strategy returns over a given window.
    """
    returns = df["Strategy"]
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    # annualize (assuming 252 trading days)
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)

    plt.figure(figsize=(12,4))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values)
    plt.title(f"Rolling {window}-Day Sharpe Ratio")
    plt.axhline(rolling_sharpe.mean(), color='black', linestyle='--', label='Mean Sharpe')
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()