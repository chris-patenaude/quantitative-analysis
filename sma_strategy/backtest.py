import pandas as pd

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      - 'adjusted_close'
      - 'Signal'
    returns it with:
      - 'Return'    : pct_change of price
      - 'Strategy'  : shifted Signal * Return
      - equity curve & buy&hold curve
      - win/loss counts and net wins
    """
    df = df.copy()
    df["Return"]   = df["adjusted_close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]
    df.dropna(inplace=True)

    # Equity curves
    df["Equity"]  = (1 + df["Strategy"]).cumprod()
    df["BuyHold"] = (1 + df["Return"]).cumprod()

    # Win/loss metrics
    df["Win"]   = (df["Strategy"] > 0).astype(int)
    df["Loss"]  = (df["Strategy"] < 0).astype(int)
    df["CumulativeWins"]   = df["Win"].cumsum()
    df["CumulativeLosses"] = df["Loss"].cumsum()
    df["NetWinLoss"]       = df["CumulativeWins"] - df["CumulativeLosses"]

    return df
