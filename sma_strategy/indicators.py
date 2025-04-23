import pandas as pd

def sma_crossover_signal(
    prices: pd.Series,
    short: int = 50,
    long: int = 200
) -> pd.Series:
    """
    Given a Series of prices, return a Series of signals:
      +1 when SMA_short > SMA_long,
      –1 when SMA_short < SMA_long,
       0 otherwise.
    """
    sma_short = prices.rolling(window=short).mean()
    sma_long  = prices.rolling(window=long).mean()

    signal = pd.Series(0, index=prices.index)
    signal[sma_short > sma_long] = 1
    signal[sma_short < sma_long] = -1
    return signal
