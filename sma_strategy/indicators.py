import pandas as pd

def rsi(prices: pd.Series, length: int = 14) -> pd.Series:
    """
    Compute the standard Wilder’s RSI.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)

    # Wilder's smoothing (EMA of gains/losses)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sma_crossover_rsi_signal(
    prices: pd.Series,
    short: int = 50,
    long: int  = 200,
    rsi_period: int = 14,
    rsi_low: float  = 30,
    rsi_high: float = 70
) -> pd.Series:
    """
    SMA crossover with RSI filter:
      +1 when SMA_short > SMA_long AND RSI < rsi_high (avoid overbought)
      -1 when SMA_short < SMA_long AND RSI > rsi_low  (avoid oversold)
       0 otherwise.
    """
    sma_s = prices.rolling(window=short).mean()
    sma_l = prices.rolling(window=long).mean()
    r      = rsi(prices, length=rsi_period)

    signal = pd.Series(0, index=prices.index)
    # go long only if crossover and not overbought
    long_mask  = (sma_s > sma_l) & (r < rsi_high)
    # go short only if crossunder and not oversold
    short_mask = (sma_s < sma_l) & (r > rsi_low)

    signal[ long_mask] =  1
    signal[ short_mask] = -1
    return signal
