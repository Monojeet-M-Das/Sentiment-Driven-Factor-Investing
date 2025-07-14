import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

def run_baseline_strategy(
    N_TICKERS=500,
    START_DATE="2019-01-01",
    END_DATE="2024-12-31",
    LOOKBACK_MONTHS=6,
    MIN_VALID_STOCKS=30,
):
    # load tickers
    tickers_df = pd.read_csv("companies_all", header=None, names=['ticker'])
    tickers_df.dropna(inplace=True)
    tickers_df['ticker'] = tickers_df['ticker'].str.strip().str.upper()
    tickers_df = tickers_df[tickers_df['ticker'] != '^GSPC']
    tickers = tickers_df['ticker'].tolist()[:N_TICKERS]  

    # download price data
    price_data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    valid_tickers = price_data.columns[price_data.notna().sum() >= 18]
    price_data = price_data[valid_tickers]
    monthly_prices = price_data.resample("ME").last()

    # get momentum
    momentum = monthly_prices.pct_change(LOOKBACK_MONTHS).dropna(axis=1, how='all')

    # get P/B
    def get_pb_ratios(tickers):
        pb_dict = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                pb = info.get('priceToBook')
                if pb and pb > 0:
                    pb_dict[ticker] = pb
            except:
                continue
        return pd.Series(pb_dict)

    value = -get_pb_ratios(momentum.columns)
    value = pd.DataFrame([value] * len(momentum), index=momentum.index, columns=momentum.columns)

    # strategy logic
    strategy_returns = []
    dates = momentum.index[LOOKBACK_MONTHS:]

    for i, date in enumerate(dates[:-1]):
        mom = momentum.loc[date]
        val = value.loc[date]
        combined = pd.DataFrame({'momentum': mom, 'value': val}).dropna()

        if len(combined) < MIN_VALID_STOCKS:
            continue

        combined = combined.apply(zscore).clip(-3, 3)
        combined['alpha'] = combined.mean(axis=1)

        n = int(0.2 * len(combined))
        if n < 5:
            continue

        top = combined['alpha'].nlargest(n).index
        bottom = combined['alpha'].nsmallest(n).index

        next_date = dates[i + 1]
        try:
            fwd_return = monthly_prices.loc[next_date] / monthly_prices.loc[date] - 1
            long_ret = fwd_return[top].mean()
            short_ret = fwd_return[bottom].mean()
            strategy_returns.append(long_ret - short_ret)
        except:
            continue

    return pd.Series(strategy_returns, index=dates[1:1+len(strategy_returns)])
