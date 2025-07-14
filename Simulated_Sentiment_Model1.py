import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import zscore
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def run_simulated_sentiment_strategy(
    N_TICKERS=500,
    N_SAMPLES=5000,
    START_DATE="2019-01-01",
    END_DATE="2024-12-31",
    LOOKBACK_MONTHS=6,
    MIN_TICKERS=10
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        tickers_df = pd.read_csv("Data/companies_all", header=None, names=['ticker'])
        tickers_df.dropna(inplace=True)
        tickers_df['ticker'] = tickers_df['ticker'].str.strip().str.upper()
        tickers_df = tickers_df[tickers_df['ticker'] != '^GSPC']
        tickers = tickers_df['ticker'].tolist()[:N_TICKERS]
    except Exception as e:
        logging.error(f"Failed to load tickers: {e}")
        raise

    def get_pb_ratios(tickers):
        pb_dict = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                pb = info.get('priceToBook')
                if pb and pb > 0:
                    pb_dict[ticker] = pb
            except Exception as e:
                logging.warning(f"Failed to fetch P/B for {ticker}: {e}")
        return pd.Series(pb_dict)

    companies_df = tickers_df[tickers_df['ticker'].isin(tickers)].copy()
    companies_df['company_name'] = companies_df['ticker'].apply(lambda x: f"{x} Inc.")

    np.random.seed(42)
    sentiments = np.random.choice(['positive', 'neutral', 'negative'], size=N_SAMPLES)
    sample_tickers = np.random.choice(companies_df['ticker'], size=N_SAMPLES)
    headlines = [f"{ticker} Inc. reports earnings." for ticker in sample_tickers]

    sentiment_df = pd.DataFrame({
        'ticker': sample_tickers,
        'sentiment': sentiments,
        'headline': headlines
    })

    sent_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    sentiment_df['sentiment_score'] = sentiment_df['sentiment'].map(sent_map)
    sentiment_df['date'] = pd.to_datetime(np.random.choice(
        pd.date_range("2022-01-01", "2024-01-01", freq='D'),
        size=N_SAMPLES
    ))
    sentiment_df['month'] = sentiment_df['date'].dt.to_period('M')

    monthly_sentiment = (
        sentiment_df.groupby(['month', 'ticker'])['sentiment_score']
        .mean()
        .unstack()
        .T
    )
    monthly_sentiment.index.name = 'ticker'
    monthly_sentiment = monthly_sentiment.T
    monthly_sentiment.index = pd.to_datetime(monthly_sentiment.index.to_timestamp(how='end'))

    logging.info(f"Tickers with sentiment: {list(monthly_sentiment.columns)}")
    logging.info(f"Sentiment matrix shape: {monthly_sentiment.shape}")

    try:
        price_data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
        valid_tickers = price_data.columns[price_data.notna().sum() >= 18]
        price_data = price_data[valid_tickers]
        logging.info(f"Tickers with sufficient price data: {len(valid_tickers)}")

        monthly_prices = price_data.resample('ME').last()
        monthly_prices = monthly_prices.dropna(axis=1, thresh=18)
        momentum = monthly_prices.pct_change(LOOKBACK_MONTHS)
        momentum = momentum.dropna(axis=1, how='all')

        value = -get_pb_ratios(momentum.columns)
        value = pd.DataFrame([value] * len(momentum), index=momentum.index, columns=momentum.columns)
    except Exception as e:
        logging.error(f"Failed to download price data: {e}")
        raise

    common_idx = momentum.index.intersection(value.index).intersection(monthly_sentiment.index)
    momentum = momentum.loc[common_idx]
    value = value.loc[common_idx]
    sentiment = monthly_sentiment.loc[common_idx]

    if momentum.empty:
        logging.warning("Momentum is empty. Simulating fallback price data...")
        fallback_index = pd.date_range("2022-01-31", periods=25, freq='M')
        fallback_columns = [f"TICKER{i}" for i in range(50)]

        monthly_prices = pd.DataFrame(
            np.cumprod(1 + np.random.normal(0.01, 0.05, (25, 50)), axis=0),
            index=fallback_index,
            columns=fallback_columns
        )
        momentum = monthly_prices.pct_change(6)
        value = -pd.DataFrame(np.random.uniform(5, 50, size=momentum.shape),
                              index=momentum.index,
                              columns=momentum.columns)
        sentiment = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=momentum.shape),
            index=momentum.index,
            columns=momentum.columns
        )

    common_idx = momentum.index.intersection(value.index).intersection(sentiment.index)
    strategy_returns = []
    dates = common_idx[LOOKBACK_MONTHS:]

    for i, date in enumerate(dates[:-1]):
        try:
            mom = momentum.loc[date]
            val = value.loc[date]
            sent = sentiment.loc[date]

            combined = pd.DataFrame({
                'momentum': mom,
                'value': val,
                'sentiment': sent
            }).dropna()

            if len(combined) < MIN_TICKERS:
                continue

            combined = combined.apply(zscore).clip(-3, 3)
            combined['alpha'] = combined.mean(axis=1)

            n = max(5, int(0.2 * len(combined)))
            top = combined['alpha'].nlargest(n).index
            bottom = combined['alpha'].nsmallest(n).index

            next_date = dates[i + 1]
            fwd_returns = monthly_prices.loc[next_date] / monthly_prices.loc[date] - 1

            long_ret = fwd_returns[top].mean()
            short_ret = fwd_returns[bottom].mean()
            strategy_returns.append(long_ret - short_ret)

        except Exception as e:
            logging.warning(f"Skipping {date.date()}: {e}")
            continue

    return pd.Series(strategy_returns, index=dates[1:1+len(strategy_returns)])