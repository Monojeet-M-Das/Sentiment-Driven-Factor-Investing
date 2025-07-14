import matplotlib.pyplot as plt
import yfinance as yf
from Simulated_Sentiment_Model1 import run_simulated_sentiment_strategy
from Baseline_Model1 import run_baseline_strategy

# constants
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# get strategy return series
sentiment_returns = run_simulated_sentiment_strategy()
baseline_returns = run_baseline_strategy()

# align both series
common_index = sentiment_returns.index.intersection(baseline_returns.index)
sentiment = sentiment_returns.loc[common_index]
baseline = baseline_returns.loc[common_index]

def performance(returns):
    cumulative = (1 + returns).cumprod()
    ann_return = cumulative.iloc[-1] ** (12 / len(returns)) - 1
    vol = returns.std() * (12 ** 0.5)
    sharpe = ann_return / vol if vol != 0 else 0
    return cumulative, ann_return, vol, sharpe

# evaluate strategy performance
cum_sent, r1, v1, s1 = performance(sentiment)
cum_base, r2, v2, s2 = performance(baseline)

# benchmark against S&P500
benchmark = yf.download("^GSPC", start=START_DATE, end=END_DATE)['Close']
benchmark = benchmark.resample('ME').last().pct_change().loc[common_index]
benchmark_cum = (1 + benchmark.fillna(0)).cumprod()

# plot comparison
plt.figure(figsize=(12, 6))
plt.plot(cum_sent, label=f"Sentiment Strategy (Sharpe={s1:.2f})", linewidth=2)
plt.plot(cum_base, label=f"Baseline Strategy (Sharpe={s2:.2f})", linewidth=2)
plt.plot(benchmark_cum, label="S&P 500 (Benchmark)", linestyle='--', color='black', linewidth=2)

plt.title("Strategy Comparison vs S&P 500", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("strategy_comparison.png")
plt.show()

# print table
print("Model       | Annual Return | Volatility | Sharpe")
print(f"Sentiment   | {r1:.2%}        | {v1:.2%}     | {s1:.2f}")
print(f"Baseline    | {r2:.2%}        | {v2:.2%}     | {s2:.2f}")
