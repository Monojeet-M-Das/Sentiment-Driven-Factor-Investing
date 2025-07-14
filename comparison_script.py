import matplotlib.pyplot as plt
from Simulated_Sentiment_Model1 import run_simulated_sentiment_strategy
from Baseline_Model1 import run_baseline_strategy

# Get strategy return series
sentiment_returns = run_simulated_sentiment_strategy()
baseline_returns = run_baseline_strategy()

# Align both series
common_index = sentiment_returns.index.intersection(baseline_returns.index)
sentiment = sentiment_returns.loc[common_index]
baseline = baseline_returns.loc[common_index]

def performance(returns):
    cumulative = (1 + returns).cumprod()
    ann_return = cumulative.iloc[-1] ** (12 / len(returns)) - 1
    vol = returns.std() * (12 ** 0.5)
    sharpe = ann_return / vol if vol != 0 else 0
    return cumulative, ann_return, vol, sharpe

cum_sent, r1, v1, s1 = performance(sentiment)
cum_base, r2, v2, s2 = performance(baseline)

# Plot
plt.figure(figsize=(12,6))
plt.plot(cum_sent, label=f"Sentiment Strategy (Sharpe={s1:.2f})")
plt.plot(cum_base, label=f"Baseline Strategy (Sharpe={s2:.2f})")
plt.legend()
plt.title("Strategy Comparison: Sentiment vs. Baseline")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print table
print("Model       | Annual Return | Volatility | Sharpe")
print(f"Sentiment   | {r1:.2%}        | {v1:.2%}     | {s1:.2f}")
print(f"Baseline    | {r2:.2%}        | {v2:.2%}     | {s2:.2f}")