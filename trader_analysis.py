"""
Primetrade.ai - Data Science Intern Assignment
Trader Performance vs Market Sentiment Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📊 TRADER PERFORMANCE ANALYSIS")
print("="*60)

# ============================================
# LOAD DATA
# ============================================

print("\n📂 Loading data...")

sentiment_df = pd.read_csv('fear_greed_index.csv')
trader_df = pd.read_csv('historical_data.csv')

print(f"Sentiment Data: {sentiment_df.shape}")
print(f"Trader Data: {trader_df.shape}")

# ============================================
# CLEAN COLUMN NAMES
# ============================================

# Rename columns for easier access
trader_df.rename(columns={
    'Timestamp IST': 'datetime',
    'Closed PnL': 'pnl',
    'Size USD': 'size_usd'
}, inplace=True)

print("\n✅ Columns renamed")

# ============================================
# CONVERT DATES - FIXED
# ============================================

print("\n📅 Converting dates...")

# Sentiment data
sentiment_df['Date'] = pd.to_datetime(sentiment_df['date'])

# Trader data - format: "02-12-2024 22:50"
trader_df['datetime'] = pd.to_datetime(trader_df['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
trader_df['Date'] = trader_df['datetime'].dt.date

# Remove invalid dates
trader_df = trader_df.dropna(subset=['Date'])
print(f"Trader data after cleaning: {len(trader_df)} rows")

# ============================================
# DAILY METRICS
# ============================================

print("\n📊 Creating daily metrics...")

# Daily PnL
daily_pnl = trader_df.groupby('Date')['pnl'].sum().reset_index()
daily_pnl.rename(columns={'pnl': 'total_pnl'}, inplace=True)

# Number of trades per day
trade_count = trader_df.groupby('Date').size().reset_index(name='trade_count')

# Win rate per day
wins = trader_df[trader_df['pnl'] > 0].groupby('Date').size().reset_index(name='wins')
daily_winrate = pd.merge(trade_count, wins, on='Date', how='left')
daily_winrate['wins'] = daily_winrate['wins'].fillna(0)
daily_winrate['win_rate'] = daily_winrate['wins'] / daily_winrate['trade_count']

# Average trade size
avg_size = trader_df.groupby('Date')['size_usd'].mean().reset_index(name='avg_trade_size')

# Long/Short ratio per day
longs = trader_df[trader_df['Side'] == 'BUY'].groupby('Date').size().reset_index(name='longs')
shorts = trader_df[trader_df['Side'] == 'SELL'].groupby('Date').size().reset_index(name='shorts')
ls_ratio = pd.merge(longs, shorts, on='Date', how='outer').fillna(0)
ls_ratio['long_short_ratio'] = ls_ratio['longs'] / (ls_ratio['shorts'] + 1)

# Merge all
daily_data = daily_pnl
daily_data = pd.merge(daily_data, trade_count, on='Date')
daily_data = pd.merge(daily_data, daily_winrate[['Date', 'win_rate']], on='Date')
daily_data = pd.merge(daily_data, avg_size, on='Date', how='left')
daily_data = pd.merge(daily_data, ls_ratio[['Date', 'long_short_ratio']], on='Date', how='left')

# Avg PnL per trade
daily_data['avg_pnl_per_trade'] = daily_data['total_pnl'] / daily_data['trade_count']

print("✅ Daily metrics created!")

# ============================================
# MERGE WITH SENTIMENT
# ============================================

print("\n🔗 Merging with sentiment data...")

daily_data['Date'] = pd.to_datetime(daily_data['Date'])
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

merged_df = pd.merge(daily_data, sentiment_df[['Date', 'classification']], on='Date', how='inner')
merged_df.rename(columns={'classification': 'Sentiment'}, inplace=True)

# Combine Extreme Fear/Fear and Extreme Greed/Greed
merged_df['Sentiment'] = merged_df['Sentiment'].replace({
    'Extreme Fear': 'Fear',
    'Extreme Greed': 'Greed'
})

print(f"Merged data: {merged_df.shape[0]} days")

# ============================================
# ANALYSIS
# ============================================

print("\n" + "="*60)
print("📊 ANALYSIS: FEAR vs GREED")
print("="*60)

fear_data = merged_df[merged_df['Sentiment'] == 'Fear']
greed_data = merged_df[merged_df['Sentiment'] == 'Greed']

print(f"\nFear days: {len(fear_data)}")
print(f"Greed days: {len(greed_data)}")

print("\n" + "-"*50)
print("💰 PERFORMANCE COMPARISON")
print("-"*50)

fear_pnl = fear_data['total_pnl'].mean()
greed_pnl = greed_data['total_pnl'].mean()
print(f"\nAverage Daily PnL:")
print(f"  Fear Days:  ₹{fear_pnl:,.2f}")
print(f"  Greed Days: ₹{greed_pnl:,.2f}")

fear_win = fear_data['win_rate'].mean()
greed_win = greed_data['win_rate'].mean()
print(f"\nWin Rate:")
print(f"  Fear Days:  {fear_win:.2%}")
print(f"  Greed Days: {greed_win:.2%}")

fear_trades = fear_data['trade_count'].mean()
greed_trades = greed_data['trade_count'].mean()
print(f"\nTrade Frequency:")
print(f"  Fear Days:  {fear_trades:.0f} trades/day")
print(f"  Greed Days: {greed_trades:.0f} trades/day")

# ============================================
# VISUALIZATION
# ============================================

print("\n📊 Creating charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Trader Performance: Fear vs Greed Days', fontsize=14)

# PnL
sns.boxplot(data=merged_df, x='Sentiment', y='total_pnl', ax=axes[0,0])
axes[0,0].set_title('Daily PnL Distribution')
axes[0,0].set_ylabel('PnL')

# Win Rate
sns.boxplot(data=merged_df, x='Sentiment', y='win_rate', ax=axes[0,1])
axes[0,1].set_title('Win Rate Distribution')
axes[0,1].set_ylabel('Win Rate')

# Trade Count
sns.boxplot(data=merged_df, x='Sentiment', y='trade_count', ax=axes[1,0])
axes[1,0].set_title('Trade Frequency')
axes[1,0].set_ylabel('Number of Trades')

# PnL per Trade
sns.boxplot(data=merged_df, x='Sentiment', y='avg_pnl_per_trade', ax=axes[1,1])
axes[1,1].set_title('Average PnL per Trade')
axes[1,1].set_ylabel('PnL per Trade')

plt.tight_layout()
plt.savefig('analysis_charts.png', dpi=150)
print("✅ Chart saved: analysis_charts.png")
plt.show()

# ============================================
# INSIGHTS
# ============================================

insights = f"""
================================================================================
📝 INSIGHTS SUMMARY
================================================================================

KEY FINDINGS:
-------------

1. PERFORMANCE:
   • Greed days: ₹{greed_pnl:,.2f} average PnL
   • Fear days: ₹{fear_pnl:,.2f} average PnL
   • Greed days outperform by ₹{greed_pnl - fear_pnl:,.2f}

2. WIN RATE:
   • Greed days: {greed_win:.2%}
   • Fear days: {fear_win:.2%}
   • Difference: {(greed_win - fear_win)*100:.1f}% higher on Greed

3. TRADE FREQUENCY:
   • {greed_trades:.0f} trades/day on Greed vs {fear_trades:.0f} on Fear
   • {(greed_trades/fear_trades - 1)*100:.0f}% more active during Greed

================================================================================
💡 STRATEGY RECOMMENDATIONS
================================================================================

1. Sentiment-Based Position Sizing:
   - Reduce position sizes by 30% during Fear days
   - Normal sizing during Greed days

2. Dynamic Trade Frequency:
   - Increase trading frequency during Greed days
   - Reduce during Fear days

3. Risk Management:
   - Implement stricter stop-losses during Fear days
   - Allow normal risk parameters during Greed days

================================================================================
"""

print(insights)

with open('insights_summary.txt', 'w') as f:
    f.write(insights)

print("\n✅ Analysis complete!")
print("\nFiles generated:")
print("  - analysis_charts.png")
print("  - insights_summary.txt")