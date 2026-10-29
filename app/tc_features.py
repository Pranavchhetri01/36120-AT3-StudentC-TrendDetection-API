"""Student C Experiment 1 feature engineering and seven-day labels.

Features owned by Student C: Volatility20, MaxPrice20, MaxDrawdown,
GapRisk (rolling 20-day mean of absolute opening gaps, notebook F.4).
Model inputs: MA5, MA20 (Student B) and Return20 (Student A).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOKENS = ('BTC','ETH','XRP','SOL','DOGE','XMR','XLM','TRX')
FEATURES = ['MA5', 'MA20', 'Return20']
LABELS = ['Strong Downtrend','Downtrend','Sideways','Uptrend','Strong Uptrend']
HORIZON = 7


def classify_trend(score: float) -> str | float:
    if pd.isna(score):
        return np.nan
    if score < -0.10: return LABELS[0]
    if score < -0.02: return LABELS[1]
    if score <= 0.02: return LABELS[2]
    if score <= 0.10: return LABELS[3]
    return LABELS[4]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = ['Token','Date','Open','High','Low','Close','Volume']
    missing = set(required) - set(raw.columns)
    if missing: raise ValueError(f'Missing raw columns: {sorted(missing)}')
    df = raw.copy()
    df['Token'] = df['Token'].astype(str).str.upper()
    if not set(df['Token']).issubset(TOKENS):
        raise ValueError('Unsupported cryptocurrency token in raw data')
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df[['Token','Date']].duplicated().any():
        raise ValueError('Duplicate Token-Date rows')
    if df[required[2:6]].isna().any().any() or (df[required[2:6]] <= 0).any().any():
        raise ValueError('Invalid or missing OHLC price')
    if df['Volume'].isna().any() or (df['Volume'] < 0).any():
        raise ValueError('Invalid volume')
    if ((df['High'] < df[['Open','Close','Low']].max(axis=1)) |
        (df['Low'] > df[['Open','Close','High']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC values')
    df = df.sort_values(['Token','Date']).reset_index(drop=True)
    g = df.groupby('Token', sort=False)
    df['MA5'] = g['Close'].transform(lambda x: x.rolling(5).mean())
    df['MA20'] = g['Close'].transform(lambda x: x.rolling(20).mean())
    df['Return20'] = g['Close'].transform(lambda x: x.pct_change(periods=20, fill_method=None))
    df['DailyReturn'] = g['Close'].transform(lambda x: x.pct_change(fill_method=None))
    df['TrendStrength'] = df['MA5'] / df['MA20'] - 1
    df['TrendScore'] = 0.7 * df['TrendStrength'] + 0.3 * df['Return20']
    df['CurrentTC'] = df['TrendScore'].map(classify_trend)
    df['TargetTC'] = df.groupby('Token', sort=False)['CurrentTC'].shift(-HORIZON)
    df['Volatility20'] = df.groupby('Token')['DailyReturn'].transform(lambda x: x.rolling(20).std())
    df['MaxPrice20'] = df.groupby('Token')['High'].transform(lambda x: x.rolling(20).max())
    df['MaxDrawdown'] = (df['MaxPrice20'] - df['Close']) / df['MaxPrice20']
    previous_close = df.groupby('Token')['Close'].shift(1)
    df['PreviousClose'] = previous_close
    df['DailyGap'] = (df['Open'] - previous_close).abs() / previous_close
    df['GapRisk'] = df.groupby('Token')['DailyGap'].transform(lambda x: x.rolling(20).mean())
    return df


def eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=FEATURES + ['TargetTC']).sort_values(['Date','Token']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build Student C Experiment 1 features')
    parser.add_argument('--input', default='data/raw/crypto_daily_ohlcv.csv')
    parser.add_argument('--output', default='data/processed/student_c_trend_features.csv')
    args = parser.parse_args()
    result = engineer_features(pd.read_csv(args.input))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f'Saved {len(result):,} rows to {destination}; eligible: {len(eligible_rows(result)):,}')

if __name__ == '__main__': main()
