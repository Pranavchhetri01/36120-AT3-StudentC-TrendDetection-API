"""Fetch the most recent *completed* daily OHLCV candles from Kraken.

Kraken REST OHLC offers limited recent history (up to 720 bars) and
cannot recreate arbitrary historic Experiment 1 snapshots. MarketCap
is not returned. API sourced daily data should be periodically stored.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

API = 'https://api.kraken.com/0/public/OHLC'
PAIRS = {
    'BTC':'XBTUSD', 'ETH':'ETHUSD', 'XRP':'XRPUSD', 'SOL':'SOLUSD',
    'DOGE':'DOGEUSD', 'XMR':'XMRUSD', 'XLM':'XLMUSD', 'TRX':'TRXUSD',
}
ALIASES = {'XBT':'BTC', 'XXBT':'BTC'}


def resolve_token(token: str) -> str:
    token = str(token).upper().strip()
    token = ALIASES.get(token, token)
    if token not in PAIRS: raise ValueError(f'Unsupported token {token!r}; allowed: {list(PAIRS)}')
    return token


def supported_tokens() -> list[str]:
    return list(PAIRS)


def remove_incomplete_bars(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    threshold = pd.Timestamp(now).floor('D')
    dates = pd.to_datetime(frame['Date'], utc=True)
    return frame.loc[dates < threshold].copy().reset_index(drop=True)


def fetch_token(token: str, session: requests.Session | None = None, now: datetime | None = None) -> pd.DataFrame:
    token = resolve_token(token)
    client = session or requests.Session()
    response = client.get(API, params={'pair': PAIRS[token], 'interval': 1440}, timeout=25)
    response.raise_for_status()
    body = response.json()
    if body.get('error'): raise RuntimeError(f'Kraken error ({token}): {body["error"]}')
    result = body.get('result', {})
    key = next((x for x in result if x != 'last'), None)
    if key is None: raise RuntimeError(f'No Kraken candles returned for {token}')
    candles = result[key]
    frame = pd.DataFrame(candles, columns=['Unix','Open','High','Low','Close','VWAP','Volume','Trades'])
    frame['Date'] = pd.to_datetime(frame['Unix'], unit='s', utc=True)
    for col in ['Open','High','Low','Close','Volume']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    frame['Token'] = token
    frame = remove_incomplete_bars(frame, now=now)
    frame = frame[['Token','Date','Open','High','Low','Close','Volume']]
    return frame.drop_duplicates(['Token','Date']).sort_values('Date').reset_index(drop=True)


def fetch_all(tokens: list[str] | None = None) -> pd.DataFrame:
    tokens = tokens or supported_tokens()
    with requests.Session() as session:
        chunks = [fetch_token(token, session=session) for token in tokens]
    return pd.concat(chunks, ignore_index=True).sort_values(['Token','Date']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch Kraken completed daily OHLCV bars')
    parser.add_argument('--output', default='data/raw/crypto_daily_ohlcv.csv')
    parser.add_argument('--tokens', nargs='+', default=supported_tokens())
    args = parser.parse_args()
    frame = fetch_all(args.tokens)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    for token, part in frame.groupby('Token'):
        print(f'{token}: {len(part)} completed candles, {part.Date.min().date()} to {part.Date.max().date()}')
    print(f'Saved {len(frame):,} rows to {target}')
    print('Kraken OHLC does not provide MarketCap; this dataset is a moving recent snapshot.')

if __name__ == '__main__': main()
