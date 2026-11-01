"""Traderbot integration for the Live Market page.

Real market candles come from yfinance and every decision comes from the same Groq-backed
Traderbot logic that the standalone demo used. `model_name` is a display label only: it is
stored with each record but is never sent to the LLM and never selects a different pipeline.
Signal-only / paper trading. There is no order execution anywhere in this module.
"""
import json, os, threading, time, uuid
from datetime import datetime, timezone
from typing import Literal

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api/tradebot', tags=['Traderbot'])

# (yfinance ticker, display label, category)
ASSETS = [
    ('BTC-USD', 'BTC/USD', 'Crypto'), ('ETH-USD', 'ETH/USD', 'Crypto'),
    ('CL=F', 'Crude Oil (WTI)', 'Energy'), ('BZ=F', 'Brent Crude', 'Energy'), ('NG=F', 'Natural Gas', 'Energy'),
    ('GC=F', 'Gold', 'Metals'), ('SI=F', 'Silver', 'Metals'),
    ('EURUSD=X', 'EUR/USD', 'Forex'), ('GBPUSD=X', 'GBP/USD', 'Forex'), ('JPY=X', 'USD/JPY', 'Forex'),
    ('ICLN', 'Clean Energy ETF (ICLN)', 'Green Energy'), ('^GSPC', 'S&P 500', 'Index'),
]
ASSET = {s: (label, cat) for s, label, cat in ASSETS}

# timeframe id -> (yfinance interval, history period, resample rule). yfinance has no native 4h bar,
# so 4h is built from real 1h bars.
TIMEFRAMES = {
    '5m': ('5m', '59d', None), '15m': ('15m', '59d', None), '30m': ('30m', '59d', None),
    '1h': ('1h', '729d', None), '4h': ('1h', '729d', '4h'), '1d': ('1d', 'max', None),
}
# what Yahoo Finance keeps per interval (shown in the UI)
PROVIDER_LIMIT = {'5m': 'Yahoo keeps about 60 days of 5-minute data', '15m': 'Yahoo keeps about 60 days of 15-minute data',
                  '30m': 'Yahoo keeps about 60 days of 30-minute data', '1h': 'Yahoo keeps about 2 years (730 days) of hourly data',
                  '4h': 'Built from hourly data, so about 2 years (730 days)', '1d': 'Yahoo keeps the full daily history (many years)'}
# Model names are display labels, but each one is tied to the market it is meant for. (assets, timeframes); None = any.
MODEL_RULES = {
    'PPO_Gold_5M': (['GC=F'], ['5m']),
    'PPO_Gold_15M': (['GC=F'], ['15m']),
    'PPO_Gold_1h': (['GC=F'], ['1h']),
    'PPO_Gold_4h': (['GC=F'], ['4h']),
    'PPO_Gold_1d': (['GC=F'], ['1d']),

    'PPO_BTC_5M': (['BTC-USD'], ['5m']),
    'PPO_BTC_15M': (['BTC-USD'], ['15m']),
    'PPO_BTC_1h': (['BTC-USD'], ['1h']),

    'PPO_ETH_1H': (['ETH-USD'], ['1h']),
    'PPO_ETH_1d': (['ETH-USD'], ['1d']),

    'PPO_ICLN_1d': (['ICLN'], ['1d']),
    'PPO_Silver_1d': (['SI=F'], ['1d']),
    'PPO_NaturalGas_1d': (['NG=F'], ['1d']),
}

TF_LABEL = {
    '5m': '5 minutes',
    '15m': '15 minutes',
    '30m': '30 minutes',
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day',
}
MAX_ROWS = 5000  # most candles kept/served per asset+timeframe
CACHE_TTL = 20  # seconds; keeps chart polling and decisions from hammering Yahoo
_cache, _cache_lock = {}, threading.Lock()
_log_lock = threading.Lock()


def _lookback() -> int: return int(os.getenv('LOOKBACK', '12'))
def _notional() -> float: return float(os.getenv('PAPER_NOTIONAL', '1000'))


def _nd(price: float) -> int:
    """Decimals that keep FX prices (1.08523) meaningful without cluttering large prices."""
    return 5 if price < 2 else (3 if price < 1000 else 2)


def _candles(symbol: str, tf: str) -> list[dict]:
    key = (symbol, tf)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL: return hit[1]
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(503, 'yfinance is not installed in the backend environment. Run: pip install yfinance')
    interval, period, rule = TIMEFRAMES[tf]
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
    except Exception as e:
        raise HTTPException(502, f'Market data provider error for {symbol}: {e}')
    if df is None or df.empty: raise HTTPException(502, f'No market data returned for {symbol} ({tf}).')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if rule:
        df = df.resample(rule).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna(subset=['Open', 'Close'])
    df = df.tail(MAX_ROWS)
    if len(df) < 5: raise HTTPException(502, f'{symbol}: only {len(df)} candles returned for {tf}.')
    rows = []
    for ts, r in df.iterrows():
        d = _nd(float(r.Close))
        rows.append({'time': ts.isoformat(), 'open': round(float(r.Open), d), 'high': round(float(r.High), d),
                     'low': round(float(r.Low), d), 'close': round(float(r.Close), d), 'volume': round(float(r.Volume), 2)})
    with _cache_lock: _cache[key] = (time.time(), rows)
    return rows


# ---------------- Traderbot decision (same schema and call as the standalone bot) ----------------
class TradeDecision(BaseModel):
    action: Literal['BUY', 'SELL', 'HOLD']
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    stop_loss: float | None = None
    take_profit: float | None = None

SCHEMA = {
    'type': 'object',
    'properties': {
        'action': {'type': 'string', 'enum': ['BUY', 'SELL', 'HOLD']},
        'confidence': {'type': 'number', 'description': '0 to 1'},
        'reasoning': {'type': 'string'},
        'stop_loss': {'type': ['number', 'null']},
        'take_profit': {'type': ['number', 'null']},
    },
    'required': ['action', 'confidence', 'reasoning', 'stop_loss', 'take_profit'],
    'additionalProperties': False,
}
SYSTEM = (
    'You are a disciplined spot trader (crypto, commodities, FX, ETFs and indices). Given recent OHLCV candles, decide '
    'BUY, SELL or HOLD for the next candle. Spot only (no shorting): SELL means exit '
    'an existing long. Base the decision only on the data given, be concise in '
    'reasoning, and prefer HOLD when the signal is unclear. Set stop_loss and '
    'take_profit as prices for BUY, otherwise null.'
)


def _decide(label: str, symbol: str, tf: str, candles: list[dict]) -> tuple[TradeDecision, str]:
    key, model = os.getenv('GROQ_API_KEY', '').strip(), os.getenv('GROQ_MODEL', '').strip()
    if not key or not model: raise HTTPException(503, 'Traderbot engine is not configured: set GROQ_API_KEY and GROQ_MODEL in backend/.env')
    base = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1').rstrip('/')
    user = f'Pair: {label} ({symbol})\nInterval: {tf}\nCandles (oldest -> newest):\n{json.dumps(candles)}'
    body = {'model': model, 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}],
            'temperature': 0.2,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'trade_decision', 'strict': True, 'schema': SCHEMA}}}
    for attempt in range(3):
        try:
            r = requests.post(f'{base}/chat/completions', headers={'Authorization': f'Bearer {key}'}, json=body, timeout=60)
        except requests.RequestException as e:
            raise HTTPException(502, f'Traderbot engine unreachable: {e}')
        if r.status_code == 429 and attempt < 2:
            time.sleep(2 ** attempt * 2); continue
        if not r.ok: raise HTTPException(502, f'Traderbot engine error {r.status_code}: {r.text[:300]}')
        try:
            return TradeDecision.model_validate_json(r.json()['choices'][0]['message']['content']), model
        except Exception as e:
            raise HTTPException(502, f'Traderbot engine returned an invalid decision: {e}')
    raise HTTPException(502, 'Traderbot engine rate-limited the request.')


# ---------------- Decision log (JSONL, one record per decision, model label stored per record) ----------------
def _log_path():
    from .main import RESULTS  # resolved lazily to avoid an import cycle, same pattern as ats.py
    d = RESULTS / 'tradebot'; d.mkdir(parents=True, exist_ok=True)
    return d / 'decisions.jsonl'


def _read_log() -> list[dict]:
    p = _log_path()
    if not p.exists(): return []
    out = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: out.append(json.loads(line))
                except ValueError: pass
    return out


class DecideRequest(BaseModel):
    symbol: str
    interval: str
    model_name: str = Field(min_length=1, max_length=64)
    window_end: str | None = None  # candle time the window ends on (replay); None = latest candles
    window_size: int | None = Field(None, ge=3, le=40)  # candles sent to the model; None = LOOKBACK from .env


def _model_problem(model: str, symbol: str, tf: str) -> str | None:
    rule = MODEL_RULES.get(model)
    if not rule: return None
    assets, tfs = rule
    if (assets is None or symbol in assets) and (tfs is None or tf in tfs): return None
    built = ' / '.join(ASSET[a][0] for a in assets) + (f' on {", ".join(tfs)}' if tfs else '')
    ok = [m for m, (a, t) in MODEL_RULES.items() if (a is None or symbol in a) and (t is None or tf in t)]
    hint = ('Select an appropriate model: ' + ', '.join(ok) + '.') if ok else f'No model is set up for {ASSET[symbol][0]} on {tf}. Try another timeframe or asset.'
    return f'{model} is built for {built}, not {ASSET[symbol][0]} on {tf}. {hint}'


def _check(symbol: str, tf: str):
    if symbol not in ASSET: raise HTTPException(400, f'Unsupported asset: {symbol}')
    if tf not in TIMEFRAMES: raise HTTPException(400, f'Unsupported timeframe: {tf}')


@router.get('/config')
def config():
    return {'assets': [{'symbol': s, 'label': l, 'category': c} for s, l, c in ASSETS],
            'timeframes': [{'id': k, 'label': v} for k, v in TF_LABEL.items()],
            'models': [{'name': m, 'assets': a, 'timeframes': t} for m, (a, t) in MODEL_RULES.items()],
            'provider_limits': PROVIDER_LIMIT, 'max_rows': MAX_ROWS,
            'lookback': _lookback(), 'paper_notional': _notional(),
            'engine_ready': bool(os.getenv('GROQ_API_KEY', '').strip() and os.getenv('GROQ_MODEL', '').strip()),
            'engine': os.getenv('GROQ_MODEL', '').strip() or None}


@router.get('/candles')
def candles(symbol: str, interval: str, limit: int = 300):
    _check(symbol, interval)
    rows = _candles(symbol, interval)
    return {'symbol': symbol, 'interval': interval, 'candles': rows[-max(20, min(limit, MAX_ROWS)):],
            'available': {'first': rows[0]['time'], 'last': rows[-1]['time'], 'count': len(rows), 'limit_note': PROVIDER_LIMIT[interval]}}


@router.post('/decide')
def decide(req: DecideRequest):
    _check(req.symbol, req.interval)
    problem = _model_problem(req.model_name, req.symbol, req.interval)
    if problem: raise HTTPException(400, problem)
    cs = _candles(req.symbol, req.interval)
    label, lb, end = ASSET[req.symbol][0], req.window_size or _lookback(), len(cs) - 1
    if req.window_end:  # replay: the model only sees real candles up to and including window_end
        end = next((i for i, c in enumerate(cs) if c['time'] == req.window_end), None)
        if end is None: raise HTTPException(400, 'That window is outside the loaded candles. Refresh and pick it again.')
    if end + 1 < lb: raise HTTPException(400, f'Not enough candles before that point (need {lb}). Choose a smaller window.')
    d, engine = _decide(label, req.symbol, req.interval, cs[end + 1 - lb:end + 1])
    rec = {'id': uuid.uuid4().hex[:12], 'ts': datetime.now(timezone.utc).isoformat(), 'model_name': req.model_name,
           'symbol': req.symbol, 'asset': label, 'interval': req.interval, 'last_close': cs[end]['close'],
           'candle_time': cs[end]['time'], 'window_end': cs[end]['time'] if req.window_end else None, 'window_size': lb,
           **d.model_dump(), 'engine': engine}
    with _log_lock, open(_log_path(), 'a', encoding='utf-8') as f: f.write(json.dumps(rec) + '\n')
    return rec


@router.get('/history')
def history(model_name: str | None = None, limit: int = 60):
    rows = [r for r in _read_log() if model_name is None or r.get('model_name') == model_name]
    return {'decisions': list(reversed(rows))[:max(1, min(limit, 500))]}


@router.get('/paper')
def paper(model_name: str):
    """Long-only paper account derived from this model name's live decision records (replays are excluded).
    BUY while flat opens a fixed-notional position at the decision's real last close; SELL while long closes it."""
    notional = _notional()
    pos, closed = {}, []
    for r in sorted((x for x in _read_log() if x.get('model_name') == model_name and not x.get('window_end')), key=lambda x: x['ts']):
        s, price = r['symbol'], r['last_close']
        if r['action'] == 'BUY' and s not in pos:
            pos[s] = {'symbol': s, 'asset': r['asset'], 'interval': r['interval'], 'entry': price, 'qty': notional / price, 'opened': r['ts']}
        elif r['action'] == 'SELL' and s in pos:
            p = pos.pop(s)
            closed.append({**p, 'exit': price, 'closed': r['ts'], 'pnl': p['qty'] * (price - p['entry'])})
    open_positions = []
    for p in pos.values():
        try: last = _candles(p['symbol'], p['interval'])[-1]['close']
        except Exception: last = None
        open_positions.append({**p, 'last': last, 'unrealized': None if last is None else p['qty'] * (last - p['entry'])})
    realized = sum(c['pnl'] for c in closed)
    unrealized = sum(p['unrealized'] or 0 for p in open_positions)
    wins = sum(1 for c in closed if c['pnl'] > 0)
    return {'model_name': model_name, 'notional': notional, 'realized': realized, 'unrealized': unrealized,
            'total': realized + unrealized, 'closed_trades': len(closed),
            'win_rate': (wins / len(closed) * 100) if closed else None,
            'open_positions': open_positions, 'closed': list(reversed(closed))[:25]}
