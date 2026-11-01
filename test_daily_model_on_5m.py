"""Exploratory cross-timeframe diagnostic: daily-trained PPO policies on 5-minute XAUUSD.

NO TRAINING is performed. This is not a valid profitability claim: daily feature windows
become bar-count windows, the instrument may differ (XAUUSD vs GC=F), and the daily GARCH
feature is approximated using rolling 10-bar volatility because the original daily GARCH
parameters are not valid for 5-minute returns. Treat B results as a rough input-transfer
probe, not a clean ATS ablation.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)


def load_bars(path, year):
    df = pd.read_csv(path, sep=';')
    required = {'Date','Open','High','Low','Close','Volume'}
    if not required.issubset(df.columns):
        raise ValueError(f'Expected semicolon-separated columns {sorted(required)}; got {df.columns.tolist()}')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M', errors='coerce')
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Date','Open','High','Low','Close']).sort_values('Date')
    df = df.drop_duplicates('Date').set_index('Date')
    df = df[df.index.year == year].copy()
    if len(df) < 100:
        raise ValueError(f'Only {len(df)} rows found for {year}')
    # Basic OHLC sanity check
    bad = (df['High'] < df[['Open','Close','Low']].max(axis=1)) | (df['Low'] > df[['Open','Close','High']].min(axis=1))
    if bad.any():
        raise ValueError(f'{int(bad.sum())} rows have inconsistent OHLC values')
    return df


def make_features(bars):
    c = bars['Close']
    ret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=bars.index)
    f['log_return'] = ret
    # These periods intentionally mean bars, not days, for this exploratory test.
    f['return_5d'] = c.pct_change(5)
    f['return_10d'] = c.pct_change(10)
    f['rolling_vol_10d'] = ret.rolling(10).std()
    f['rsi_14'] = rsi(c, 14)
    # Approximation only; daily GARCH parameters should not be reused on 5-minute returns.
    f['garch_cond_vol'] = ret.rolling(10).std()
    # Use a rolling, past-only 85th percentile volatility regime estimate.
    past_vol = f['garch_cond_vol'].shift(1)
    past_q = past_vol.rolling(288*20, min_periods=288).quantile(0.85)
    f['regime_flag'] = (f['garch_cond_vol'] > past_q).astype(float)
    f['momentum_signal'] = c.pct_change(20)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def run_one(name, model_dir, bars, features, outdir, cost):
    md = Path(model_dir)
    meta = json.loads((md/'metadata.json').read_text(encoding='utf-8'))
    with open(md/'scaler.pkl','rb') as fh:
        scaler = pickle.load(fh)
    model = PPO.load(str(md/'model.zip'), device='cpu')
    cols = meta['feature_cols']
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f'{name}: missing model features: {missing}')
    data = bars.join(features[cols], how='inner').dropna(subset=cols + ['Close'])
    if len(data) < 100:
        raise ValueError(f'{name}: only {len(data)} usable rows after feature warmup')
    X = scaler.transform(data[cols])
    # The project's custom scaler may return a DataFrame; iterate over numeric rows, not column names.
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    # The project environment may append current position to the feature vector.
    # Reconstruct that common observation format from the saved model's input shape.
    obs_dim = int(model.observation_space.shape[0])
    if obs_dim not in (len(cols), len(cols) + 1):
        raise ValueError(f'{name}: model expects observation size {obs_dim}, but feature count is {len(cols)}. '
                         'This script cannot safely reconstruct the original observation.')
    actions = []
    position = 0
    pos = []
    transitions = []
    for i, feat_obs in enumerate(X):
        obs = np.asarray(feat_obs, dtype=np.float32)
        if obs_dim == len(cols) + 1:
            obs = np.concatenate([obs, np.asarray([position], dtype=np.float32)])
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        actions.append(action)
        old = position
        if action == 1:
            position = 1
        elif action == 2:
            position = 0
        pos.append(position)
        if position != old:
            transitions.append((i, old, position, action))
    # Actions: 0=hold, 1=long, 2=flat/sell. P&L begins on the next bar.
    data = data.copy()
    data['action'] = actions
    data['position'] = pos
    data['next_log_return'] = np.log(data['Close'].shift(-1) / data['Close'])
    data['turnover'] = data['position'].diff().abs().fillna(abs(data['position'].iloc[0]))
    data['strategy_logret'] = data['position'] * data['next_log_return'] - data['turnover'] * cost
    data = data.iloc[:-1].copy()
    data['equity'] = np.exp(data['strategy_logret'].fillna(0).cumsum())
    data['buy_signal'] = (data['position'].diff().fillna(data['position'].iloc[0]) > 0)
    data['sell_signal'] = (data['position'].diff().fillna(data['position'].iloc[0]) < 0)
    # Trade ledger from position transitions, exit at close of transition bar.
    ledger=[]; entry=None
    for i, row in enumerate(data.itertuples()):
        if row.buy_signal:
            entry=(row.Index, float(row.Close))
        if row.sell_signal and entry is not None:
            t0,p0=entry; p1=float(row.Close)
            ledger.append({'entry_time':t0,'entry_price':p0,'exit_time':row.Index,'exit_price':p1,
                           'holding_bars':int((row.Index-t0).total_seconds()/300),
                           'gross_return_pct':(p1/p0-1)*100,
                           'net_return_pct':((p1/p0)*(1-cost)**2-1)*100})
            entry=None
    if entry is not None:
        ledger.append({'entry_time':entry[0],'entry_price':entry[1],'exit_time':None,'exit_price':None,
                       'holding_bars':None,'gross_return_pct':None,'net_return_pct':None})
    tag=f'{name}_{data.index.min():%Y}_{data.index.max():%Y}'
    data.to_csv(outdir/f'{tag}_bar_log.csv', index_label='Date')
    pd.DataFrame(ledger).to_csv(outdir/f'{tag}_trades.csv', index=False)
    r=data['strategy_logret'].dropna()
    eq=data['equity']
    dd=(eq/eq.cummax()-1).min()*100
    bars_per_year=252*24*12
    sharpe=(r.mean()/r.std()*np.sqrt(bars_per_year)) if r.std()>0 else np.nan
    summary={'model':name,'bars':len(data),'start':str(data.index.min()),'end':str(data.index.max()),
             'total_return_pct':(eq.iloc[-1]-1)*100,'annualized_sharpe_rough':sharpe,'max_drawdown_pct':dd,
             'entries':int(data['buy_signal'].sum()),'exits':int(data['sell_signal'].sum()),
             'time_in_market_pct':float(data['position'].mean()*100),'transaction_cost_per_position_change':cost,
             'feature_note':'Cross-timeframe diagnostic only; daily feature windows now mean bars; B GARCH feature approximated.'}
    pd.DataFrame([summary]).to_csv(outdir/f'{tag}_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(13,5)); ax.plot(data.index,data['equity'],linewidth=0.8)
    ax.set_title(f'{name}: daily-trained PPO on {data.index.min():%Y} 5-minute bars (diagnostic)')
    ax.set_ylabel('Equity (start = 1.0)'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(outdir/f'{tag}_equity.png',dpi=140); plt.close(fig)
    # Readable chart for first 10 trading days, with signals.
    start=data.index.min(); end=start+pd.Timedelta(days=14); chart=data.loc[start:end]
    fig,ax=plt.subplots(figsize=(14,6)); ax.plot(chart.index,chart['Close'],linewidth=.8,label='Close')
    ax.scatter(chart.index[chart['buy_signal']],chart.loc[chart['buy_signal'],'Close'],marker='^',s=45,label='BUY')
    ax.scatter(chart.index[chart['sell_signal']],chart.loc[chart['sell_signal'],'Close'],marker='v',s=45,label='SELL/FLAT')
    ax.set_title(f'{name}: signals on first 14 calendar days'); ax.set_ylabel('XAUUSD price'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(outdir/f'{tag}_signals_first14days.png',dpi=140); plt.close(fig)
    print('\n',json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=r'data\XAU_5m_data.csv')
    ap.add_argument('--year',type=int,default=2024)
    ap.add_argument('--out_dir',default=r'results\cross_timeframe_5m_2024')
    ap.add_argument('--cost',type=float,default=0.0005)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    bars=load_bars(args.csv,args.year)
    features=make_features(bars)
    print(f'Bars for {args.year}: {len(bars):,}; {bars.index.min()} -> {bars.index.max()}')
    print('WARNING: exploratory only; no training. 5m features change the meaning of daily windows.')
    summaries=[]
    for name in ['A','B']:
        model_dir=Path('models')/'fixed_split'/'selected'/name
        run_one(name,model_dir,bars,features,out,args.cost)
    print(f'\nSaved outputs to: {out.resolve()}')
    print('Outputs: *_bar_log.csv, *_trades.csv, *_summary.csv, *_equity.png, *_signals_first14days.png')

if __name__=='__main__': main()
