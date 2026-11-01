import os, sys, uuid, threading, subprocess, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
_configured_root=Path(os.getenv('RL_PROJECT_ROOT','')).expanduser()
ROOT=_configured_root if str(os.getenv('RL_PROJECT_ROOT','')).strip() and _configured_root.is_dir() else Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA=Path(os.getenv('DATA_DIR') or (ROOT/'data'))
RESULTS=Path(os.getenv('RESULTS_DIR') or (ROOT/'results'))
SCRIPT=ROOT/'run_seed_selection.py'
app=FastAPI(title='RL Trader API',version='0.2.0')
from .ats import router as ats_router
app.include_router(ats_router)
from .tradebot import router as tradebot_router
app.include_router(tradebot_router)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])
jobs={}; lock=threading.Lock()

def safe_csv(name, folder):
    if Path(name).name != name or not name.lower().endswith('.csv'): raise HTTPException(400,'Use a CSV filename, not a path.')
    p=(folder/name).resolve()
    try:p.relative_to(folder.resolve())
    except ValueError:raise HTTPException(400,'File outside allowed folder.')
    if not p.is_file():raise HTTPException(404,f'CSV not found: {name}')
    return p

def worker(jid,cmd):
    with lock: jobs[jid]['status']='running'; jobs[jid]['started_at']=datetime.now(timezone.utc).isoformat()
    try:
        p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        with lock:jobs[jid]['pid']=p.pid
        lines=[]
        for line in p.stdout:
            lines.append(line.rstrip())
            with lock:jobs[jid]['log']='\n'.join(lines[-500:])
        code=p.wait()
        with lock:jobs[jid].update(status='completed' if code==0 else 'failed',return_code=code,finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        with lock:jobs[jid].update(status='failed',log=str(e),finished_at=datetime.now(timezone.utc).isoformat())

class FeatureRequest(BaseModel):
    source_csv:str
    output_csv:str='features_generated.csv'
    use_ats:bool=True
    features:list[str]|None=None  # base features to keep (log_return is always kept: the model replay needs it)
class TrainRequest(BaseModel):
    features_csv:str
    train_end:str='2019-12-31'
    valid_end:str='2022-12-31'
    timesteps:int=Field(60000,ge=1000,le=5000000)
    seeds:int=Field(3,ge=1,le=20)
    experiments:list[str]=Field(default_factory=lambda:['A','B'])
    turnover_penalty:float=Field(0.0008,ge=0,le=.1)
    out_dir:str='results/frontend_experiment'
    model_dir:str='models/frontend_experiment'

@app.get('/api/health')
def health():
    return {'status':'ok','project_root':str(ROOT),'project_root_exists':ROOT.is_dir(),'training_script_exists':SCRIPT.is_file(),'data_dir':str(DATA),'data_dir_exists':DATA.is_dir(),'results_dir':str(RESULTS),'results_dir_exists':RESULTS.is_dir()}
@app.get('/api/data/files')
def data_files():
    return {'files':[{'name':p.name,'size_bytes':p.stat().st_size,'modified':datetime.fromtimestamp(p.stat().st_mtime).isoformat()} for p in sorted(DATA.glob('*.csv'))] if DATA.is_dir() else [],'warning':None if DATA.is_dir() else f'Data folder not found: {DATA}'}
@app.get('/api/data/summary')
def data_summary(filename:str):
    p=safe_csv(filename,DATA)
    try:d=pd.read_csv(p, sep=None, engine='python')
    except Exception as e:raise HTTPException(400,f'Could not read CSV: {e}')
    dc=next((c for c in d if c.lower() in ('date','timestamp','datetime','time')),None); cc=next((c for c in d if c.lower()=='close'),None)
    dates=pd.to_datetime(d[dc],errors='coerce').dropna() if dc else pd.Series(dtype='datetime64[ns]')
    close=pd.to_numeric(d[cc],errors='coerce').dropna() if cc else pd.Series(dtype=float)
    return {'filename':filename,'rows':len(d),'columns':list(d.columns),'missing_values':int(d.isna().sum().sum()),'duplicate_rows':int(d.duplicated().sum()),'date_min':str(dates.min()) if len(dates) else None,'date_max':str(dates.max()) if len(dates) else None,'last_close':float(close.iloc[-1]) if len(close) else None,'preview':d.tail(12).astype(object).where(pd.notna(d.tail(12)),None).to_dict('records')}
@app.get('/api/results/files')
def result_files():
    if not RESULTS.is_dir():return {'files':[]}
    return {'files':[{'name':str(p.relative_to(RESULTS)),'size_bytes':p.stat().st_size,'modified':datetime.fromtimestamp(p.stat().st_mtime).isoformat()} for p in sorted(RESULTS.rglob('*'),key=lambda x:x.stat().st_mtime,reverse=True) if p.is_file() and p.suffix.lower() in ('.csv','.md','.png','.json')][:100]}
@app.post('/api/features/generate')
def generate_features(req:FeatureRequest):
    src=safe_csv(req.source_csv,DATA)
    if Path(req.output_csv).name!=req.output_csv or not req.output_csv.endswith('.csv'):raise HTTPException(400,'output_csv must be a filename ending in .csv')
    d=pd.read_csv(src, sep=None, engine='python'); dc=next((c for c in d if c.lower() in ('date','timestamp','datetime','time')),None); cc=next((c for c in d if c.lower()=='close'),None)
    if not dc or not cc:raise HTTPException(400,'CSV must contain Date and Close columns')
    d['Date']=pd.to_datetime(d[dc],errors='coerce'); d['Close']=pd.to_numeric(d[cc],errors='coerce'); d=d.dropna(subset=['Date','Close']).sort_values('Date').drop_duplicates('Date')
    c=d['Close']; lr=np.log(c).diff(); f=pd.DataFrame({'Date':d.Date,'log_return':lr,'return_5d':c.pct_change(5),'return_10d':c.pct_change(10),'rolling_vol_10d':lr.rolling(10).std()})
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); f['rsi_14']=100-100/(1+gain/loss.replace(0,np.nan))
    if req.use_ats:
        f['garch_cond_vol']=lr.rolling(20).std(); med=f.garch_cond_vol.rolling(252,min_periods=60).median(); f['regime_flag']=(f.garch_cond_vol>med).astype(float); f['momentum_signal']=c.pct_change(20)
    if req.features is not None:
        drop=[c for c in ('return_5d','return_10d','rolling_vol_10d','rsi_14') if c not in req.features]; f=f.drop(columns=drop)
    f=f.replace([np.inf,-np.inf],np.nan).dropna(); RESULTS.mkdir(parents=True,exist_ok=True); out=(RESULTS/req.output_csv).resolve()
    try:out.relative_to(RESULTS.resolve())
    except ValueError:raise HTTPException(400,'Output must remain inside results folder')
    f.to_csv(out,index=False)
    return {'status':'ok','source':req.source_csv,'output':str(out),'rows':len(f),'columns':list(f.columns),'note':'garch_cond_vol is a rolling volatility proxy, not fitted GARCH.' if req.use_ats else None}
@app.post('/api/training/start')
def start_training(req:TrainRequest):
    if not ROOT.is_dir() or not SCRIPT.is_file():raise HTTPException(400,'Set RL_PROJECT_ROOT in backend/.env to the folder containing run_seed_selection.py')
    fp=safe_csv(req.features_csv,RESULTS)
    if not req.experiments or any(x not in ('A','B','C') for x in req.experiments):raise HTTPException(400,'Experiments must be A, B, and/or C')
    if '..' in Path(req.out_dir).parts or '..' in Path(req.model_dir).parts or Path(req.out_dir).is_absolute() or Path(req.model_dir).is_absolute():raise HTTPException(400,'Output folders must be safe relative paths')
    featarg=str(fp.relative_to(ROOT)) if fp.is_relative_to(ROOT) else str(fp)
    cmd=[sys.executable,str(SCRIPT),'--features_csv',featarg,'--train_end',req.train_end,'--valid_end',req.valid_end,'--timesteps',str(req.timesteps),'--seeds',str(req.seeds),'--experiments',*req.experiments,'--turnover_penalty',str(req.turnover_penalty),'--out_dir',req.out_dir,'--model_dir',req.model_dir]
    jid=uuid.uuid4().hex[:12]
    with lock:jobs[jid]={'job_id':jid,'status':'queued','command':cmd,'log':'','created_at':datetime.now(timezone.utc).isoformat()}
    threading.Thread(target=worker,args=(jid,cmd),daemon=True).start()
    return {'job_id':jid,'status':'queued'}
@app.get('/api/training/jobs/{job_id}')
def get_job(job_id:str):
    with lock:
        if job_id not in jobs:raise HTTPException(404,'Job not found in this API session')
        return jobs[job_id]
@app.get('/api/training/jobs')
def list_jobs():
    with lock:return {'jobs':list(jobs.values())}

# Real-data market workspace and saved-artifact discovery.
@app.get('/api/market/candles')
def market_candles(filename: str, date_column: str = 'Date', limit: int = 1500, start: str | None = None, end: str | None = None):
    p=safe_csv(filename,DATA)
    try:
        # Sniff comma/semicolon/tab/pipe instead of silently treating a whole row as one column.
        d=pd.read_csv(p, sep=None, engine='python')
    except Exception as e: raise HTTPException(400,f'Could not parse CSV: {e}')
    d.columns=[str(c).strip().lstrip('\ufeff') for c in d.columns]
    low={c.lower():c for c in d.columns}
    dc=low.get(date_column.lower()) or next((low[k] for k in ('date','datetime','timestamp','time','open time') if k in low),None)
    if not dc: raise HTTPException(400,f'No date/time column detected. Columns: {list(d.columns)}')
    actual={k:low.get(k) for k in ('open','high','low','close','volume')}
    if not actual['close']: raise HTTPException(400,'No Close column detected; chart requires at least Date and Close.')
    d['_date']=pd.to_datetime(d[dc],errors='coerce',utc=True)
    for k,c in actual.items():
        if c: d[k]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['_date','close']).sort_values('_date')
    if start: d=d[d['_date']>=pd.to_datetime(start,utc=True)]
    if end: d=d[d['_date']<=pd.to_datetime(end,utc=True)+pd.Timedelta(days=1)]
    total=len(d); d=d.tail(max(20,min(limit,5000)))
    rows=[]
    for _,r in d.iterrows():
        close=float(r['close']); o=float(r['open']) if actual['open'] and pd.notna(r.get('open')) else close
        h=float(r['high']) if actual['high'] and pd.notna(r.get('high')) else max(o,close)
        l=float(r['low']) if actual['low'] and pd.notna(r.get('low')) else min(o,close)
        rows.append({'time':r['_date'].isoformat(),'open':o,'high':h,'low':l,'close':close,'volume':float(r['volume']) if actual['volume'] and pd.notna(r.get('volume')) else None})
    return {'filename':filename,'rows':rows,'total_matching_rows':total,'returned_rows':len(rows),'columns':list(d.columns),'ohlc_available':all(actual[k] for k in ('open','high','low','close')),'source':'local CSV'}

@app.get('/api/models')
def list_models():
    roots=[ROOT/'models', ROOT/'archives']
    found=[]
    for base in roots:
        if not base.is_dir(): continue
        for p in base.rglob('model.zip'):
            rel=str(p.relative_to(ROOT))
            meta=p.parent/'metadata.json'
            details={}
            if meta.is_file():
                try: details=json.loads(meta.read_text(encoding='utf-8'))
                except Exception: pass
            found.append({'id':rel,'name':p.parent.name+' · '+p.parent.parent.name,'path':rel,'size_bytes':p.stat().st_size,'metadata':details,'has_scaler':(p.parent/'scaler.pkl').exists(),'has_metadata':meta.exists()})
    found.sort(key=lambda x:('selected' not in x['path'].lower(),x['path']))
    return {'models':found[:500],'note':'Model discovery only. Compatibility must be checked before inference.'}

TABLE_FILES=('summary_test.csv','summary_valid.csv','runs.csv','action_diagnostics.csv','seed_action_history.csv','walkforward_runs.csv')
def _exp_label(rel):
    if rel=='': return 'Results root (top-level files)'
    low=rel.lower()
    kind='Walk-forward' if 'walkforward' in low or 'walk_forward' in low else ('Fixed split' if 'fixed' in low else 'Experiment')
    return f'{kind} · {rel}'
def _exp_dir(experiment):
    rel=Path(experiment or '.')
    if rel.is_absolute() or '..' in rel.parts: raise HTTPException(400,'Invalid experiment folder')
    d=(RESULTS/rel).resolve()
    try:d.relative_to(RESULTS.resolve())
    except ValueError:raise HTTPException(400,'Invalid experiment folder')
    return d
@app.get('/api/evaluations')
def list_evaluations(experiment:str=''):
    out=[]
    if RESULTS.is_dir():
        for p in RESULTS.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.csv','.md','.json','.png'):
                out.append({'name':str(p.relative_to(RESULTS)),'extension':p.suffix.lower(),'size_bytes':p.stat().st_size,'modified':datetime.fromtimestamp(p.stat().st_mtime).isoformat()})
    out.sort(key=lambda x:x['modified'],reverse=True)
    base=_exp_dir(experiment)
    summaries={}
    for name in TABLE_FILES:
        p=base/name
        if p.is_file() and p.stat().st_size>0:
            try: summaries[name]=pd.read_csv(p, sep=None, engine='python').replace({np.nan:None}).to_dict('records')
            except Exception: pass
    experiments=[]
    if RESULTS.is_dir():
        for d in [RESULTS]+sorted(x for x in RESULTS.rglob('*') if x.is_dir() and len(x.relative_to(RESULTS).parts)<=2):
            if any((d/n).is_file() and (d/n).stat().st_size>0 for n in TABLE_FILES):
                rel='' if d==RESULTS else str(d.relative_to(RESULTS)); experiments.append({'path':rel,'label':_exp_label(rel)})
    if not any(e['path']==(experiment or '') for e in experiments): experiments.insert(0,{'path':experiment or '','label':_exp_label(experiment or '')})
    return {'files':out[:300],'tables':summaries,'experiment':experiment or '','experiments':experiments}

def _result_path(name):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts: raise HTTPException(400,'Invalid path')
    p=(RESULTS/rel).resolve()
    try:p.relative_to(RESULTS.resolve())
    except ValueError:raise HTTPException(400,'Invalid path')
    if not p.is_file(): raise HTTPException(404,'File not found')
    if p.suffix.lower() not in ('.csv','.md','.png','.json'): raise HTTPException(400,'Unsupported file type')
    return p
@app.get('/api/results/preview')
def result_preview(name:str, rows:int=200):
    p=_result_path(name); ext=p.suffix.lower()
    if ext=='.png': return {'kind':'image','name':name}
    if ext=='.csv':
        try: d=pd.read_csv(p, sep=None, engine='python').replace({np.nan:None})
        except Exception as e: return {'kind':'csv','name':name,'columns':[],'rows':[],'total_rows':0,'note':f'No table to show ({e}). The file may be empty: it is created at the start of a run and filled when the run finishes.'}
        return {'kind':'csv','name':name,'columns':list(d.columns),'rows':d.head(max(1,min(rows,1000))).to_dict('records'),'total_rows':len(d)}
    return {'kind':'text','name':name,'text':p.read_text(encoding='utf-8',errors='replace')[:20000]}
@app.get('/api/results/file')
def result_file(name:str, download:bool=False):
    from fastapi.responses import FileResponse
    p=_result_path(name)
    return FileResponse(p,filename=p.name,content_disposition_type='attachment') if download else FileResponse(p)

@app.get('/api/evaluations/csv')
def evaluation_csv(name:str):
    if Path(name).name != name or not name.endswith('.csv'): raise HTTPException(400,'Select a CSV filename in results root.')
    p=(RESULTS/name).resolve()
    try:p.relative_to(RESULTS.resolve())
    except ValueError:raise HTTPException(400,'Invalid path')
    if not p.is_file():raise HTTPException(404,'Result CSV not found')
    d=pd.read_csv(p, sep=None, engine='python').replace({np.nan:None})
    return {'name':name,'columns':list(d.columns),'rows':d.head(5000).to_dict('records'),'total_rows':len(d)}

def _json_default(v):
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return float(v)
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(pd.Timestamp,datetime)): return v.isoformat()
    return str(v)

class ModelBacktestRequest(BaseModel):
    model_path: str
    features_csv: str = 'features_all.csv'
    start: str | None = None
    end: str | None = None
    initial_capital: float = Field(10000, gt=0, le=1000000000)
    transaction_cost: float = Field(0.0005, ge=0, lt=0.1)

@app.post('/api/models/backtest')
def run_model_backtest(req: ModelBacktestRequest):
    """Run a deterministic historical replay for a compatible saved SB3 PPO model."""
    import pickle
    try:
        from stable_baselines3 import PPO
    except Exception as e: raise HTTPException(500,f'Stable-Baselines3 unavailable in backend environment: {e}')
    rel=Path(req.model_path)
    if rel.is_absolute() or '..' in rel.parts: raise HTTPException(400,'Model path must be relative to project root.')
    mp=(ROOT/rel).resolve()
    try: mp.relative_to(ROOT.resolve())
    except ValueError: raise HTTPException(400,'Invalid model path.')
    if not mp.is_file(): raise HTTPException(404,'Model file not found.')
    meta_path=mp.parent/'metadata.json'; scaler_path=mp.parent/'scaler.pkl'
    if not meta_path.is_file() or not scaler_path.is_file(): raise HTTPException(400,'This model is missing metadata.json or scaler.pkl; safe compatibility cannot be established.')
    try: meta=json.loads(meta_path.read_text(encoding='utf-8')); cols=meta['feature_cols']
    except Exception as e: raise HTTPException(400,f'Invalid model metadata: {e}')
    fp=safe_csv(req.features_csv,RESULTS)
    try:
        f=pd.read_csv(fp, sep=None, engine='python', index_col='Date',parse_dates=True)
        missing=[c for c in cols if c not in f.columns]
        if missing: raise HTTPException(400,f'Model expects features absent from CSV: {missing}. Select a feature CSV generated for this model.')
        f=f[cols].apply(pd.to_numeric,errors='coerce')
        f=f.replace([np.inf,-np.inf],np.nan).dropna()
        if req.start: f=f[f.index>=pd.to_datetime(req.start)]
        if req.end: f=f[f.index<=pd.to_datetime(req.end)]
        if len(f)<2: raise HTTPException(400,'Selected date range has fewer than 2 valid feature rows.')
        raw=pd.read_csv(fp, sep=None, engine='python', index_col='Date',parse_dates=True)
        returns=pd.to_numeric(raw.loc[f.index,'log_return'],errors='coerce').to_numpy(dtype=float)
        with open(scaler_path,'rb') as h: scaler=pickle.load(h)
        scaled = scaler.transform(f)
        x = scaled.to_numpy(dtype=float) if isinstance(scaled, pd.DataFrame) else np.asarray(scaled, dtype=float)
        model=PPO.load(str(mp),device='cpu')
        if tuple(model.observation_space.shape)!=(len(cols)+1,): raise HTTPException(400,f'Model observation shape {model.observation_space.shape} is incompatible with {len(cols)} features plus position.')
        pos=0; equity=float(req.initial_capital); peak=equity; log=[]; actions={0:'HOLD',1:'BUY',2:'SELL'}
        # Same timing convention as the training environment: decision at t earns next row's return.
        for i in range(len(f)-1):
            obs=np.append(x[i],pos).astype(np.float32)
            action,_=model.predict(obs,deterministic=True); requested=int(np.asarray(action).item())
            if requested not in (0,1,2): raise HTTPException(400,'Model emitted an unsupported action; only HOLD/BUY/SELL are supported.')
            previous=pos; pos=1 if requested==1 else (0 if requested==2 else pos); traded=int(pos!=previous)
            cost_log=float(np.log1p(-req.transaction_cost)) if traded else 0.0
            net=pos*float(returns[i+1])+cost_log; equity*=float(np.exp(net)); peak=max(peak,equity)
            log.append({'date':pd.Timestamp(f.index[i]).isoformat(),'pnl_date':pd.Timestamp(f.index[i+1]).isoformat(),'price_return':float(returns[i+1]),'features':{c:float(f.iloc[i][c]) for c in cols},'requested_action':actions[requested],'position':pos,'previous_position':previous,'traded':traded,'transaction_cost_log':cost_log,'reward':float((net)*100),'net_log_return':net,'equity':equity,'drawdown':equity/peak-1})
        eq=np.array([r['equity'] for r in log]); daily=np.array([r['net_log_return'] for r in log]); years=max(len(log)/252,1/252)
        sharpe=float(np.mean(daily)/(np.std(daily,ddof=1)+1e-12)*np.sqrt(252)) if len(daily)>1 else 0
        result={'run_id':uuid.uuid4().hex[:12],'model_path':str(rel),'experiment':meta.get('experiment'),'features_csv':req.features_csv,'start':log[0]['date'],'end':log[-1]['pnl_date'],'initial_capital':req.initial_capital,'final_equity':float(equity),'total_return_pct':float((equity/req.initial_capital-1)*100),'sharpe':sharpe,'max_drawdown_pct':float(min([r['drawdown'] for r in log],default=0)*100),'trades':int(sum(r['traded'] for r in log)),'steps':len(log),'decisions':log,'note':'Historical deterministic replay; uses next-row log return after each decision. Does not simulate intrabar fills or slippage.'}
        # Persist JSON so the run can be reopened after backend restart.
        hist=RESULTS/'platform_runs'; hist.mkdir(parents=True,exist_ok=True); (hist/f"{result['run_id']}.json").write_text(json.dumps(result,default=_json_default),encoding='utf-8')
        return result
    except HTTPException: raise
    except Exception as e: raise HTTPException(400,f'Backtest failed: {e}')
