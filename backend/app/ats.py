"""Persisted background time-series analysis for RL Trader."""
import base64, json, threading, uuid, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix='/api/ats', tags=['Time-Series Analysis'])
_jobs = {}
_lock = threading.Lock()

class UploadRequest(BaseModel):
    filename: str
    content_base64: str
    date_column: str | None = None
    target_column: str | None = None
    frequency: str = 'auto'
    seasonal_period: int = Field(0, ge=0, le=10000)

class JobRequest(BaseModel):
    report_id: str


def _dirs():
    # main.py defines RESULTS from .env; resolve lazily to avoid import cycles.
    from .main import RESULTS
    reports = RESULTS / 'ats_reports'; uploads = RESULTS / 'ats_uploads'
    reports.mkdir(parents=True, exist_ok=True); uploads.mkdir(parents=True, exist_ok=True)
    return reports, uploads

def _now(): return datetime.now(timezone.utc).isoformat()
def _json_default(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (pd.Timestamp, datetime)): return v.isoformat()
    if pd.isna(v): return None
    return str(v)
def _save(report):
    reports, _ = _dirs(); (reports / f"{report['report_id']}.json").write_text(json.dumps(report, default=_json_default, indent=2), encoding='utf-8')
def _read(report_id):
    reports, _ = _dirs(); p=reports/f'{report_id}.json'
    if not p.is_file(): raise HTTPException(404, 'Saved analysis report not found')
    return json.loads(p.read_text(encoding='utf-8'))

@router.post('/upload')
def upload(req: UploadRequest):
    if not req.filename.lower().endswith('.csv'): raise HTTPException(400, 'Please upload a CSV file.')
    try:
        raw=base64.b64decode(req.content_base64, validate=True)
        if len(raw)>30*1024*1024: raise HTTPException(413, 'CSV must be 30 MB or smaller.')
        from io import BytesIO
        df=pd.read_csv(BytesIO(raw), sep=None, engine='python')
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, f'Could not parse CSV: {e}')
    if df.empty or len(df.columns)<2: raise HTTPException(400, 'CSV needs rows and at least two columns.')
    date_col=req.date_column or next((c for c in df if c.lower().strip() in ('date','datetime','timestamp','time')), None)
    target=req.target_column or next((c for c in df if c.lower().strip() in ('close','price','value','y')), None)
    if not date_col or date_col not in df: raise HTTPException(400, 'Select a valid date/time column.')
    if not target or target not in df: raise HTTPException(400, 'Select a valid numeric target column (e.g. Close).')
    df[date_col]=pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_convert(None)
    df[target]=pd.to_numeric(df[target], errors='coerce')
    df=df.dropna(subset=[date_col,target]).sort_values(date_col).drop_duplicates(date_col)
    if len(df)<40: raise HTTPException(400, 'At least 40 valid observations are recommended for this analysis.')
    rid=uuid.uuid4().hex[:12]; reports, uploads=_dirs(); stored=uploads/f'{rid}.csv'; df.to_csv(stored,index=False)
    report={'report_id':rid,'filename':Path(req.filename).name,'created_at':_now(),'updated_at':_now(),'status':'queued','stage':'Queued','progress':0,'error':None,'date_column':date_col,'target_column':target,'frequency_requested':req.frequency,'seasonal_period_requested':req.seasonal_period,'rows':int(len(df)),'columns':list(df.columns),'stored_csv':str(stored),'stages':[],'results':None}
    _save(report)
    with _lock: _jobs[rid]=report
    threading.Thread(target=_run,args=(rid,),daemon=True).start()
    return {'report_id':rid,'status':'queued'}

@router.get('/reports')
def reports_list():
    reports,_=_dirs(); out=[]
    for p in reports.glob('*.json'):
        try:
            r=json.loads(p.read_text(encoding='utf-8'))
            out.append({k:r.get(k) for k in ('report_id','filename','created_at','updated_at','status','stage','progress','error','rows','target_column')})
        except Exception: pass
    return {'reports':sorted(out,key=lambda x:x.get('created_at') or '',reverse=True)}

@router.get('/reports/{report_id}')
def report_detail(report_id:str): return _read(report_id)

@router.post('/reports/{report_id}/rerun')
def rerun(report_id:str):
    old=_read(report_id)
    if old.get('status') in ('queued','running'): return {'report_id':report_id,'status':old['status']}
    with _lock: _jobs[report_id]=old
    threading.Thread(target=_run,args=(report_id,),daemon=True).start()
    return {'report_id':report_id,'status':'queued'}

def _run(rid):
    r=_read(rid); r.update(status='running',stage='Preparing data',progress=2,error=None,stages=[],updated_at=_now()); _save(r)
    def stage(name,pct,status='completed',detail=None):
        r['stage']=name; r['progress']=pct; r['updated_at']=_now(); r['stages'].append({'name':name,'status':status,'detail':detail,'completed_at':_now() if status=='completed' else None}); _save(r)
    try:
        df=pd.read_csv(r['stored_csv']); date=r['date_column']; target=r['target_column']; df[date]=pd.to_datetime(df[date],errors='coerce'); df[target]=pd.to_numeric(df[target],errors='coerce'); df=df.dropna(subset=[date,target]).sort_values(date).drop_duplicates(date).reset_index(drop=True)
        y=df[target].astype(float); dates=df[date]
        if len(y)<40: raise ValueError('Not enough valid observations after cleaning.')
        freq=r['frequency_requested']; inferred=pd.infer_freq(pd.DatetimeIndex(dates)) if len(dates)>3 else None
        stage('Data quality & profile',10,detail='Parsed timestamps, sorted rows and checked duplicates/missing values.')
        clean=df[[date,target]].copy(); clean[target]=y
        returns=y.pct_change().replace([np.inf,-np.inf],np.nan).dropna(); logret=np.log(y.where(y>0)).diff().replace([np.inf,-np.inf],np.nan).dropna()
        desc=y.describe().to_dict(); miss=int(df.isna().sum().sum()); dup=int(df[date].duplicated().sum())
        chart_step=max(1,len(df)//500); series=[{'date':dates.iloc[i].strftime('%Y-%m-%d %H:%M:%S'),'value':float(y.iloc[i])} for i in range(0,len(df),chart_step)]
        returns_data=[{'date':dates.iloc[i+1].strftime('%Y-%m-%d'),'value':float(returns.iloc[i])} for i in range(0,len(returns),max(1,len(returns)//400)) if np.isfinite(returns.iloc[i])]
        stage('Exploratory analysis',22,detail='Computed descriptive statistics, returns, rolling statistics and chart series.')
        from statsmodels.tsa.stattools import adfuller, acf, pacf
        adf={}
        for label,series0 in [('level',y),('first_difference',y.diff().dropna()),('returns',returns)]:
            try:
                a=adfuller(series0.dropna().values,maxlag=None,autolag='AIC'); adf[label]={'statistic':float(a[0]),'p_value':float(a[1]),'used_lag':int(a[2]),'observations':int(a[3]),'critical_values':{k:float(v) for k,v in a[4].items()},'stationary_at_5pct':bool(a[1]<.05)}
            except Exception as e: adf[label]={'error':str(e)}
        stage('Stationarity & transformations',35,detail='ADF applied to level, first difference and simple returns where calculable.')
        n=len(y); maxlag=max(1,min(40,n//5)); acfvals=acf(y.diff().dropna(),nlags=maxlag,fft=True); pacfvals=pacf(y.diff().dropna(),nlags=min(maxlag,max(1,len(y.diff().dropna())//2-1)),method='ywm')
        lagplot=[{'lag':int(i),'acf':float(acfvals[i]),'pacf':float(pacfvals[i]) if i<len(pacfvals) else None} for i in range(1,min(maxlag,30)+1)]
        period=int(r.get('seasonal_period_requested') or 0)
        if period==0:
            f=str(freq).lower()
            period=7 if f in ('daily','d','day') else (12 if f in ('monthly','m','month') else 0)
        decomposition=None
        if period>=2 and len(y)>=2*period:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                dec=seasonal_decompose(y,model='additive',period=period,extrapolate_trend='freq')
                decomposition={'period':period,'points':[{'date':dates.iloc[i].strftime('%Y-%m-%d'),'trend':float(dec.trend.iloc[i]),'seasonal':float(dec.seasonal.iloc[i]),'residual':float(dec.resid.iloc[i])} for i in range(0,n,max(1,n//350)) if pd.notna(dec.trend.iloc[i]) and pd.notna(dec.resid.iloc[i])]}
            except Exception as e: decomposition={'error':str(e)}
        stage('Temporal structure & lags',45,detail='Calculated ACF/PACF of first differences; decomposition attempted only when a period is specified or inferred from a known frequency.')
        # Chronological holdout: fit on earlier observations, evaluate on final 20%.
        split=max(30,int(n*.8)); split=min(split,n-5); train=y.iloc[:split]; test=y.iloc[split:]; test_dates=dates.iloc[split:]
        candidates=[]
        def add_candidate(name, fitted_train, forecast, params=None, aic=None, bic=None):
            fc=np.asarray(forecast,dtype=float).reshape(-1)[:len(test)]
            if len(fc)<len(test): fc=np.pad(fc,(0,len(test)-len(fc)),constant_values=np.nan)
            valid=np.isfinite(fc)&np.isfinite(test.values)
            err=test.values[valid]-fc[valid]
            candidates.append({'name':name,'parameters':params or {},'aic':float(aic) if aic is not None and np.isfinite(aic) else None,'bic':float(bic) if bic is not None and np.isfinite(bic) else None,'mae':float(np.mean(np.abs(err))) if len(err) else None,'rmse':float(np.sqrt(np.mean(err**2))) if len(err) else None,'forecast':fc.tolist(),'train_residuals':np.asarray(fitted_train,dtype=float).tolist()})
        # Baseline: last observation carried forward.
        add_candidate('Naive / random-walk baseline',np.diff(train.values),np.repeat(train.iloc[-1],len(test)),{'type':'baseline'})
        stage('Baseline forecast',53,detail='Established a last-value baseline on a chronological holdout.')
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
            try:
                m=SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
                add_candidate('Simple Exponential Smoothing',m.resid,m.forecast(len(test)),{'type':'SES'},m.aic,m.bic)
            except Exception: pass
            try:
                m=ExponentialSmoothing(train,trend='add',initialization_method='estimated').fit(optimized=True)
                add_candidate('Holt trend',m.resid,m.forecast(len(test)),{'type':'Holt additive trend'},m.aic,m.bic)
            except Exception: pass
        except Exception: pass
        stage('Exponential smoothing candidates',62,detail='Tried suitable non-seasonal smoothing candidates; failed fits are skipped rather than fabricated.')
        try:
            from statsmodels.tsa.arima.model import ARIMA
            for order in [(1,0,0),(0,1,1),(1,1,0),(1,1,1),(2,1,1),(2,0,1)]:
                try:
                    m=ARIMA(train,order=order).fit()
                    add_candidate(f'ARIMA{order}',m.resid,m.forecast(len(test)),{'order':order},m.aic,m.bic)
                except Exception: continue
        except Exception: pass
        stage('ARIMA model comparison',73,detail='Fitted a bounded set of candidate ARIMA models; AIC/BIC and holdout metrics are recorded when available.')
        usable=[c for c in candidates if c.get('rmse') is not None and np.isfinite(c['rmse'])]
        if not usable: raise ValueError('No forecasting candidate could be evaluated. Check that the target series is numeric and has enough variation.')
        # Prefer holdout RMSE; AIC/BIC are descriptive fit criteria, not the sole selection rule.
        best=min(usable,key=lambda c:c['rmse']); pred=best['forecast']; residual=[float(a-b) for a,b in zip(test.values,pred) if np.isfinite(b)]
        # Fit selected family on training data and diagnostic residuals from that fit.
        fitted_resid=None; fitted_model=None
        try:
            nm=best['name']
            if nm.startswith('ARIMA'):
                import ast
                order=best['parameters']['order']; fitted_model=ARIMA(train,order=tuple(order)).fit(); fitted_resid=np.asarray(fitted_model.resid,dtype=float)
            elif nm=='Holt trend':
                fitted_model=ExponentialSmoothing(train,trend='add',initialization_method='estimated').fit(optimized=True); fitted_resid=np.asarray(fitted_model.resid,dtype=float)
            elif nm=='Simple Exponential Smoothing':
                fitted_model=SimpleExpSmoothing(train,initialization_method='estimated').fit(optimized=True); fitted_resid=np.asarray(fitted_model.resid,dtype=float)
        except Exception: pass
        if fitted_resid is None: fitted_resid=np.diff(train.values)
        stage('Forecast & model selection',82,detail=f"Selected {best['name']} by lowest chronological holdout RMSE among successfully evaluated candidates.")
        resid_clean=pd.Series(fitted_resid).replace([np.inf,-np.inf],np.nan).dropna(); lb=[]
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lags=sorted(set([min(10,max(1,len(resid_clean)//5)),min(20,max(1,len(resid_clean)//3))]))
            lbdf=acorr_ljungbox(resid_clean,lags=lags,return_df=True)
            lb=[{'lag':int(i),'statistic':float(row['lb_stat']),'p_value':float(row['lb_pvalue']),'no_autocorrelation_not_rejected_at_5pct':bool(row['lb_pvalue']>=.05)} for i,row in lbdf.iterrows()]
        except Exception as e: lb=[{'error':str(e)}]
        residual_sample=[{'index':int(i),'residual':float(resid_clean.iloc[i])} for i in range(0,len(resid_clean),max(1,len(resid_clean)//350)) if np.isfinite(resid_clean.iloc[i])]
        stage('Residual diagnostics',90,detail='Ran Ljung–Box on fitted training residuals where available. A high p-value means failure to reject autocorrelation, not proof of all white-noise assumptions.')
        forecast_points=[]
        for i,(dt,actual,fc) in enumerate(zip(test_dates,pred,test.values)):
            if np.isfinite(fc): forecast_points.append({'date':dt.strftime('%Y-%m-%d %H:%M:%S'),'forecast':float(fc),'actual':float(actual)})
        metrics={'mae':best['mae'],'rmse':best['rmse'],'holdout_rows':len(test),'train_rows':len(train),'holdout_fraction':len(test)/n}
        conclusion=[]
        conclusion.append(f"Selected {best['name']} because it had the lowest holdout RMSE ({best['rmse']:.6g}) among {len(usable)} evaluated candidates.")
        if adf.get('level',{}).get('stationary_at_5pct') is False: conclusion.append('The ADF test did not find sufficient evidence of stationarity in the level series at the 5% significance level; differencing/returns may be more appropriate for some models.')
        if lb and 'p_value' in lb[0]:
            failed=[x for x in lb if x['p_value']<.05]
            conclusion.append('Ljung–Box detected residual autocorrelation at at least one tested lag; revisit model specification.' if failed else 'Ljung–Box did not reject the no-autocorrelation null at the tested lags; this alone does not prove residuals are fully white noise.')
        conclusion.append('Forecast scores are from a chronological holdout; results are descriptive and do not imply trading profitability.')
        results={'profile':{'rows':int(n),'columns':list(df.columns),'date_min':dates.min().isoformat(),'date_max':dates.max().isoformat(),'inferred_frequency':inferred,'missing_values_after_required_column_cleaning':int(df[[date,target]].isna().sum().sum()),'duplicate_timestamps_after_deduplication':0,'target':target,'date_column':date,'descriptive_statistics':{k:float(v) for k,v in desc.items() if np.isfinite(v)}},'charts':{'series':series,'returns':returns_data,'acf_pacf':lagplot,'decomposition':decomposition,'forecast':forecast_points,'residuals':residual_sample},'stationarity':adf,'models':[{k:v for k,v in c.items() if k!='train_residuals'} for c in sorted(usable,key=lambda c:c['rmse'])],'selected_model':{k:v for k,v in best.items() if k!='train_residuals'},'forecast_metrics':metrics,'ljung_box':lb,'conclusion':conclusion,'caveats':['Model selection is a small candidate search, not exhaustive Auto-ARIMA.','Seasonal ARIMA is not automatically fitted; provide a defensible seasonal period and sufficient history before adding it.','Ljung–Box is a residual autocorrelation diagnostic, not proof of all white-noise properties.','Forecast accuracy is not equivalent to trading profitability.']}
        r.update(status='completed',stage='Complete',progress=100,results=results,updated_at=_now()); _save(r)
    except Exception as e:
        r.update(status='failed',stage='Failed',error=str(e),updated_at=_now()); _save(r)
