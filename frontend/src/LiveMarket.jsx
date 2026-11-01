import React,{useEffect,useMemo,useRef,useState}from'react';
import{Radio,Play,RefreshCw,Download,AlertTriangle}from'lucide-react';

/* Display labels only. Every one of these is served by the same Traderbot backend.
   The backend sends the real list (with the asset/timeframe each name is meant for); this is the fallback. */
const MODELS=['PPO_Gold_5M','PPO_BTC_15M','PPO_ETH_1H','TrendFollower_CrudeOil','Momentum_Forex','NaturalGas_Trader','GreenEnergy_Trader','SilverSwing_Metals','IndexTrader_SP500'];
const HISTORY=[100,300,500,1000,2000,5000],VISIBLE=[48,100,200];
const LS='rl_live_market';
const reduce=typeof matchMedia!=='undefined'&&matchMedia('(prefers-reduced-motion: reduce)').matches;

/* decimals: 2 as in the original chart; 5 only for sub-2 prices (FX) so EUR/USD is not rounded to 1.09 */
const fmt=n=>n==null?'—':Number(n).toLocaleString(undefined,{minimumFractionDigits:Math.abs(n)<2?5:2,maximumFractionDigits:Math.abs(n)<2?5:2});
const cur=cat=>cat==='Forex'||cat==='Index'?'':'$';

/* ---------- ORIGINAL CHART (ported verbatim from the Traderbot demo's drawChart) ----------
   Additions that do not change its look when unused: `offset` moves the "sent to model" box,
   `marks` draws BUY/SELL/HOLD signals on the exact candle. With offset=0 and no marks the output is identical to the demo. */
export function buildChartSVG(cs,d,lookback,W,H,offset=0,marks=[]){
  if(!cs.length) return '';
  const padR=62, padT=10, padB=22;
  let hi=Math.max(...cs.map(c=>c.high)), lo=Math.min(...cs.map(c=>c.low));
  if(d){ [d.stop_loss,d.take_profit].forEach(v=>{ if(v!=null){hi=Math.max(hi,v);lo=Math.min(lo,v);} }); }
  const span=(hi-lo)||1; hi+=span*.06; lo-=span*.06;
  const y=v=>padT+(hi-v)/(hi-lo)*(H-padT-padB);
  const cw=(W-padR)/cs.length, bw=Math.max(2,cw*.62);
  let s='<defs><linearGradient id="win" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#0a84ff" stop-opacity=".16"/><stop offset="1" stop-color="#7c4dff" stop-opacity=".04"/></linearGradient></defs>';
  for(let i=0;i<=4;i++){ const v=lo+(hi-lo)*i/4, yy=y(v);
    s+=`<line x1="0" x2="${W-padR}" y1="${yy}" y2="${yy}" stroke="rgba(127,127,160,.18)"/><text x="${W-padR+8}" y="${yy+4}">${fmt(v)}</text>`; }
  const winEnd=cs.length-offset, winStart=Math.max(0,winEnd-lookback), winStop=Math.min(cs.length,winEnd);
  if(winStop>winStart){
    s+=`<rect x="${winStart*cw}" y="${padT}" width="${(winStop-winStart)*cw}" height="${H-padT-padB}" rx="12" fill="url(#win)"/>`;
    s+=`<text x="${winStart*cw+8}" y="${H-6}" style="font-weight:700">sent to model</text>`;
  }
  cs.forEach((c,i)=>{
    const up=c.close>=c.open, col=up?'#00c853':'#ff2d55', x=i*cw+cw/2;
    s+=`<line x1="${x}" x2="${x}" y1="${y(c.high)}" y2="${y(c.low)}" stroke="${col}" stroke-width="1.6" stroke-linecap="round"/>`;
    const top=y(Math.max(c.open,c.close)), h=Math.max(1.5,Math.abs(y(c.open)-y(c.close)));
    s+=`<rect x="${x-bw/2}" y="${top}" width="${bw}" height="${h}" rx="2.5" fill="${col}"/>`;
  });
  const line=(v,col,txt)=>v==null?'':`<line x1="0" x2="${W-padR}" y1="${y(v)}" y2="${y(v)}" stroke="${col}" stroke-width="1.8" stroke-dasharray="6 5"/><rect x="${W-padR+2}" y="${y(v)-10}" width="${padR-4}" height="20" rx="10" fill="${col}"/><text x="${W-padR+(padR-4)/2+2}" y="${y(v)+4}" text-anchor="middle" style="fill:#fff;font-weight:800">${txt}</text>`;
  if(d){ s+=line(d.take_profit,'#00c853','TP')+line(d.stop_loss,'#ff2d55','SL'); }
  /* signal markers on the exact candle the decision was made on */
  marks.forEach(m=>{
    const i=cs.findIndex(c=>c.time===m.time); if(i<0) return;
    const c=cs[i], x=i*cw+cw/2, col=m.action==='BUY'?'#00c853':m.action==='SELL'?'#ff2d55':'#ff9f0a';
    const left=x>W-padR-120, tx=left?x-11:x+11, anc=left?'end':'start';
    const tip=`<title>${m.action} · ${m.model} · ${fmt(m.price)}${m.current?' · latest signal':''}</title>`;
    const txt=yy=>(m.current||m.action!=='HOLD')?`<text x="${tx}" y="${yy}" text-anchor="${anc}" style="fill:${col};font-weight:800;font-size:${m.current?12:10}px">${m.action}${m.current?' @ '+fmt(m.price):''}</text>`:'';
    if(m.current) s+=`<line x1="${x}" x2="${x}" y1="${padT}" y2="${H-padB}" stroke="${col}" stroke-width="1" stroke-dasharray="2 4" opacity=".7"/>`;
    if(m.action==='BUY'){ const yy=y(c.low)+6; s+=`<g>${tip}<path d="M${x} ${yy} l-7 11 h14 z" fill="${col}" stroke="#fff" stroke-width="1"/>${txt(yy+10)}</g>`; }
    else if(m.action==='SELL'){ const yy=y(c.high)-6; s+=`<g>${tip}<path d="M${x} ${yy} l-7 -11 h14 z" fill="${col}" stroke="#fff" stroke-width="1"/>${txt(yy-3)}</g>`; }
    else { const yy=y(c.high)-8; s+=`<g>${tip}<circle cx="${x}" cy="${yy}" r="${m.current?5:3.5}" fill="${col}" stroke="#fff" stroke-width="1"/>${txt(yy-3)}</g>`; }
  });
  return s;
}

function LiveChart({candles,decision,lookback,offset,marks,onPan}){
  const ref=useRef(null),drag=useRef(null);
  useEffect(()=>{
    const svg=ref.current; if(!svg) return;
    const draw=()=>{const W=svg.clientWidth,H=svg.clientHeight;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=buildChartSVG(candles,decision,lookback,W,H,offset,marks)};
    draw(); const ro=new ResizeObserver(draw); ro.observe(svg); return()=>ro.disconnect();
  },[candles,decision,lookback,offset,marks]);
  /* drag sideways to move through time */
  const down=e=>{drag.current={x:e.clientX};e.currentTarget.setPointerCapture?.(e.pointerId)};
  const move=e=>{const d=drag.current;if(!d)return;const per=(ref.current.clientWidth-62)/Math.max(1,candles.length),k=Math.trunc((e.clientX-d.x)/per);if(k){onPan(k);d.x+=k*per}};
  const up=()=>{drag.current=null};
  return <svg ref={ref} className="lm-chart" role="img" aria-label="Candlestick chart (drag sideways to scroll through time)" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}/>;
}

/* Spring from the original: critically damped, velocity carried through retargets */
function spring(from,to,onUpdate,{response=.45,damping=1,velocity=0}={}){
  if(reduce){onUpdate(to);return{stop(){}}}
  const k=(2*Math.PI/response)**2,c=2*damping*Math.sqrt(k);
  let x=from,v=velocity,last=performance.now(),raf,dead=false;
  (function tick(now){
    if(dead) return;
    const dt=Math.min((now-last)/1000,.032);last=now;
    v+=(-k*(x-to)-c*v)*dt;x+=v*dt;
    if(Math.abs(x-to)<.0005&&Math.abs(v)<.0005){onUpdate(to);return}
    onUpdate(x);raf=requestAnimationFrame(tick);
  })(last);
  return{stop(){dead=true;cancelAnimationFrame(raf)}};
}

const when=ts=>new Date(ts).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const day=ts=>new Date(ts).toLocaleDateString([],{day:'numeric',month:'short',year:'numeric'});
const saved=()=>{try{return JSON.parse(localStorage.getItem(LS)||'{}')}catch{return{}}};
const esc=v=>{const t=v==null?'':String(v);return /[",\n]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t};
const toCSV=(rows,cols)=>[cols.join(','),...rows.map(r=>cols.map(c=>esc(r[c])).join(','))].join('\n');
const save=(name,text)=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
const safeName=s=>String(s).replace(/[^A-Za-z0-9_-]/g,'');

export default function LiveMarket({api:API}){
  const init=useMemo(saved,[]);
  const[cfg,setCfg]=useState(null),[symbol,setSymbol]=useState(init.symbol||'BTC-USD'),[tf,setTf]=useState(init.tf||'15m');
  const[model,setModel]=useState(MODELS.includes(init.model)?init.model:MODELS[0]);
  const[histN,setHistN]=useState(init.histN||300),[view,setView]=useState(init.view||48);
  const[candles,setCandles]=useState([]),[avail,setAvail]=useState(null),[status,setStatus]=useState('connecting'),[updated,setUpdated]=useState(null);
  const[err,setErr]=useState(''),[hist,setHist]=useState([]),[showAll,setShowAll]=useState(false),[paper,setPaper]=useState(null),[busy,setBusy]=useState(false);
  const[price,setPrice]=useState(0),[anchor,setAnchor]=useState(null),[viewEnd,setViewEnd]=useState(null),[winSize,setWinSize]=useState(init.winSize??null);
  const priceNow=useRef(0),priceAnim=useRef(null),badge=useRef(null),arc=useRef(null),pct=useRef(null);

  const api=async(path,opt)=>{const r=await fetch(API+'/api/tradebot'+path,{headers:{'Content-Type':'application/json'},...opt});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.detail||r.statusText);return d};
  const asset=cfg?.assets.find(a=>a.symbol===symbol);
  const C=cur(asset?.category);
  const modelList=cfg?.models||MODELS.map(name=>({name}));

  useEffect(()=>{try{localStorage.setItem(LS,JSON.stringify({symbol,tf,model,winSize,histN,view}))}catch{}},[symbol,tf,model,winSize,histN,view]);
  useEffect(()=>{api('/config').then(setCfg).catch(e=>{setStatus('error');setErr('Traderbot API not reachable: '+e.message)})},[]);

  /* chart data: backend-driven, refreshed every 60 s; stale responses are ignored */
  useEffect(()=>{
    let dead=false;setCandles([]);setAnchor(null);setViewEnd(null);setStatus('connecting');
    const load=()=>api(`/candles?symbol=${encodeURIComponent(symbol)}&interval=${tf}&limit=${histN}`).then(r=>{if(dead)return;setCandles(r.candles);setAvail(r.available);setStatus('live');setUpdated(new Date());setErr('')}).catch(e=>{if(dead)return;setStatus('error');setErr(e.message)});
    load();const t=setInterval(load,60000);return()=>{dead=true;clearInterval(t)};
  },[symbol,tf,histN]);

  const loadRecords=async()=>{
    try{
      const q=showAll?'':'&model_name='+encodeURIComponent(model);
      const[h,p]=await Promise.all([api('/history?limit=200'+q),api('/paper?model_name='+encodeURIComponent(model))]);
      setHist(h.decisions);setPaper(p);
    }catch(e){setErr(e.message)}
  };
  useEffect(()=>{setHist([]);setPaper(null);loadRecords();const t=setInterval(loadRecords,60000);return()=>clearInterval(t)},[model,showAll]);

  /* ---- window (what the model sees) and view (what you see) ---- */
  const n=candles.length;
  const lb=Math.min(40,Math.max(3,Number(winSize??cfg?.lookback??12)||3));
  const V=Math.min(view,n),maxPan=Math.max(0,n-V);
  const vi=viewEnd?candles.findIndex(c=>c.time===viewEnd):-1;
  const pan=vi>=0?Math.min(n-1-vi,maxPan):0;
  const vEnd=n-pan,vStart=Math.max(0,vEnd-V);
  const vis=useMemo(()=>candles.slice(vStart,vEnd),[candles,vStart,vEnd]);
  const panBy=k=>{const p=Math.min(maxPan,Math.max(0,pan+k));setViewEnd(p?candles[n-1-p].time:null)};
  const aIdx=anchor?candles.findIndex(c=>c.time===anchor):-1;
  const winAnchor=aIdx>=lb-1?anchor:null;
  const wEndIdx=winAnchor?aIdx:n-1;
  const offset=n-1-wEndIdx,maxOff=Math.max(0,n-lb);
  const chartOffset=(vEnd-1)-wEndIdx;
  const goBack=o=>{
    const k=Math.min(maxOff,Math.max(0,o)),idx=n-1-k;setAnchor(k?candles[idx].time:null);
    if(idx>vEnd-1||idx-lb+1<vStart){const ve=Math.min(n-1,Math.max(V-1,idx+Math.floor(V/6)));setViewEnd(ve===n-1?null:candles[ve].time)}
  };

  const decision=useMemo(()=>hist.find(r=>r.model_name===model&&r.symbol===symbol&&r.interval===tf&&(r.window_end||null)===winAnchor&&(r.window_size||cfg?.lookback||12)===lb)||null,[hist,model,symbol,tf,winAnchor,lb,cfg]);
  const marks=useMemo(()=>hist.filter(r=>r.model_name===model&&r.symbol===symbol&&r.interval===tf).map(r=>({time:r.candle_time||r.window_end,action:r.action,price:r.last_close,model:r.model_name,current:r.id===decision?.id})).filter(m=>m.time),[hist,model,symbol,tf,decision]);

  /* model <-> market rule (the backend enforces the same rule) */
  const rule=modelList.find(m=>m.name===model);
  const fits=m=>(!m.assets||m.assets.includes(symbol))&&(!m.timeframes||m.timeframes.includes(tf));
  const mismatch=!!rule&&!fits(rule);
  const compatible=modelList.filter(fits).map(m=>m.name);
  const builtFor=rule?(rule.assets||[]).map(s=>cfg?.assets.find(a=>a.symbol===s)?.label||s).join(' / ')+(rule.timeframes?` on ${rule.timeframes.join(', ')}`:''):'';

  const last=candles[n-1],first=vis[0]||candles[0];
  const delta=last?(last.close-first.open)/first.open*100:null;

  useEffect(()=>{if(!last)return;priceAnim.current?.stop();priceAnim.current=spring(priceNow.current,last.close,x=>{priceNow.current=x;setPrice(x)},{response:.6});return()=>priceAnim.current?.stop()},[last?.close,symbol,tf]);
  useEffect(()=>{
    if(!decision||!arc.current)return;const Cc=2*Math.PI*32;
    const a=spring(.82,1,s=>{if(badge.current)badge.current.style.transform=`scale(${s})`},{response:.4,damping:.72});
    const b=spring(0,decision.confidence,p=>{if(arc.current)arc.current.style.strokeDashoffset=Cc*(1-p);if(pct.current)pct.current.textContent=Math.round(p*100)+'%'},{response:.7});
    return()=>{a.stop();b.stop()};
  },[decision?.id]);

  async function analyze(){
    setBusy(true);setErr('');
    try{await api('/decide',{method:'POST',body:JSON.stringify({symbol,interval:tf,model_name:model,window_end:winAnchor,window_size:lb})});await loadRecords()}
    catch(e){setErr(e.message)}
    setBusy(false);
  }
  const dlCandles=()=>save(`${safeName(symbol)}_${tf}_${n}candles.csv`,toCSV(candles,['time','open','high','low','close','volume']));
  const dlHistory=()=>save(`decisions_${showAll?'all_models':safeName(model)}.csv`,toCSV(hist,['ts','model_name','asset','symbol','interval','candle_time','window_end','window_size','action','confidence','last_close','stop_loss','take_profit','reasoning','engine']));

  const dot=status==='live'?'green':status==='error'?'red':'amber';
  const Cc=2*Math.PI*32;
  const tfLabel=cfg?.timeframes.find(t=>t.id===tf)?.label||tf;
  return <>
    <div className="head"><div className="eyebrow">LIVE MARKET / SIGNALS &amp; PAPER TRADING</div><h1>Live market</h1>
      <p>Real market candles with AI trade signals. Signal-only and paper trading: no real orders are ever placed.</p></div>

    <section className="card lm-toolbar">
      <label>Asset<select className="input" value={symbol} onChange={e=>setSymbol(e.target.value)}>
        {cfg?(['Crypto','Energy','Metals','Forex','Green Energy','Index']).map(cat=><optgroup key={cat} label={cat}>{cfg.assets.filter(a=>a.category===cat).map(a=><option key={a.symbol} value={a.symbol}>{a.label}</option>)}</optgroup>):<option value={symbol}>{symbol}</option>}
      </select></label>
      <label>Model<select className="input" value={model} onChange={e=>setModel(e.target.value)}>
        <optgroup label={`Matches ${asset?.label||symbol} · ${tf}`}>{modelList.filter(fits).map(m=><option key={m.name}>{m.name}</option>)}</optgroup>
        <optgroup label="Other models (will not run on this market)">{modelList.filter(m=>!fits(m)).map(m=><option key={m.name}>{m.name}</option>)}</optgroup>
      </select></label>
      <div className="lm-tf"><span>Timeframe</span><div>{(cfg?.timeframes||[{id:tf,label:tf}]).map(t=><button key={t.id} className={'lm-chip'+(t.id===tf?' on':'')} title={t.label} aria-pressed={t.id===tf} onClick={()=>setTf(t.id)}>{t.id}</button>)}</div></div>
      <div className="lm-status"><i className={dot}/><div><b>{status==='live'?'Market data live':status==='error'?'Market data error':'Connecting…'}</b>
        <small>{updated?'Updated '+updated.toLocaleTimeString():'Waiting for first candles'} · Traderbot API {cfg?(cfg.engine_ready?'ready':'not configured'):'…'}</small></div></div>
    </section>

    {err&&<div className="alert error">{err}<button onClick={()=>setErr('')}>×</button></div>}

    <div className="lm-grid">
      <section className="card">
        <div className="lm-model"><Radio size={13}/> Active model <b>{model}</b></div>
        <div className="lm-price"><div className="lm-pv">{last?C+fmt(price):'—'}</div>{delta!=null&&<div className={'lm-delta '+(delta>=0?'up':'down')}>{(delta>=0?'+':'')+delta.toFixed(2)}%</div>}</div>
        <div className="lm-meta">{asset?.label||symbol} · {tfLabel} candles{vis.length?` · showing ${vis.length} of ${n.toLocaleString()} loaded`:''}{pan?` · ${pan} candles back from latest`:''}</div>
        <div className="lm-chartbox"><LiveChart candles={vis} decision={decision} lookback={lb} offset={chartOffset} marks={marks} onPan={panBy}/>{!n&&<div className="lm-loading">{status==='error'?'No data':'Loading candles…'}</div>}</div>

        <div className="lm-slider"><span className="lm-lbl">Scroll chart</span>
          <button className="lm-chip" title="Show earlier candles" disabled={pan>=maxPan} onClick={()=>panBy(Math.max(1,Math.floor(V/4)))}>◀</button>
          <input type="range" aria-label="Chart position" min="0" max={maxPan} value={maxPan-pan} disabled={maxPan<1} onChange={e=>panBy(pan-(maxPan-Number(e.target.value)))}/>
          <button className="lm-chip" title="Show later candles" disabled={pan<=0} onClick={()=>panBy(-Math.max(1,Math.floor(V/4)))}>▶</button>
          <button className="lm-chip" disabled={!pan} onClick={()=>setViewEnd(null)}>Now</button>
          <span>{vis.length?`${when(vis[0].time)} → ${when(vis[vis.length-1].time)} · drag the chart sideways too`:''}</span></div>

        <div className="lm-slider"><span className="lm-lbl">Model window</span>
          <button className="lm-chip" title="Move window back one candle" disabled={offset>=maxOff} onClick={()=>goBack(offset+1)}>◀</button>
          <input type="range" aria-label="Window position" min="0" max={maxOff} value={maxOff-offset} disabled={maxOff<1} onChange={e=>goBack(maxOff-Number(e.target.value))}/>
          <button className="lm-chip" title="Move window forward one candle" disabled={offset<=0} onClick={()=>goBack(offset-1)}>▶</button>
          <button className="lm-chip" disabled={!offset} onClick={()=>goBack(0)}>Latest</button>
          <label className="lm-size">Size <input type="number" min="3" max="40" value={winSize??(cfg?.lookback||12)} onChange={e=>setWinSize(e.target.value===''?'':Number(e.target.value))} aria-label="Window size in candles"/> candles</label>
          <span>{offset?`Replay: window ends ${offset} candle${offset>1?'s':''} back · ${when(candles[n-1-offset].time)}`:'Window: latest candles (live)'}</span></div>

        <div className="lm-slider lm-data">
          <label className="lm-size">History <select className="input" value={histN} onChange={e=>setHistN(Number(e.target.value))}>{HISTORY.map(h=><option key={h} value={h}>{h.toLocaleString()} candles</option>)}</select></label>
          <label className="lm-size">Visible <select className="input" value={view} onChange={e=>setView(Number(e.target.value))}>{VISIBLE.map(v=><option key={v} value={v}>{v} candles</option>)}</select></label>
          <button className="btn secondary" disabled={!n} onClick={dlCandles}><Download size={13}/> Download candles CSV</button>
        </div>
        {avail&&<p className="lm-avail">Yahoo Finance has <b>{avail.count.toLocaleString()}</b> {tf} candles for {asset?.label||symbol}, from <b>{day(avail.first)}</b> to <b>{day(avail.last)}</b>. {avail.limit_note}.{histN>avail.count?` You asked for ${histN.toLocaleString()}, so only ${avail.count.toLocaleString()} are available.`:''} Loaded {n.toLocaleString()}.</p>}
      </section>

      <section className="card">
        <div className="card-head"><div><h3>AI decision</h3><p>Active model: {model}</p></div></div>
        {mismatch?<div className="lm-warn"><AlertTriangle size={16}/><div><b>{model} does not match this market</b>
          <p>{model} is built for {builtFor}, not {asset?.label||symbol} on {tf}.</p>
          {compatible.length?<><p>Select an appropriate model:</p><div className="lm-sugg">{compatible.map(m=><button key={m} className="lm-chip" onClick={()=>setModel(m)}>{m}</button>)}</div></>:<p>No model is set up for {asset?.label||symbol} on {tf}. Try another timeframe or asset.</p>}</div></div>
        :decision?<>
          <div className="lm-verdict"><div className={'lm-badge '+decision.action} ref={badge}>{decision.action}</div>
            <div className="lm-ring"><svg viewBox="0 0 76 76"><circle cx="38" cy="38" r="32" fill="none" stroke="rgba(127,127,160,.2)" strokeWidth="8"/>
              <circle ref={arc} cx="38" cy="38" r="32" fill="none" stroke="url(#lmg)" strokeWidth="8" strokeLinecap="round" strokeDasharray={Cc} strokeDashoffset={Cc}/>
              <defs><linearGradient id="lmg"><stop offset="0" stopColor="#0a84ff"/><stop offset="1" stopColor="#ff4fa3"/></linearGradient></defs></svg><b ref={pct}>0%</b></div></div>
          <div className="lm-levels"><div><small>Signal price</small><b>{C+fmt(decision.last_close)}</b></div><div><small>Signal candle</small><b>{when(decision.candle_time||decision.window_end||decision.ts)}</b></div>
            <div><small>Stop loss</small><b>{decision.stop_loss==null?'—':C+fmt(decision.stop_loss)}</b></div><div><small>Take profit</small><b>{decision.take_profit==null?'—':C+fmt(decision.take_profit)}</b></div></div>
          <p className="lm-stamp">{model} · {asset?.label||symbol} · {tf} · decided {when(decision.ts)}{decision.window_end&&<> · <b>Replay</b>, window ending {when(decision.window_end)}</>} · {decision.window_size||lb}-candle window</p>
          {decision.window_end&&last&&<p className="lm-since">Since that window: <b className={last.close>=decision.last_close?'lm-up':'lm-down'}>{((last.close-decision.last_close)/decision.last_close*100).toFixed(2)}%</b> (latest close {C+fmt(last.close)})</p>}
        </>:<div className="empty lm-emptydec"><b>No decision yet</b><p>Run {model} on the latest {tf} candles for {asset?.label||symbol}.</p></div>}
        <button className="btn primary full" disabled={busy||!n||mismatch||(cfg&&!cfg.engine_ready)} onClick={analyze}><Play size={14}/>{busy?'Thinking…':offset?'Analyze this window':'Analyze now'}</button>
        {cfg&&!cfg.engine_ready&&<p className="note">Traderbot engine not configured. Set GROQ_API_KEY and GROQ_MODEL in backend/.env and restart the API.</p>}
      </section>
    </div>

    <section className="card">
      <div className="card-head"><div><h3>Paper trading · {model}</h3><p>{paper?`Long-only, ${C||''}${fmt(paper.notional)} per trade (${C?'USD':'quote currency'}). BUY opens, SELL closes. Simulated from real candle prices; replays are not counted.`:'Loading…'}</p></div></div>
      <div className="metrics lm-metrics">{[['Realized P&L',paper&&fmt(paper.realized)],['Unrealized P&L',paper&&fmt(paper.unrealized)],['Closed trades',paper&&paper.closed_trades],['Win rate',paper&&(paper.win_rate==null?'—':paper.win_rate.toFixed(0)+'%')]].map(([k,v])=><div className="metric" key={k}><span>{k}</span><b>{v??'—'}</b></div>)}</div>
      <div className="tablebox"><table><thead><tr>{['Open position','Timeframe','Opened','Entry','Last','Unrealized'].map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>
        {paper?.open_positions.length?paper.open_positions.map(p=><tr key={p.symbol}><td>{p.asset}</td><td>{p.interval}</td><td>{when(p.opened)}</td><td>{fmt(p.entry)}</td><td>{fmt(p.last)}</td><td className={p.unrealized>=0?'lm-up':'lm-down'}>{fmt(p.unrealized)}</td></tr>):<tr><td colSpan="6">No open positions for {model}.</td></tr>}
      </tbody></table></div>
      <p className="lm-avail">This table lists the paper positions {model} is holding right now. A BUY signal opens one; the next SELL signal for the same asset closes it and moves the profit or loss into Realized P&amp;L.</p>
    </section>

    <section className="card" style={{marginTop:15}}>
      <div className="card-head lm-histhead"><div><h3>Decision history · {showAll?'all models':model}</h3><p>Each record keeps the model name it was created under.</p></div>
        <label className="lm-check"><input type="checkbox" checked={showAll} onChange={e=>setShowAll(e.target.checked)}/> Show all models</label>
        <button className="btn secondary" disabled={!hist.length} onClick={dlHistory}><Download size={13}/> CSV</button>
        <button className="btn secondary" onClick={loadRecords}><RefreshCw size={13}/> Refresh</button></div>
      <div className="tablebox"><table><thead><tr>{['Time','Model','Asset','TF','Window','Action','Conf.','Price','Reasoning'].map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>
        {hist.length?hist.map(h=><tr key={h.id}><td>{when(h.ts)}</td><td>{h.model_name}</td><td>{h.asset}</td><td>{h.interval}</td><td>{(h.window_end?'Replay · '+when(h.window_end):'Live')+(h.window_size?' · '+h.window_size+' candles':'')}</td><td><span className={'lm-tag '+h.action}>{h.action}</span></td><td>{Math.round(h.confidence*100)}%</td><td>{fmt(h.last_close)}</td><td className="lm-rsn" title={h.reasoning}>{h.reasoning}</td></tr>):<tr><td colSpan="9">No decisions yet{showAll?'':' for '+model}.</td></tr>}
      </tbody></table></div>
    </section>
  </>;
}
