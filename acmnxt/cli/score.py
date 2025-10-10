from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from acmnxt.io.loaders import read_table, ensure_datetime_index
from acmnxt.core.h1_ar1 import score_h1
from acmnxt.core.h2_pca import score_h2
from acmnxt.core.fusion import fuse_scores
from acmnxt.core.events import build_events
from joblib import load
import json

def main()->int:
    ap=argparse.ArgumentParser(description='ACMnxt score-only')
    ap.add_argument('--csv',required=True); ap.add_argument('--equip',required=True); ap.add_argument('--art-dir',required=True)
    a=ap.parse_args(); root=Path(a.art_dir)/("".join(ch if ch.isalnum() else '_' for ch in a.equip))
    import pandas as pd
    df_raw=read_table(a.csv); df=ensure_datetime_index(df_raw)
    df=df.apply(pd.to_numeric, errors='coerce')
    df=df.resample('1min').mean().interpolate(limit_direction='both')
    df_imp=df.ffill().bfill()
    h1=score_h1(df_imp)
    h2=score_h2(df_imp, art_dir=root)
    try:
        km=load(root/'acm_regimes.joblib'); reg=pd.Series(km.predict(df_imp.select_dtypes(include=[float,int]).values), index=df_imp.index, name='Regime')
    except Exception:
        reg=None
    fused=fuse_scores({'H1_Forecast':h1,'H2_Recon':h2})
    scores=df.copy(); scores['FusedScore']=fused; scores['H1_Forecast']=h1; scores['H2_Recon']=h2
    if reg is not None: scores['Regime']=reg
    scores.reset_index(names=['Ts']).to_csv(root/'scores.csv', index=False)
    ev=build_events(fused)
    rows=[]
    if not ev.empty:
        for _,r in ev.iterrows():
            t0=int(scores.index.get_indexer([r['Start']], method='nearest')[0]) if isinstance(scores.index, pd.DatetimeIndex) else 0
            t1=int(scores.index.get_indexer([r['End']], method='nearest')[0]) if isinstance(scores.index, pd.DatetimeIndex) else 0
            rows.append({'event_id': int(r['id']), 't0': t0, 't1': t1})
    (root/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in rows), encoding='utf-8')
    print('[score] wrote', root/'scores.csv', 'and', root/'events.jsonl')
    return 0

if __name__=='__main__': raise SystemExit(main())
