from __future__ import annotations
import argparse
from pathlib import Path
from acmnxt.io.loaders import read_table, ensure_datetime_index
from acmnxt.core.h2_pca import fit_pca
from acmnxt.core.regimes import fit_assign_regimes

def main()->int:
    ap=argparse.ArgumentParser(description='ACMnxt train')
    ap.add_argument('--csv',required=True); ap.add_argument('--equip',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--fast',action='store_true')
    a=ap.parse_args(); out=Path(a.out_dir)/("".join(ch if ch.isalnum() else '_' for ch in a.equip)); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df_raw=read_table(a.csv); df=ensure_datetime_index(df_raw)
    df=df.apply(pd.to_numeric, errors='coerce')
    df=df.resample('1min').mean().interpolate(limit_direction='both')
    df_imp=df.ffill().bfill()
    fit_pca(df_imp, n_components=min(5, max(1, min(df_imp.shape)-1)), out_dir=out)
    fit_assign_regimes(df_imp, out_dir=out, max_rows=20000 if a.fast else 100000)
    print('[train] artifacts saved under', out)
    return 0

if __name__=='__main__': raise SystemExit(main())
