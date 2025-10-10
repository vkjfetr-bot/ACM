from __future__ import annotations
import argparse
from acmnxt.report.native import build_native_report

def main()->int:
    ap=argparse.ArgumentParser(description='ACMnxt report')
    ap.add_argument('--art-dir',required=True); ap.add_argument('--equip',required=True)
    a=ap.parse_args(); out=build_native_report(a.art_dir,a.equip); print('[report] Wrote', out); return 0

if __name__=='__main__': raise SystemExit(main())
