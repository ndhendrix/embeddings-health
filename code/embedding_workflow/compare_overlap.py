"""Compare single-task and merged rectangular overlap rasters."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

def main():
    p=argparse.ArgumentParser(); p.add_argument("--reference",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--reference-csv",type=Path); p.add_argument("--candidate-csv",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--rtol",type=float,default=2e-5); p.add_argument("--atol",type=float,default=2e-6); p.add_argument("--band-chunk",type=int,default=16)
    a=p.parse_args(); maximum=0.; compared=0
    with rasterio.open(a.reference) as ref,rasterio.open(a.candidate) as got:
        if (ref.crs,ref.transform,ref.shape,ref.count)!=(got.crs,got.transform,got.shape,got.count): raise ValueError("raster geometry/schema differs")
        for _,window in ref.block_windows(1):
            for start in range(1,ref.count+1,a.band_chunk):
                idx=list(range(start,min(ref.count+1,start+a.band_chunk))); x=ref.read(idx,window=window); y=got.read(idx,window=window)
                if not np.array_equal(np.isfinite(x),np.isfinite(y)): raise ValueError(f"finite mask differs at {window}")
                valid=np.isfinite(x)
                if valid.any():
                    delta=float(np.max(np.abs(x[valid]-y[valid]))); maximum=max(maximum,delta); compared+=int(valid.sum())
                    if not np.allclose(x[valid],y[valid],rtol=a.rtol,atol=a.atol): raise ValueError(f"values differ at {window}; max_abs={delta}")
    result={"reference":str(a.reference),"candidate":str(a.candidate),"compared_values":compared,"max_absolute_difference":maximum,"rtol":a.rtol,"atol":a.atol}
    if bool(a.reference_csv) != bool(a.candidate_csv): raise ValueError("both tract CSV paths are required together")
    if a.reference_csv:
        ref=pd.read_csv(a.reference_csv,dtype={"GEOID":str}).sort_values("GEOID").reset_index(drop=True); got=pd.read_csv(a.candidate_csv,dtype={"GEOID":str}).sort_values("GEOID").reset_index(drop=True)
        if list(ref.columns)!=list(got.columns) or ref.GEOID.tolist()!=got.GEOID.tolist(): raise ValueError("tract CSV schema/GEOIDs differ")
        numeric=[c for c in ref.columns if c!="GEOID"]
        if not np.allclose(ref[numeric],got[numeric],rtol=a.rtol,atol=a.atol,equal_nan=True): raise ValueError("tract statistics differ")
        result["tract_rows"]=len(ref); result["tract_statistics_equal"]=True
    a.output.write_text(json.dumps(result,indent=2)); print(a.output)
if __name__=="__main__": main()
