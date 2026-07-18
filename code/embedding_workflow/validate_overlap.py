"""Fail-fast structural and numerical validation for an overlap tile."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from models import get_model

def main():
    p=argparse.ArgumentParser(); p.add_argument("--raster",type=Path,required=True); p.add_argument("--model",required=True); p.add_argument("--allow-partial",action="store_true"); p.add_argument("--allow-empty",action="store_true"); p.add_argument("--min-finite-fraction",type=float,default=0.001); p.add_argument("--json",type=Path)
    a=p.parse_args(); spec=get_model(a.model)
    with rasterio.open(a.raster) as src:
        if src.count!=spec.dimensions: raise ValueError(f"expected {spec.dimensions} bands, got {src.count}")
        expected_res=spec.patch_pixels*10
        if not np.allclose(src.res,(expected_res,expected_res)): raise ValueError(f"unexpected resolution {src.res}")
        tags=src.tags(); required=("model","model_revision","chip_pixels","stride_pixels","retained_center_pixels","source_composite","workflow")
        missing=[k for k in required if not tags.get(k)]
        if missing: raise ValueError(f"missing provenance tags: {missing}")
        finite=0; total=0; minimum=np.inf; maximum=-np.inf
        for _,window in src.block_windows(1):
            data=src.read(1,window=window); mask=np.isfinite(data); finite+=int(mask.sum()); total+=data.size
            if mask.any(): minimum=min(minimum,float(data[mask].min())); maximum=max(maximum,float(data[mask].max()))
        fraction=finite/total
        if finite==0:
            if not a.allow_empty: raise ValueError("no finite embedding values")
            minimum=None; maximum=None
        elif not maximum>minimum: raise ValueError("no varying embedding values")
        if finite and not a.allow_partial and fraction<a.min_finite_fraction: raise ValueError(f"finite fraction too low: {fraction:.3%}")
        result={"path":str(a.raster),"model":a.model,"shape":[src.count,src.height,src.width],"resolution":list(src.res),"finite_fraction_band1":fraction,"min_band1":minimum,"max_band1":maximum,"tags":tags}
    target=a.json or a.raster.with_suffix(".validation.json"); target.write_text(json.dumps(result,indent=2,sort_keys=True)); print(target)
if __name__=="__main__": main()
