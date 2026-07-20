"""Memory-bounded tile-to-tract sufficient statistics and exact reduction."""
import argparse
import json
import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

def aggregate_tile(tile: Path, tracts_path: Path, output: Path, band_chunk: int = 64) -> None:
    tracts = gpd.read_file(tracts_path)
    with rasterio.open(tile) as src:
        tracts = tracts.to_crs(src.crs)
        tracts = tracts[tracts.geometry.intersects(box(*src.bounds))]
        geoids=[]; counts=[]; sums=[]; sumsqs=[]; minima=[]; maxima=[]
        for _, tract in tracts.iterrows():
            raw = from_bounds(*tract.geometry.bounds, transform=src.transform)
            c0=max(0,int(np.floor(raw.col_off))); r0=max(0,int(np.floor(raw.row_off)))
            c1=min(src.width,int(np.ceil(raw.col_off+raw.width))); r1=min(src.height,int(np.ceil(raw.row_off+raw.height)))
            if c0>=c1 or r0>=r1: continue
            window=Window(c0,r0,c1-c0,r1-r0)
            mask=geometry_mask([tract.geometry],out_shape=(r1-r0,c1-c0),transform=rasterio.windows.transform(window,src.transform),invert=True,all_touched=False)
            valid_reference=None; count=0; row_sum=np.zeros(src.count,np.float64); row_sumsq=np.zeros(src.count,np.float64)
            row_min=np.full(src.count,np.inf,np.float32); row_max=np.full(src.count,-np.inf,np.float32)
            for start in range(1,src.count+1,band_chunk):
                idx=list(range(start,min(src.count+1,start+band_chunk)))
                data=src.read(idx,window=window).astype(np.float32); values=data[:,mask]
                valid=np.isfinite(values).all(axis=0)
                if valid_reference is None: valid_reference=valid; count=int(valid.sum())
                elif not np.array_equal(valid_reference,valid): raise ValueError("band-dependent nodata is not supported")
                if count:
                    values=values[:,valid].astype(np.float64); sl=slice(start-1,start-1+len(idx))
                    row_sum[sl]=values.sum(1); row_sumsq[sl]=np.square(values).sum(1)
                    row_min[sl]=values.min(1); row_max[sl]=values.max(1)
            if count:
                geoids.append(str(tract.GEOID)); counts.append(count); sums.append(row_sum); sumsqs.append(row_sumsq); minima.append(row_min); maxima.append(row_max)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,GEOID=np.asarray(geoids),count=np.asarray(counts,np.int64),sum=np.asarray(sums),sumsq=np.asarray(sumsqs),min=np.asarray(minima),max=np.asarray(maxima))

def reduce_partials(
    paths: list[Path],
    output: Path,
    year: int,
    prefix: str,
    tracts_path: Path | None = None,
    qa_json: Path | None = None,
) -> None:
    totals={}
    for path in paths:
        with np.load(path) as part:
            for i,geoid in enumerate(part["GEOID"]):
                key=str(geoid); values=(int(part["count"][i]),part["sum"][i],part["sumsq"][i],part["min"][i],part["max"][i])
                if key not in totals: totals[key]=[values[0],values[1].copy(),values[2].copy(),values[3].copy(),values[4].copy()]
                else:
                    total=totals[key]; total[0]+=values[0]; total[1]+=values[1]; total[2]+=values[2]; total[3]=np.minimum(total[3],values[3]); total[4]=np.maximum(total[4],values[4])
    rows=[]
    dimensions=None
    for geoid,total in sorted(totals.items()):
        count,sums,sumsq,minimum,maximum=total; mean=sums/count; std=np.sqrt(np.maximum(sumsq/count-np.square(mean),0))
        dimensions=len(mean) if dimensions is None else dimensions
        if len(mean) != dimensions: raise ValueError("partial embedding dimensions differ")
        row={"GEOID":geoid,"year":year,"pixel_count":count}
        for i in range(len(mean)):
            name=f"{prefix}{i:04d}"; row[f"{name}_MEAN"]=mean[i]; row[f"{name}_MINIMUM"]=minimum[i]; row[f"{name}_MAXIMUM"]=maximum[i]; row[f"{name}_STD"]=std[i]
        rows.append(row)
    expected=set(totals)
    duplicate_expected=[]
    if tracts_path is not None:
        tract_geoids=gpd.read_file(tracts_path)["GEOID"].astype(str)
        duplicate_expected=sorted(tract_geoids[tract_geoids.duplicated()].unique().tolist())
        if duplicate_expected: raise ValueError(f"duplicate GEOIDs in tract file: {duplicate_expected[:10]}")
        expected=set(tract_geoids)
    missing=sorted(expected-set(totals)); unexpected=sorted(set(totals)-expected)
    if tracts_path is not None and dimensions is None:
        raise ValueError("no valid tract partials; embedding dimension is unknown")
    for geoid in missing:
        row={"GEOID":geoid,"year":year,"pixel_count":0}
        for i in range(dimensions or 0):
            name=f"{prefix}{i:04d}"
            row[f"{name}_MEAN"]=np.nan; row[f"{name}_MINIMUM"]=np.nan; row[f"{name}_MAXIMUM"]=np.nan; row[f"{name}_STD"]=np.nan
        rows.append(row)
    frame=pd.DataFrame(rows).sort_values("GEOID").reset_index(drop=True)
    if frame["GEOID"].duplicated().any(): raise ValueError("duplicate GEOIDs in reduced output")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(f".{output.name}.partial")
    frame.to_csv(temporary,index=False); os.replace(temporary,output)
    if qa_json is not None:
        report={
            "status":"complete" if not missing and not unexpected else "complete_with_missing_or_unexpected_tracts",
            "output":str(output),
            "year":year,
            "embedding_dimensions":dimensions,
            "expected_tracts":len(expected),
            "tracts_with_valid_pixels":len(totals),
            "zero_valid_pixel_tracts":len(missing),
            "zero_valid_pixel_geoids":missing,
            "unexpected_tracts":len(unexpected),
            "unexpected_geoids":unexpected,
            "duplicate_expected_geoids":duplicate_expected,
        }
        qa_json.parent.mkdir(parents=True,exist_ok=True)
        temporary_qa=qa_json.with_name(f".{qa_json.name}.partial")
        temporary_qa.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        os.replace(temporary_qa,qa_json)

def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("partial"); a.add_argument("--tile",type=Path,required=True); a.add_argument("--tracts",type=Path,required=True); a.add_argument("--output",type=Path,required=True); a.add_argument("--band-chunk",type=int,default=64)
    r=sub.add_parser("reduce"); r.add_argument("--partials",type=Path,nargs="+",required=True); r.add_argument("--output",type=Path,required=True); r.add_argument("--year",type=int,default=2022); r.add_argument("--prefix",default="E"); r.add_argument("--tracts",type=Path); r.add_argument("--qa-json",type=Path)
    x=p.parse_args()
    if x.command=="partial": aggregate_tile(x.tile,x.tracts,x.output,x.band_chunk)
    else: reduce_partials(x.partials,x.output,x.year,x.prefix,x.tracts,x.qa_json)
if __name__=="__main__": main()
