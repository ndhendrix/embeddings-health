"""Compare tile-partial reduction with aggregation of the merged RI raster."""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aggregate_tiles import aggregate_tile, reduce_partials

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--merged",type=Path,required=True); p.add_argument("--tiles",type=Path,nargs="+",required=True); p.add_argument("--tracts",type=Path,required=True); p.add_argument("--work-dir",type=Path)
    a=p.parse_args(); root=a.work_dir or Path(tempfile.mkdtemp(prefix="ri-aggregate-")); root.mkdir(parents=True,exist_ok=True)
    try:
        tile_parts=[]
        for i,tile in enumerate(a.tiles):
            part=root/f"tile{i}.npz"; aggregate_tile(tile,a.tracts,part,band_chunk=64); tile_parts.append(part)
        merged_part=root/"merged.npz"; aggregate_tile(a.merged,a.tracts,merged_part,band_chunk=64)
        tile_csv=root/"tiles.csv"; merged_csv=root/"merged.csv"
        reduce_partials(tile_parts,tile_csv,2022,"OE"); reduce_partials([merged_part],merged_csv,2022,"OE")
        tiled=pd.read_csv(tile_csv,dtype={"GEOID":str}).sort_values("GEOID").reset_index(drop=True)
        direct=pd.read_csv(merged_csv,dtype={"GEOID":str}).sort_values("GEOID").reset_index(drop=True)
        assert list(tiled.columns)==list(direct.columns); assert tiled["GEOID"].tolist()==direct["GEOID"].tolist()
        numeric=[c for c in tiled.columns if c not in ("GEOID",)]
        assert np.allclose(tiled[numeric],direct[numeric],rtol=2e-6,atol=1e-7,equal_nan=True)
        print(f"PASS: tile partials equal merged-raster aggregation for {len(tiled)} RI tracts")
    finally:
        if a.work_dir is None: shutil.rmtree(root)
if __name__=="__main__": main()
