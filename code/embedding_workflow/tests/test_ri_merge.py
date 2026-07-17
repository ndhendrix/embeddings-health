"""Exact real-grid merge validation using the Rhode Island composite."""
import argparse, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge import merge_tiles

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--work-dir",type=Path)
    a=p.parse_args(); root=a.work_dir or Path(tempfile.mkdtemp(prefix="ri-merge-")); root.mkdir(parents=True,exist_ok=True)
    try:
        with rasterio.open(a.input) as src:
            h,w=min(src.height,4096),min(src.width,4096)
            base=src.read(1,window=Window(0,0,w,h),out_shape=(h//4,w//4)).astype("float32")
            data=np.stack([base,base*2,base+7,np.square(base)])
            transform=src.transform*Affine.scale(4,4)
            profile=dict(driver="GTiff",dtype="float32",count=4,height=data.shape[1],width=data.shape[2],crs=src.crs,transform=transform,nodata=np.nan,tiled=True,blockxsize=256,blockysize=256,compress="deflate",BIGTIFF="IF_SAFER")
        ref=root/"reference.tif"
        with rasterio.open(ref,"w",**profile) as dst:
            dst.write(data); dst.update_tags(model="ri-test",year="2022")
            for i in range(1,5): dst.set_band_description(i,f"T{i:04d}")
        cuts=[0,data.shape[1]//3,2*data.shape[1]//3,data.shape[1]]; tiles=[]
        for i,(start,stop) in enumerate(zip(cuts,cuts[1:])):
            tile=root/f"tile{i:03d}.tif"; tp=profile|{"height":stop-start,"transform":rasterio.windows.transform(Window(0,start,data.shape[2],stop-start),transform)}
            with rasterio.open(tile,"w",**tp) as dst:
                dst.write(data[:,start:stop]); dst.update_tags(model="ri-test",year="2022")
                for b in range(1,5): dst.set_band_description(b,f"T{b:04d}")
            tiles.append(tile)
        merged=root/"merged.tif"; merge_tiles(tiles,merged)
        with rasterio.open(ref) as expected,rasterio.open(merged) as actual:
            assert expected.crs==actual.crs and expected.transform==actual.transform and expected.bounds==actual.bounds
            assert expected.shape==actual.shape and expected.descriptions==actual.descriptions and expected.tags()==actual.tags()
            for _,window in expected.block_windows(1): assert np.array_equal(expected.read(window=window),actual.read(window=window),equal_nan=True)
            print(f"PASS: exact RI tile merge ({actual.width}x{actual.height}, 4 bands, 3 tiles)")
        assert all(t.exists() for t in tiles)
    finally:
        if a.work_dir is None: shutil.rmtree(root)
if __name__=="__main__": main()
