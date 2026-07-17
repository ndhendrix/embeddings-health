"""Create a projected, block-aligned location crop with context."""
import argparse
from pathlib import Path
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--lat",type=float,required=True); p.add_argument("--lon",type=float,required=True); p.add_argument("--box-km",type=float,default=26)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(a.input) as src:
        x,y=Transformer.from_crs("EPSG:4326",src.crs,always_xy=True).transform(a.lon,a.lat); half=a.box_km*500
        raw=rasterio.windows.from_bounds(x-half,y-half,x+half,y+half,src.transform)
        c0=max(0,int(raw.col_off)); r0=max(0,int(raw.row_off)); c1=min(src.width,int(raw.col_off+raw.width+1)); r1=min(src.height,int(raw.row_off+raw.height+1)); window=Window(c0,r0,c1-c0,r1-r0)
        profile=src.profile.copy(); profile.update(width=int(window.width),height=int(window.height),transform=rasterio.windows.transform(window,src.transform),BIGTIFF="IF_SAFER")
        with rasterio.open(a.output,"w",**profile) as dst:
            for _,block in dst.block_windows(1):
                source=Window(window.col_off+block.col_off,window.row_off+block.row_off,block.width,block.height)
                dst.write(src.read(window=source),window=block)
    print(a.output)
if __name__=="__main__": main()
