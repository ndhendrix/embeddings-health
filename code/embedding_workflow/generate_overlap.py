"""Resumable half-chip-stride inference retaining only chip centers."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import torch

HERE=Path(__file__).resolve(); sys.path.insert(0,str(HERE.parents[1]/"embedding_generation"))
import embed as engine
from models import get_model
from overlap_geometry import OverlapGrid
from utils.cog_writer import write_cog

def _read_chip(src,row,col,size):
    return src.read(window=Window(col,row,size,size),boundless=True,fill_value=np.nan).astype("float32")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tile-index",type=int,default=0); p.add_argument("--num-tiles",type=int,default=1); p.add_argument("--grid-rows",type=int); p.add_argument("--grid-cols",type=int,default=1); p.add_argument("--year",type=int,default=2022); p.add_argument("--batch-size",type=int); p.add_argument("--test-blocks",type=int); p.add_argument("--force",action="store_true")
    a=p.parse_args(); spec=get_model(a.model); grid=OverlapGrid(spec.chip_pixels,spec.patch_pixels); batch_size=a.batch_size or spec.batch_size
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.family=="olmoearth": model=engine.load_olmoearth(spec.variant).to(device); infer=engine.run_olmoearth_batch
    else: model=engine.load_clay().to(device); infer=engine.run_clay_batch
    with rasterio.open(a.input) as src:
        grid_rows=a.grid_rows or a.num_tiles; grid_cols=a.grid_cols
        if grid_rows*grid_cols != a.num_tiles: raise ValueError("num_tiles must equal grid_rows * grid_cols")
        tile_row,tile_col=divmod(a.tile_index,grid_cols)
        br0,br1=grid.tile_block_bounds(src.height,tile_row,grid_rows); bc0,bc1=grid.tile_block_bounds(src.width,tile_col,grid_cols)
        y0,y1=grid.owned_pixel_bounds(src.height,tile_row,grid_rows); x0,x1=grid.owned_pixel_bounds(src.width,tile_col,grid_cols)
        out_h=(y1-y0+grid.patch-1)//grid.patch; out_w=(x1-x0+grid.patch-1)//grid.patch
        mmap=a.output.with_name(a.output.name+f".tile{a.tile_index:03d}.mmap"); ckpt=mmap.with_suffix(".checkpoint.json")
        shape=(spec.dimensions,out_h,out_w); stat=a.input.stat()
        config={"schema":1,"model":a.model,"repository":spec.repository,"revision":spec.revision,"year":a.year,"chip":grid.chip,"stride":grid.stride,"patch":grid.patch,"crop":"center50","tile_index":a.tile_index,"num_tiles":a.num_tiles,"grid":[grid_rows,grid_cols],"block_rows":[br0,br1],"block_cols":[bc0,bc1],"shape":list(shape),"input":str(a.input.resolve()),"input_size":stat.st_size,"input_mtime_ns":stat.st_mtime_ns,"test_blocks":a.test_blocks}
        if a.force: mmap.unlink(missing_ok=True); ckpt.unlink(missing_ok=True)
        done=0
        if mmap.exists() or ckpt.exists():
            if not (mmap.exists() and ckpt.exists()): raise RuntimeError("incomplete checkpoint pair; rerun with --force")
            saved=json.loads(ckpt.read_text())
            if saved.get("config") != config: raise RuntimeError("checkpoint configuration differs; rerun with --force")
            done=int(saved["done"])
        out=np.memmap(mmap,dtype="float32",mode="r+" if done else "w+",shape=shape)
        if not done: out[:]=np.nan; out.flush()
        blocks=[(r,c) for r in range(br0,br1) for c in range(bc0,bc1)]
        if a.test_blocks is not None: blocks=blocks[:a.test_blocks]
        for offset in range(done,len(blocks),batch_size):
            batch=blocks[offset:offset+batch_size]; chips=[]; latlons=[]
            for r,c in batch:
                row,col=grid.chip_origin(r),grid.chip_origin(c); chips.append(_read_chip(src,row,col,grid.chip))
                latlons.append(engine.chip_center_latlon(src.transform,src.crs,row,col,grid.chip))
            chip_array=np.stack(chips)
            spatial=infer(model,chip_array,np.asarray(latlons,np.float32),device,a.year)
            # Preserve composite nodata: model imputation is needed for attention,
            # but must not turn wholly empty state-bbox patches into embeddings.
            pixel_valid=np.isfinite(chip_array).any(axis=1)
            patch_valid=pixel_valid.reshape(len(batch),grid.chip//grid.patch,grid.patch,grid.chip//grid.patch,grid.patch).any(axis=(2,4))
            t0=grid.crop_token_start; t1=t0+grid.keep_tokens
            for i,(r,c) in enumerate(batch):
                owner_y=r*grid.stride; owner_x=c*grid.stride
                valid_h=(min(owner_y+grid.stride,src.height)-owner_y+grid.patch-1)//grid.patch
                valid_w=(min(owner_x+grid.stride,src.width)-owner_x+grid.patch-1)//grid.patch
                rr=(owner_y-y0)//grid.patch; cc=(owner_x-x0)//grid.patch
                kept=spatial[i,t0:t1,t0:t1,:][:valid_h,:valid_w].transpose(2,0,1)
                valid=patch_valid[i,t0:t1,t0:t1][:valid_h,:valid_w]
                kept[:,~valid]=np.nan
                out[:,rr:rr+valid_h,cc:cc+valid_w]=kept
            out.flush(); temporary=ckpt.with_suffix(".tmp")
            temporary.write_text(json.dumps({"config":config,"done":offset+len(batch)},sort_keys=True)); os.replace(temporary,ckpt)
        transform=Affine(src.transform.a*grid.patch,src.transform.b,src.transform.c+src.transform.a*x0,src.transform.d,src.transform.e*grid.patch,src.transform.f+src.transform.e*y0)
        crs=src.crs
    output=a.output if a.num_tiles==1 else a.output.with_name(f"{a.output.stem}_tile{a.tile_index:03d}{a.output.suffix}")
    names=[f"{'OE' if spec.family=='olmoearth' else 'CL'}{i:04d}" for i in range(spec.dimensions)]
    os.environ.setdefault("EMBEDDING_COG_INTERLEAVE","band"); write_cog(out,transform,crs,output,band_names=names,overviews=False,interleave="band")
    with rasterio.open(output,"r+") as dst:
        dst.update_tags(model=a.model,model_repository=spec.repository,model_revision=spec.revision,year=str(a.year),chip_pixels=str(grid.chip),stride_pixels=str(grid.stride),retained_center_pixels=str(grid.stride),patch_pixels=str(grid.patch),ownership="half-open",tile_grid=f"{grid_rows}x{grid_cols}",nodata_policy="empty_source_patches_preserved",state_edge_context="boundless_nan_imputed",source_composite=str(a.input.resolve()),source_size=str(stat.st_size),code_git_sha=os.environ.get("CODE_GIT_SHA","unknown"),workflow="overlap-center50-v1")
    del out; mmap.unlink(missing_ok=True); ckpt.unlink(missing_ok=True)
    print(f"wrote {output}; chip={grid.chip}, stride={grid.stride}, center={grid.stride}")
if __name__=="__main__": main()
