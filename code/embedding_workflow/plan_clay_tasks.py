"""Plan stable Clay overlap tiles from state composite dimensions."""
import argparse
import json
import math
from pathlib import Path
import rasterio
from overlap_geometry import OverlapGrid

def choose_rectangular_grid(rows: int, cols: int, target_blocks: int) -> tuple[int, int]:
    """Choose a rectangular tile grid with bounded retained-block count per tile."""
    if rows < 1 or cols < 1:
        raise ValueError("block grid must be positive")
    target_blocks = max(1, target_blocks)
    ideal = max(1, math.ceil(rows * cols / target_blocks))
    preferred_limit = min(rows * cols, max(ideal, math.ceil(ideal * 1.25)))
    best = None
    fallback = None
    for gr in range(1, rows + 1):
        for gc in range(1, cols + 1):
            tiles = gr * gc
            if tiles < ideal:
                continue
            max_tile_rows = math.ceil(rows / gr)
            max_tile_cols = math.ceil(cols / gc)
            max_blocks = max_tile_rows * max_tile_cols
            if max_blocks > target_blocks:
                continue
            aspect_penalty = abs(math.log((gr / gc) / (rows / cols))) if gc and cols else 0.0
            score = (aspect_penalty, tiles - ideal, max_blocks, tiles, gr, gc)
            if fallback is None or score < fallback[0]:
                fallback = (score, gr, gc)
            if tiles <= preferred_limit and (best is None or score < best[0]):
                best = (score, gr, gc)
    if best is None:
        best = fallback
    if best is None:
        return rows, cols
    return best[1], best[2]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--composites",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--task-file",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--year",type=int,default=2022); p.add_argument("--target-blocks",type=int,default=1200); p.add_argument("--max-tasks",type=int,default=80); p.add_argument("--states",nargs="*"); p.add_argument("--tiling",choices=("rectangular","rows"),default="rectangular")
    a=p.parse_args(); grid=OverlapGrid(256,8); tasks=[]; records=[]
    allowed=set(a.states or [])
    for source in sorted(a.composites.glob(f"s2_annual_*_{a.year}_olmoearth.tif")):
        state=source.name.split("_")[2]
        if allowed and state not in allowed: continue
        with rasterio.open(source) as src: rows=grid.block_count(src.height); cols=grid.block_count(src.width); shape=[src.height,src.width]
        if a.tiling == "rows":
            grid_rows,grid_cols=min(rows,max(1,math.ceil(rows*cols/a.target_blocks))),1
        else:
            grid_rows,grid_cols=choose_rectangular_grid(rows,cols,a.target_blocks)
        ntiles=grid_rows*grid_cols; stem=f"clay-1.5_overlap-center50_{state}_{a.year}"; directory=a.output_root/"clay-1.5"/state
        missing=[]
        for index in range(ntiles):
            tile=directory/f"{stem}_tile{index:03d}.tif" if ntiles>1 else directory/f"{stem}.tif"
            validation=tile.with_suffix(".validation.json")
            if not (tile.exists() and validation.exists()): missing.append(index)
        records.append({"state":state,"source":str(source),"input_shape":shape,"block_grid":[rows,cols],"tile_grid":[grid_rows,grid_cols],"num_tiles":ntiles,"target_blocks":a.target_blocks,"max_blocks_per_tile":math.ceil(rows/grid_rows)*math.ceil(cols/grid_cols),"missing_tiles":missing})
        for index in missing:
            if len(tasks)>=a.max_tasks: break
            tasks.append((state,index,ntiles,grid_rows,grid_cols))
        if len(tasks)>=a.max_tasks: break
    a.task_file.parent.mkdir(parents=True,exist_ok=True); a.task_file.write_text("".join(f"{s} {i} {n} {gr} {gc}\n" for s,i,n,gr,gc in tasks)); a.manifest.write_text(json.dumps({"schema":2,"model":"clay-1.5","chip":256,"stride":128,"patch":8,"tiling":a.tiling,"target_blocks":a.target_blocks,"tasks":len(tasks),"states":records},indent=2))
    print(json.dumps({"task_file":str(a.task_file),"manifest":str(a.manifest),"tasks":len(tasks),"states_considered":len(records)}))
if __name__=="__main__": main()
