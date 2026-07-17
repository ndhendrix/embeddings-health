"""Ownership, halo, and tile-count invariance tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from overlap_geometry import OverlapGrid

def check(chip,patch,height,width):
    grid=OverlapGrid(chip,patch)
    assert grid.stride==chip//2 and grid.margin==chip//4
    assert grid.keep_tokens==chip//2//patch
    # Adjacent reads overlap by exactly half a chip; retained centers abut.
    assert grid.chip_origin(1)-grid.chip_origin(0)==grid.stride
    assert grid.crop_token_start*patch==grid.margin
    reference=[(r*grid.stride,min((r+1)*grid.stride,height)) for r in range(grid.block_count(height))]
    assert reference[0][0]==0 and reference[-1][1]==height
    for left,right in zip(reference,reference[1:]): assert left[1]==right[0]
    # Any legal row-tile count yields the identical ordered ownership intervals.
    for n in (1,2,3,7):
        if n>grid.block_count(height): continue
        assembled=[]
        for i in range(n):
            a,b=grid.tile_block_bounds(height,i,n); assembled.extend(reference[a:b])
            y0,y1=grid.owned_pixel_bounds(height,i,n)
            assert (y0,y1)==(reference[a][0],reference[b-1][1])
        assert assembled==reference
    assert grid.block_count(width)*grid.stride>=width
    # A rectangular 2x2 partition covers the full plane once.
    rectangles=[]
    for index in range(4):
        row,col=divmod(index,2); y0,y1=grid.owned_pixel_bounds(height,row,2); x0,x1=grid.owned_pixel_bounds(width,col,2); rectangles.append((x0,y0,x1,y1))
    assert sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in rectangles)==height*width
    for i,a in enumerate(rectangles):
        for b in rectangles[i+1:]: assert not (max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]))

if __name__=="__main__":
    check(128,4,9879,7225); check(256,8,9879,7225)
    print("PASS: half-stride center ownership is gap-free, unique, and tile-count invariant")
