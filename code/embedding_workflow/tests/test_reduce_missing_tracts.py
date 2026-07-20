"""Reducer retains expected zero-valid-pixel tracts and writes a QA report."""
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aggregate_tiles import reduce_partials


def test_reduce_missing_tracts(tmp_path):
    partial=tmp_path/"tile000.npz"
    np.savez(
        partial,
        GEOID=np.asarray(["01001000100"]),
        count=np.asarray([2],np.int64),
        sum=np.asarray([[4.0,8.0]]),
        sumsq=np.asarray([[10.0,34.0]]),
        min=np.asarray([[1.0,3.0]]),
        max=np.asarray([[3.0,5.0]]),
    )
    tracts=tmp_path/"tracts.geojson"
    gpd.GeoDataFrame(
        {"GEOID":["01001000100","01001000200"]},
        geometry=[box(0,0,1,1),box(1,0,2,1)],
        crs="EPSG:4326",
    ).to_file(tracts,driver="GeoJSON")
    output=tmp_path/"tracts.csv"; qa=tmp_path/"tracts.validation.json"
    reduce_partials([partial],output,2022,"CL",tracts,qa)

    frame=pd.read_csv(output,dtype={"GEOID":str})
    assert frame.GEOID.tolist()==["01001000100","01001000200"]
    assert frame.pixel_count.tolist()==[2,0]
    assert frame.loc[1,["CL0000_MEAN","CL0001_STD"]].isna().all()
    report=json.loads(qa.read_text())
    assert report["expected_tracts"]==2
    assert report["tracts_with_valid_pixels"]==1
    assert report["zero_valid_pixel_geoids"]==["01001000200"]
