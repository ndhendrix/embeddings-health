---
active: true
iteration: 1
session_id: f5c47520-6fad-48bc-a32b-59ec315248e3
max_iterations: 10
completion_promise: "DONE"
started_at: "2026-05-25T14:54:50Z"
---

It starts but it's giving me the following error. Can you run this locally just until you're sure that it can process tiles sucessfully for a single state please? Here's the error: [                                        ] | 0% Completed | 1.15 sAborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022149T153559.v2.0/HLS.S30.T19TCG.2022149T153559.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022149T153559.v2.0/HLS.S30.T19TCG.2022149T153559.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022149T153559.v2.0/HLS.S30.T19TCG.2022149T153559.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022149T153559.v2.0/HLS.S30.T19TCG.2022149T153559.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TBF.2022137T153821.v2.0/HLS.S30.T19TBF.2022137T153821.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TBF.2022137T153821.v2.0/HLS.S30.T19TBF.2022137T153821.v2.0.Fmask.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022092T153809.v2.0/HLS.S30.T19TCG.2022092T153809.v2.0.B03.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022092T153809.v2.0/HLS.S30.T19TCG.2022092T153809.v2.0.B03.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022092T153809.v2.0/HLS.S30.T19TCG.2022092T153809.v2.0.B03.tif:1
Aborting load due to failure while reading: https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022092T153809.v2.0/HLS.S30.T19TCG.2022092T153809.v2.0.B03.tif:1
[                                        ] | 0% Completed | 1.25 s
    ERROR during dask compute:
Traceback (most recent call last):
  File rasterio/_base.pyx, line 320, in rasterio._base.DatasetBase.__init__
  File rasterio/_base.pyx, line 232, in rasterio._base.open_dataset
  File rasterio/_err.pyx, line 359, in rasterio._err.exc_wrap_pointer
rasterio._err.CPLE_OpenFailedError: '/vsicurl/https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSS30.020/HLS.S30.T19TCG.2022149T153559.v2.0/HLS.S30.T19TCG.2022149T153559.v2.0.Fmask.tif' not recognized as being in a supported file format.
