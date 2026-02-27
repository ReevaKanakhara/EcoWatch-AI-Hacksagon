import ee
import geemap
import numpy as np

def fetch_historical_loss(lat, lon, size_px=64):
    region = ee.Geometry.Point([lon, lat]).buffer(size_px * 30 / 2).bounds()
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    base_2000 = gfc.select('treecover2000').clip(region)
    loss_year = gfc.select('lossyear').clip(region)
    base_np = geemap.ee_to_numpy(base_2000, region=region) / 100.0
    loss_np = (geemap.ee_to_numpy(loss_year, region=region) > 0).astype(np.float32)
    return base_np, loss_np
