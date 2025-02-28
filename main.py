from src.Footprint import Footprint

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, help="Path to Ameriflux .dat file")
parser.add_argument("--output", "-o", type=str, help="Path to output .csv file")
args = parser.parse_args()

def main():
    # args = parser.parse_args()
    
    # afdat = args.input
    # out = args.output
    afdat = "./data/B_SanLuis/Data/0_Input_Datasets/Lettuce/Lettuce2024_30min_gapfilled.csv"
    if not pathlib.Path(afdat).exists():
        raise FileNotFoundError(f"File {afdat} does not exist")
    
    df = pd.read_csv(afdat)
    
    tower_location = 36.456667, 121.382222
    hemisphere = "North"
    tower_spec = {
        "zm": 2.0,
        "z0": 0.1,
        "d": 0.335
    }
    
    footprint = Footprint(tower_location, tower_spec, hemisphere)
    footprint_raster = footprint.attach(df).draw().rasterize()
    
    polygon = footprint_raster.polygonize(0.8)
    
    fig, ax = plt.subplots(figsize = (6, 6))
    assert footprint_raster.raster is not None
    assert footprint_raster.geometry is not None
    
    minx, miny, maxx, maxy = footprint_raster.geometry.union_all().bounds
    im = ax.imshow(footprint_raster.raster, cmap='hot', extent=(minx, maxx, miny, maxy))
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title('Ameriflux Footprint')
    fig.colorbar(im, ax=ax, label='Overlap Count')
    plt.savefig("footprint.png")

if __name__ == "__main__":
    main()