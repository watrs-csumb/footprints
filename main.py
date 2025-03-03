from src.Footprint import Footprint

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, help="Path to Ameriflux .dat file")
parser.add_argument("--output", "-o", nargs="?", type=str, help="Path to output .csv file")
parser.add_argument("--contour", "-c", nargs="?", help="""Percentage of source area for which to provide contours, must be between 10% and 90%.\n
            Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")\n
            Expressed either in percentages ("80") or as fractions of 1 ("0.8").\n
            Default is [90]. Set to "None" for no output of percentages""")
parser.add_argument("--boundary-layer-height", "-b", nargs="?", type=float, help="Boundary layer height [m]")

def main():
    args = parser.parse_args()
    
    afdat = args.input
    blh = args.boundary_layer_height
    contour = args.contour
    
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
    if blh:
        footprint.boundary_layer_height = blh
    if contour:
        footprint.contour_src_pct = contour
    
    footprint_raster = footprint.attach(df).draw().rasterize(30)
    
    polygon = footprint_raster.polygonize(0.8)
    
    fig, ax = plt.subplots(figsize = (6, 6))
    assert footprint_raster.raster is not None
    assert footprint_raster.geometry is not None
    
    minx, miny, maxx, maxy = footprint_raster.geometry.union_all().bounds
    im = ax.imshow(footprint_raster.raster, cmap='hot', extent=(minx, maxx, miny, maxy))
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title(f'Accumulated Raster\n({tower_location[0]}, {tower_location[1]})')
    fig.colorbar(im, ax=ax, label='Overlap Count')
    plt.savefig("footprint_heat.png")
    
    fig, ax = plt.subplots(figsize = (6, 6))
    polygon.plot(ax = ax, edgecolor = 'black', facecolor = 'none')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_title(f'Tower Footprint Polygon\n({tower_location[0]}, {tower_location[1]})')
    plt.savefig("footprint_polygon.png")

if __name__ == "__main__":
    main()