from footprint import Footprint

import pathlib
import tomllib

import matplotlib.pyplot as plt
import pandas as pd

def main():
    cfg = tomllib.load(open("app.toml", "rb"))
    afdat = f".{cfg['input']['file']}"
    tower_location = cfg["input"]["location"]
    tower_spec = cfg["input"]["tower_spec"]
    blh = cfg["input"]["boundary_layer_height"]
    contour = cfg["input"]["source_contour_ratio"]
    hemisphere = cfg["input"]["hemisphere"]
    outputdir = cfg["output"]["output_dir"]
    resolution = cfg["output"]["spatial_resolution"]
    overlap_threshold = cfg["output"]["overlap_threshold"]
    
    # Validate input data exists.
    if not pathlib.Path(afdat).exists():
        raise FileNotFoundError(f"File {afdat} does not exist")
    
    # Validate output directory exists.
    if not pathlib.Path(outputdir).exists() or not pathlib.Path(outputdir).is_dir():
        raise FileNotFoundError(f"Directory {outputdir} does not exist")
    
    df = pd.read_csv(afdat)
    
    # Create footprint object.
    footprint = Footprint(tower_location, tower_spec, hemisphere)
    
    # Override footprint parameters.
    if blh:
        footprint.boundary_layer_height = blh
    if contour:
        footprint.contour_src_pct = contour
    
    # Attach data to object then draw footprint and create a raster.
    footprint_raster = footprint.attach(df).draw().rasterize(resolution)
    # Create a polygon from the raster.
    polygon = footprint_raster.polygonize(overlap_threshold)
    
    fig, ax = plt.subplots(figsize = (6, 6))
    assert footprint_raster.raster is not None
    assert footprint_raster.geometry is not None
    
    minx, miny, maxx, maxy = footprint_raster.geometry.union_all().bounds
    im = ax.imshow(footprint_raster.raster, cmap="hot", extent=(minx, maxx, miny, maxy))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Accumulated Raster\n({tower_location[0]}, {tower_location[1]})")
    fig.colorbar(im, ax=ax, label="Overlap Count")
    plt.savefig(f".{outputdir}footprint_heat.png")
    
    fig, ax = plt.subplots(figsize = (6, 6))
    polygon.plot(ax = ax, edgecolor = "black", facecolor = "none")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Tower Footprint Polygon\n({tower_location[0]}, {tower_location[1]})")
    plt.savefig(f".{outputdir}footprint_polygon.png")

if __name__ == "__main__":
    main()