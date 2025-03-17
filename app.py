from footprint import Footprint

import pathlib
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    cfg = tomllib.load(open("app.toml", "rb"))
    afdat = f"{cfg['input']['file'].replace("\\",'/')}"
    file = pathlib.Path(afdat)
    
    using_reference_eto = cfg["input"]["weigh_by_eto"]
    reference_eto_file = cfg["input"]["eto_file"]
    
    tower_location = cfg["input"]["location"]
    blh = cfg["input"]["boundary_layer_height"]
    contour = cfg["input"]["source_contour_ratio"]
    
    outputdir = cfg["output"]["output_dir"]
    resolution = cfg["output"]["spatial_resolution"]
    overlap_threshold = cfg["output"]["overlap_threshold"]
    
    # Validate input data exists.
    if not file.exists():
        raise FileNotFoundError(f"File '{afdat}' could not be found.")
    
    # Validate output directory exists.
    if not pathlib.Path(outputdir).exists() or not pathlib.Path(outputdir).is_dir():
        raise NotADirectoryError(f"Directory {outputdir} does not exist")
    
    if type(using_reference_eto) is not bool:
        raise TypeError("weigh_by_eto must be a boolean")
    
    if using_reference_eto:
        if not pathlib.Path(reference_eto_file).exists():
            raise FileNotFoundError(f"File '{reference_eto_file}' could not be found.")
    
    if type(resolution) is not int or resolution <= 0:
        raise ValueError("Spatial resolution must be a positive integer")
    if (type(blh) not in [float, int] and type(blh) is not int) or blh < 0.:
        raise ValueError("Boundary layer height must be a positive number")
    if type(contour) not in [float, int, list] or np.min(contour) < 0.:
        raise ValueError("Source contour ratio must be positive number(s)")
    if type(overlap_threshold) not in [float, int] or overlap_threshold < 0 or overlap_threshold > 1:
        raise ValueError("Overlap threshold must be a number between 0 and 1")
    
    df = pd.read_csv(afdat)
    rdf = pd.read_csv(reference_eto_file) if using_reference_eto else None
    
    # Create footprint object.
    footprint = Footprint(tower_location)
    
    # Override footprint parameters.
    if blh:
        footprint.boundary_layer_height = blh
    if contour:
        footprint.contour_src_pct = contour
    
    # Attach data to object then draw footprint and create a raster.
    footprint_raster = footprint.attach(df, rdf).draw().rasterize(resolution)
    # Create a polygon from the raster.
    polygon = footprint_raster.polygonize(overlap_threshold)
    
    pathlib.Path(f"{outputdir + file.stem}").mkdir(exist_ok=True)
    polygon.to_file(f"{outputdir + file.stem}/{file.stem}_footprint.shp")
    polygon.to_file(f"{outputdir}{file.stem}_footprint.geojson", driver="GeoJSON")
    
    fig, ax = plt.subplots(figsize = (6, 6))
    assert footprint_raster.raster is not None
    assert footprint_raster.geometry is not None
    
    minx, miny, maxx, maxy = footprint_raster.geometry.union_all().bounds
    im = ax.imshow(footprint_raster.raster, cmap="hot", extent=(minx, maxx, miny, maxy))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Accumulated Raster\n({tower_location[0]}, {tower_location[1]})")
    fig.colorbar(im, ax=ax, label="Overlap Count")
    plt.savefig(f"{outputdir}{file.stem}_footprint_heat.png")
    
    fig, ax = plt.subplots(figsize = (6, 6))
    polygon.plot(ax = ax, edgecolor = "black", facecolor = "none")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Tower Footprint Polygon\n({tower_location[0]}, {tower_location[1]})")
    plt.savefig(f"{outputdir}{file.stem}_footprint_polygon.png")

if __name__ == "__main__":
    main()