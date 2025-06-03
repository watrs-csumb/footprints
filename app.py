import sys
import pathlib
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

try:
    from footprint import Footprint
except ImportError:
    print("Footprint module could not be loaded. Are you in the correct directory?")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import pandas as pd
    import rasterio
    import rasterio.features
    import tomllib
except ImportError:
    print("Required dependencies not found. Please install them using 'pip install -r requirements.txt'")
    sys.exit(1)

def main():
    cfg = tomllib.load(open("app.toml", "rb"))
    afdat = f"{cfg['input']['file'].replace("\\",'/')}"
    file = pathlib.Path(afdat)
    
    using_reference_eto = cfg["input"]["weigh_by_eto"]
    reference_eto_file = cfg["input"]["eto_file"]
    
    tower_location = cfg["input"]["location"]
    blh = cfg["input"]["boundary_layer_height"]
    contour = cfg["input"]["source_contour_ratio"]
    
    max_rows = cfg["input"]["max_rows"]
    
    outputdir = cfg["output"]["output_dir"]
    resolution = cfg["output"]["spatial_resolution"]
    overlap_threshold = cfg["output"]["overlap_threshold"]
    smoothing_factor = cfg["output"]["smoothing_factor"]
    use_coverage_union = cfg["output"]["coverage_union"]
    
    produce_heatmap = cfg["graphs"]["heatmap"]
    produce_polygon_chart = cfg["graphs"]["polygon"]
    
    # Validate input data exists.
    if not file.exists():
        raise FileNotFoundError(f"File '{afdat}' could not be found.")
    
    # Validate output directory exists.
    if not pathlib.Path(outputdir).exists() or not pathlib.Path(outputdir).is_dir():
        raise NotADirectoryError(f"Directory {outputdir} does not exist")

    # Append a trailing / if missing to avoid file path issues.
    if not outputdir.endswith("/"):
        outputdir += "/"
    
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
    if type(overlap_threshold) is not float or overlap_threshold < 0 or overlap_threshold > 1:
        raise ValueError("Overlap threshold must be a number between 0 and 1")
    if type(smoothing_factor) is not int or smoothing_factor < 1.:
        raise ValueError("Smoothing factor must be a number greater than or equal to 1")
    if type(produce_heatmap) is not bool:
        raise TypeError("heatmap must be a boolean")
    if type(produce_polygon_chart) is not bool:
        raise TypeError("polygon must be a boolean")
    
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
    footprint_raster = footprint.attach(df, rdf).draw(max_rows).rasterize(resolution)
    # Create a polygon from the raster.
    polygon = footprint_raster.polygonize(overlap_threshold, smoothing_factor, use_coverage_union)
    
    pathlib.Path(f"{outputdir + file.stem}").mkdir(exist_ok=True)
    polygon.to_file(f"{outputdir + file.stem}/{file.stem}_footprint.shp")
    polygon.to_file(f"{outputdir}{file.stem}_footprint.geojson", driver="GeoJSON")
    
    if footprint_raster.daily_timeseries is not None:
        footprint_raster.daily_timeseries.to_file(f"{outputdir}{file.stem}_footprint_timeseries.geojson", driver="GeoJSON")
    
    assert footprint_raster.raster is not None
    assert footprint_raster.geometry is not None
    
    if produce_heatmap:
        fig, ax = plt.subplots(figsize = (6, 6))
        minx, miny, maxx, maxy = footprint_raster.geometry.union_all().bounds
        im = ax.imshow(footprint_raster.raster, cmap="hot", extent=(minx, maxx, miny, maxy))
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Accumulated Raster\n({tower_location[0]}, {tower_location[1]})")
        fig.colorbar(im, ax=ax, label="Overlap Contribution", format=mtick.PercentFormatter(1.0))
        plt.savefig(f"{outputdir}{file.stem}_footprint_heat.png")

    
    if produce_polygon_chart:
        print("Creating footprint polygon graph...")
        fig, ax = plt.subplots(figsize = (6, 6))
        polygon.plot(ax = ax, edgecolor = "black", facecolor = "none")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Tower Footprint Polygon\n({tower_location[0]}, {tower_location[1]})")
        plt.savefig(f"{outputdir}{file.stem}_footprint_polygon.png")
    
    # Export raster.
    print("Exporting footprint raster to .tif file...")
    with rasterio.open(
        f"{outputdir}{file.stem}_footprint_raster.tif", 
        "w+",
        transform=footprint_raster.transform,
        crs=footprint_raster.utm_crs,
        driver="GTiff",
        count=3,
        dtype=footprint_raster.raster.dtype,
        width=footprint_raster.raster.shape[1],
        height=footprint_raster.raster.shape[0]) as dst:
        
        dst.write(footprint_raster.raster, 1)
        dst.set_band_description(1, "Weighted Overlaps")
        
        # Get the polygon from the footprint's geodataframe.
        shape = [feature["geometry"] for index, feature in polygon.iterrows()]
        
        shape_data = rasterio.features.rasterize(
            shape,
            out_shape=footprint_raster.raster.shape,
            transform=footprint_raster.transform, # type: ignore
            fill=0)
        
        
        dst.write(shape_data, 2)
        dst.set_band_description(2, "Footprint Mask")
        
        dst.write(footprint_raster.raw_raster, 3)
        dst.set_band_description(3, "Raw Raster")

    print("Done!")
if __name__ == "__main__":
    main()