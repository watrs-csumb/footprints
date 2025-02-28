from typing import Self
from .core import FPP

import pandas as pd
import geopandas as gpd
import numpy as np

from pyproj import Transformer
from rasterio import features
from rasterio.features import shapes
from rasterio.transform import from_origin, Affine
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapelysmooth import taubin_smooth

BOUNDARY_LAYER_HEIGHT = 2000
CONTOUR_SRC_PCT = [90]

class Footprint:
    _req_columns = ["date_time", "WS", "USTAR", "WD", "V_SIGMA", "MO_LENGTH"]
    
    def __init__(self, 
                tower_location: tuple[float, float],
                tower_spec: dict,
                hemisphere: str = "North"):
        self.latitude = tower_location[0]
        self.longitude = tower_location[1]
        self.hemisphere = hemisphere
        
        self.utm_crs: str = ""
        self.easting: float = 0.0
        self.northing: float = 0.0
        self.raster: np.ndarray | None = None
        self.data: pd.DataFrame | None = None
        self.geometry: gpd.GeoDataFrame | None = None
        self.transform: Affine | None = None
        
        self._to_utm()
        
        if "zm" not in tower_spec.keys():
            raise ValueError("tower_spec must contain keys: ['zm']")
        
        if "z0" not in tower_spec.keys() and "umean" not in tower_spec.keys():
            raise ValueError("tower_spec must contain either 'z0' or 'umean'.")
        
        self.tower_spec = tower_spec
    
    def _to_utm(self):
        utm_zone = int((self.longitude + 180) / 6) + 1  # calculate UTM zone based on longitude
        self.utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"  # create UTM CRS
        
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs)
        self.easting, self.northing = transformer.transform(self.latitude, self.longitude)
    
    def attach(self, data: pd.DataFrame, displacement: float = 0.0) -> Self:
        # Slice and copy.
        self.data = data[["date_time", "WS", "USTAR", "WD", "V_SIGMA", "MO_LENGTH"]].copy()
        
        # Add spec columns.
        self.data["zm"] = self.tower_spec["zm"]
        self.data["z0"] = self.tower_spec["z0"]
        self.data["d"] = displacement

        # Convert to datetime.
        self.data["date_time"] = pd.to_datetime(self.data["date_time"])
        
        # Separate year month day etc.
        self.data["yyyy"] = self.data["date_time"].dt.year
        self.data["mm"] = self.data["date_time"].dt.month
        self.data["day"] = self.data["date_time"].dt.day
        self.data["HH"] = self.data["date_time"].dt.hour
        self.data["MM"] = self.data["date_time"].dt.minute
        
        # Rearrange columns.
        self.data = self.data[["yyyy", "mm", "day", "HH", "MM", "zm", "d", "z0", "WS", "MO_LENGTH", "V_SIGMA", "USTAR", "WD"]]
        
        # Rename columns.
        self.data = self.data.rename(columns={"WS": "u_mean", "MO_LENGTH": "L", "V_SIGMA": "sigma_v", "USTAR": "u_star", "WD": "wind_dir"})

        # Subset to only include data between 9 AM and 3PM.
        self.data = self.data[(self.data["HH"] >= 9) & (self.data["HH"] <= 15)]
        
        self.data = self.data.reset_index()
        
        return self
    
    def draw(self) -> Self:
        if self.data is None:
            raise ValueError("data must be set before drawing.")
        
        # Empty list to hold all footprint polygons.
        polygons = []
        
        for index, row in self.data.iterrows():
            if index >= 300: # type: ignore
                break
            
            try:
                footprint = FPP(
                    zm = row["zm"],
                    z0 = row["z0"],
                    umean = row["u_mean"],
                    h = BOUNDARY_LAYER_HEIGHT,
                    ol = row["L"],
                    sigmav = row["sigma_v"],
                    ustar = row["u_star"],
                    wind_dir = row["wind_dir"],
                    rs = CONTOUR_SRC_PCT,
                    fig = False
                )
                
                xr = np.array(footprint["xr"][0]) + self.easting
                yr = np.array(footprint["yr"][0]) + self.northing
                
                # Create polygon from xr, yr coordinates.
                polygon = Polygon(zip(xr, yr))
                polygons.append(polygon)
            except KeyError as e:
                print(f"KeyError in row {index}: {e}")
            except ValueError as e:
                print(f"ValueError in row {index}: {e}")
            except Exception as e:
                print(f"Error in row {index}: {e}")
        
        # Create GeoDataFrame from all collected polygons at once.
        self.geometry = gpd.GeoDataFrame(geometry = polygons, crs = self.utm_crs) # type: ignore
        
        # Validate an output.
        if not self.geometry.empty:
            minx, miny, maxx, maxy = self.geometry.union_all().bounds
            print(f"Overall extent: minx = {minx}, miny = {miny}, maxx = {maxx}, maxy = {maxy}")
        else:
            print("No polygons were processed.")
        
        return self
    
    def rasterize(self, resolution: int = 1) -> Self:
        if self.geometry is None:
            raise ValueError("geometry must be set before rasterizing.")
        
        minx, miny, maxx, maxy = self.geometry.union_all().bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        
        self.transform = from_origin(minx, maxy, resolution, resolution)
        self.raster = np.zeros((height, width), dtype=np.uint8)
        
        for index, row in self.geometry.iterrows():
            try:
                polygon = [row["geometry"]]
                raster = features.rasterize(polygon, 
                                            out_shape = (height, width), 
                                            transform = self.transform, 
                                            fill = 0)
                self.raster += raster.astype(self.raster.dtype)
                
            except KeyError as e:
                print(f"KeyError in row {index}: {e}")
            except ValueError as e:
                print(f"ValueError in row {index}: {e}")
            except Exception as e:
                print(f"Error in row {index}: {e}")
        
        return self
    
    def polygonize(self, threshold: float = 0.0) -> gpd.GeoDataFrame:
        if self.raster is None:
            raise ValueError("raster must be set before polygonizing.")
        assert self.transform
        
        max_overlaps = np.max(self.raster)
        mask = self.raster > (max_overlaps * threshold)
        
        shapes_gen = shapes(self.raster, mask = mask, transform = self.transform)
        polygons = [shape(geom) for geom, _ in shapes_gen]
        
        combined_polygon = unary_union(polygons)
        
        gdf = gpd.GeoDataFrame(geometry = [combined_polygon], crs = self.utm_crs) # type: ignore
        
        try:
            gdf.loc[0, "geometry"] = taubin_smooth(gdf.loc[0, "geometry"], steps = 50) # type: ignore
        except Exception as e:
            print(f"Error in smoothing polygon: {e}")
        
        return gdf