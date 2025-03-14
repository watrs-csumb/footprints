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
    _req_columns = ["date_time", "WS", "USTAR", "WD", "V_SIGMA", "MO_LENGTH", "instr_height_m", "canopy_height_m"]
    
    def __init__(self, 
                tower_location: tuple[float, float],
                tower_spec: dict[str, float | int],
                hemisphere: str = "North"):
        """
        Initialize the Footprint object.

        Parameters
        ----------
        tower_location : tuple[float, float]
            The latitude and longitude of the tower location.
        tower_spec : dict
            A dictionary with the following required keys:
                - zm: The height of the measurement in meters.
                - z0 or umean: The surface roughness length in meters or the mean wind speed in meters per second.
        hemisphere : str, optional
            The hemisphere of the tower location, either 'North' or 'South'. Default is 'North'.

        Notes
        -----
        The Footprint object is initialized with the tower location and tower specification.
        The UTM coordinates of the tower location are calculated and stored as easting and northing.
        """
        self.latitude = tower_location[0]
        self.longitude = tower_location[1]
        self.hemisphere = hemisphere
        self._ws_limit_quantile = 0.95
        
        self.utm_crs: str = ""
        self.easting: float = 0.0
        self.northing: float = 0.0
        self.raster: np.ndarray | None = None
        self.data: pd.DataFrame | None = None
        self.geometry: gpd.GeoDataFrame | None = None
        self.transform: Affine | None = None
        
        self._to_utm()
        
        if "z0" not in tower_spec.keys() and "umean" not in tower_spec.keys():
            raise ValueError("tower_spec must contain either 'z0' or 'umean'.")
        
        self.tower_spec = tower_spec
    
    @property
    def ws_limit_quantile(self) -> float:
        return self._ws_limit_quantile
    
    @ws_limit_quantile.setter
    def ws_limit_quantile(self, value: float):
        self._ws_limit_quantile = value
    
    @property
    def boundary_layer_height(self) -> float:
        return self.__dict__.get("_boundary_layer_height") or BOUNDARY_LAYER_HEIGHT
    
    @boundary_layer_height.setter
    def boundary_layer_height(self, value: float):
        self._boundary_layer_height = value
    
    @property
    def contour_src_pct(self):
        return self.__dict__.get("_contour_src_pct") or CONTOUR_SRC_PCT
    
    @contour_src_pct.setter
    def contour_src_pct(self, value: float | list[float | int]):
        self._contour_src_pct = value
    
    def __repr__(self):
        if self.geometry is not None:
            return f"Footprint with {len(self.geometry)} polygons.\n{self.geometry.head(1)}"

        return f"Footprint for latitude: {self.latitude}, longitude: {self.longitude}"

    def _to_utm(self):
        utm_zone = int((self.longitude + 180) / 6) + 1  # calculate UTM zone based on longitude
        self.utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"  # create UTM CRS
        
        transformer = Transformer.from_crs("epsg:4326", self.utm_crs)
        self.easting, self.northing = transformer.transform(self.latitude, self.longitude)
    
    def attach(self, data: pd.DataFrame) -> Self:
        """
        Attach a pandas DataFrame to the Footprint object.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing the required columns.

        Returns
        -------
        Self
            The Footprint object with the attached data.

        Notes
        -----
        The required columns are:
        - date_time: A datetime column.
        - instr_height_m: The instrument height in meters.
        - canopy_height_m: The canopy height in meters (used as displacement height).
        - WS: The mean wind speed in meters per second.
        - USTAR: The friction velocity in meters per second.
        - WD: The wind direction in degrees.
        - V_SIGMA: The lateral velocity in meters per second.
        - MO_LENGTH: The Obukhov length in meters.

        The `attach` method will subset the data to only include data between 9 AM and 3PM.
        """
        self.data = data[Footprint._req_columns].copy()
        
        # Filter WS (wind speed) so outliers are removed.
        wind_lim = self.data["WS"].quantile(self._ws_limit_quantile)
        self.data["WS"] = self.data["WS"].clip(upper=wind_lim)

        self.data["z0"] = self.tower_spec["z0"]

        # Convert to datetime.
        self.data["date_time"] = pd.to_datetime(self.data["date_time"])
        
        # Separate year month day etc.
        self.data["yyyy"] = self.data["date_time"].dt.year
        self.data["mm"] = self.data["date_time"].dt.month
        self.data["day"] = self.data["date_time"].dt.day
        self.data["HH"] = self.data["date_time"].dt.hour
        self.data["MM"] = self.data["date_time"].dt.minute
        
        # Rearrange columns.
        self.data = self.data[["yyyy", "mm", "day", "HH", "MM", "instr_height_m", "canopy_height_m", "z0", "WS", "MO_LENGTH", "V_SIGMA", "USTAR", "WD"]]
        
        # Rename columns.
        self.data = self.data.rename(
            columns={"instr_height_m": "zm", 
                     "canopy_height_m": "d",
                     "WS": "u_mean", 
                     "MO_LENGTH": "L", 
                     "V_SIGMA": "sigma_v",
                     "USTAR": "u_star", 

                     "WD": "wind_dir"})

        # Subset to only include data between 9 AM and 3PM.
        self.data = self.data[(self.data["HH"] >= 9) & (self.data["HH"] <= 15)]
        
        self.data = self.data.reset_index()
        
        return self
    
    def draw(self) -> Self:
        """
        Generate footprint polygons from the attached data.

        This method processes the attached DataFrame to calculate footprint polygons
        based on the specified parameters for each row of data. It uses the Footprint 
        Model (FPP) to compute the footprint coordinates, creates polygons, and stores 
        them in a GeoDataFrame.

        Returns
        -------
        Self
            The Footprint object with the computed footprint polygons.

        Raises
        ------
        ValueError
            If the data is not set before calling the method.

        Notes
        -----
        The processing is limited to the first 300 rows for memory reasons. The 
        footprint polygons are created using the UTM coordinates of the tower and 
        stored in the `geometry` attribute as a GeoDataFrame. The overall extent of 
        the computed geometry is printed if polygons are successfully processed.
        """

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
                    h = self.boundary_layer_height,
                    ol = row["L"],
                    sigmav = row["sigma_v"],
                    ustar = row["u_star"],
                    wind_dir = row["wind_dir"],
                    rs = self.contour_src_pct,
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
        """
        Rasterize the footprint polygons to a numpy array.

        Parameters
        ----------
        resolution : int
            The desired resolution of the output raster in meters per pixel.
            Default is 1.

        Returns
        -------
        Self
            The Footprint object with the rasterized footprint polygons.

        Raises
        ------
        ValueError
            If the geometry is not set before calling the method.

        Notes
        -----
        The raster is created by accumulating the rasterized footprint polygons
        from the GeoDataFrame `geometry`. The output raster has a shape based
        on the bounds of the geometry and is stored in the `raster` attribute. 
        The transformation matrix is stored in the `transform` attribute.
        """
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
    
    def polygonize(self, threshold: float = 1.0) -> gpd.GeoDataFrame:
        """
        Create a single polygon from rasters that meet overlap threshold.

        Parameters
        ----------
        threshold : float, optional
            The threshold for the overlap count to be considered a valid polygon.
            Defaults to 0.0.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing a single polygon that meets the overlap
            threshold, with a single row and column named 'geometry'.

        Raises
        ------
        ValueError
            If the raster is not set before calling the method.

        Notes
        -----
        The polygon is created by masking the raster with the given threshold,
        and then creating a single polygon from the resulting shapes. The
        polygon is then smoothed to reduce the number of vertices. The
        GeoDataFrame is created with a single row and column named 'geometry'.
        """
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