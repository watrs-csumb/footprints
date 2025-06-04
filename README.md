# Flux Footprints by CSUMB WATRS

## Prerequisites

* Python 3.10+
* Powershell (Windows)
  
> [!NOTE]
> If using conda, ensure you are using a terminal with an activated environment.

> [!CAUTION]
> For Windows users, the dependency 'shapelysmooth' requires Microsoft Visual C++ 14.0 or greater. You can download the [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools) to install this.

## Installation

Use a terminal to follow the steps below to get setup.

1. Clone the repository

    ```bash
    git clone https://github.com/aetriusgx/footprints.git
    ```

2. Open folder

    ```bash
    cd footprints
    ```

3. Activate a virtual environment and install dependencies
    1. (Method 1) Run the setup script

        ```bash
        # For Linux/Unix:
        source ./Scripts/setup.sh

        # For Windows Powershell:
        ./Scripts/setup.ps1

        # For Windows Terminal (experimental):
        ./Scripts/setup.bat
        ```

    2. (Method 2) Manually start and install

        ```bash
        # Start the virtual environment
        python -m venv .
        source bin/activate # <- Linux/Unix
        .\Scripts\Activate.ps1 # <- Windows

        # Update pip, if necessary
        python -m pip --upgrade pip
        
        # Install dependencies
        pip install -r requirements.txt
        ```

## app.toml

This is the configuration file and should be the only file you modify. The configuration schema is explained below.

### Input

#### `file`

Path to the input file. \
Input data must contain the following columns (case-sensitive): \
["date_time", "WS", "USTAR", "WD", "V_SIGMA", "MO_LENGTH", "instr_height_m", "canopy_height_m", "Z0_roughness"]

```toml
# Example (relative)
file = "./mydata.csv"
# Example (absolute)
file = "/Users/you/Downloads/data.csv"
```

> [!NOTE]
> If your data file is elsewhere, copy the path and paste into this field.
>
> `\` characters are replaced with `/`

#### `location`

A tuple of latitude and longitude of tower location. This is used in the transformation matrix for the footprint geometry. Uses EPSG:4326.

#### `weigh_by_eto`

Enables weighing the footprint data by reference ETo data. This data is assumed to be a separate file that must be provided in `eto_file`.

#### `eto_file`

Path to the reference ETo data. This field is ignored if `weigh_by_eto` is `false`.

#### `boundar_layer_height`

Estimated height of the boundary layer in meters.

#### `source_contour_ratio`

Percentage of source area for which to provide contours, must be between 10% and 90%

* Affects daily footprint calculation.
* Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
* Expressed either in percentages ("80") or as fractions of 1 ("0.8").

#### `max_rows`

Specify the amount of rows that will be read from input data. Use `-1` to read all rows.

### Output

#### `output_dir`

Directory/Folder to place the output files in.

#### `spatial_resolution` [m]

The spatial resolution (meters) to use for the polygon in meters per pixel.

![image](https://github.com/user-attachments/assets/d79db78c-7462-499f-9f1f-e4bdf2224560)

#### `overlap_threshold`[valid values 0 - 1.0]

Minimum percentile of footprint overlaps to be included in a time-integrated footprint. A value of `0.2' means cells included in generation of the output GeoJSON footprint were included in 20% of all daily footprints.

![image](https://github.com/user-attachments/assets/87c7944e-a60e-463e-9048-31bfdc600b93)

#### `smoothing_factor`

The number of pixel steps to smooth the output geometry. Higher number results in a smoother shape. *See <https://pypi.org/project/shapelysmooth/#taubin>*

![image](https://github.com/user-attachments/assets/e6b3666a-2758-4f0c-b9bf-f96d075698fa)

#### `coverage_union`

If the resulting footprint results in disjointed polygons, uses best-effort approach to combine disjointed polygons.

### `heatmap`

Whether to export a heatmap image showing the cummulative contribution.

### `polygon`

Whether to export a chart displaying the footprint polygon.

## app.py

This is the script application that generates the footprint and exports:

* A folder containing a shapefile
* A GeoJSON file with the geometry of the time-integrated footprint polygon.
* A heatmap image centered around the tower. Values representing the % of footprints covering a pixel.
* A figure of the footprint polygon centered around the tower.
  
### Run app.py

```bash
# To run the script, simply run:
python app.py
```

### Outputs

* Shapefile containing footprint polygon
* Raster GeoTiFF containing 3 bands:
  * Band 1: Normalized dataset
  * Band 2: Polygon Mask
  * Band 3: Raw dataset containing cummulative overlaps
* Daily timeseries GeoJSON containing two columns:
  * Time: YYYY-MM-DD
  * Geometry: Footprint polygon for each timestep
* GeoJSON of footprint polygon
* Any specified charts from the [graphs] options

## Acknowledgements

### Footprint Prediction Calculation

The footprint scripts were developed from work of Natascha Kljun of Swansea University in Swansea, UK. Please see [Kljun Flux Footprint Prediction](https://footprint.kljun.net/) model for more information on the methods.

#### The original License text as is
>
> Copyright (c) 2015 - 2024 Natascha Kljun
>
> Permission to use, copy, modify, and/or distribute this software for any
> purpose with or without fee is hereby granted, provided that the above
> copyright notice and this permission notice appear in all copies.
>
> THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
> WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
> MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
> ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
> WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
> ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
> OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
