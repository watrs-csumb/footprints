# Flux Footprints by CSUMB WATRS
# Usage
## Prerequisites
* Python 3.10+
* Powershell (Windows)
  
> [!NOTE]
>If using conda, ensure you are using a terminal with an activated environment.

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

        # For Windows:
        ./Scripts/setup.ps1
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

#### `spatial_resolution`
The spatial resolution to use for the polygon in meters per pixel.

![image](https://github.com/user-attachments/assets/d79db78c-7462-499f-9f1f-e4bdf2224560)


#### `overlap_threshold`
Minimum percentile of max overlaps to be included in footprint . e.g. `0.2` results in overlaps above the 20th percentile to be included.

![image](https://github.com/user-attachments/assets/87c7944e-a60e-463e-9048-31bfdc600b93)


#### `smoothing_factor`
The number of pixel steps to smooth. Higher number results in a smoother shape. *See https://pypi.org/project/shapelysmooth/#taubin*

![image](https://github.com/user-attachments/assets/e6b3666a-2758-4f0c-b9bf-f96d075698fa)


## app.py
This is the script application that generates the footprint and exports:
* A folder containing a shapefile
* A GeoJSON file
* An overlap heatmap centered around the tower
* A polygon figure of the footprint with the source_contour_ratio factored in.
  
### Run app.py
```bash
# To run the script, simply run:
python app.py
```

# Acknowledgements
## Footprint Prediction Calculation
The source code is heavily inspired by the work of Natascha Kljun of Swansea University in Swansea, UK.

### The original License text as is:
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
