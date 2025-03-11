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
A relative path to the input file. 
```toml
# Example (relative)
file = "./mydata.csv"
# Example (absolute)
file = "/Users/you/Downloads/data.csv"
```
> [!NOTE]
> If your data file is elsewhere, copy the path and paste into this field.

#### `location`
A tuple of latitude and longitude of tower location. This is used in the transformation matrix for the footprint geometry. Uses EPSG:4326.

#### `hemisphere`
The hemisphere that the tower is located. For US users, this can be left alone.

#### `boundar_layer_height`
Estimated height of the boundary layer in meters.

#### `source_contour_ratio`
Percentage of source area for which to provide contours, must be between 10% and 90%
* Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
* Expressed either in percentages ("80") or as fractions of 1 ("0.8"). 

#### `tower_spec.zm`
Measurement height above displacement height in meters.

#### `tower_spec.z0`
Roughless length in meters

#### `tower_spec.d`
Displacement height in meters

### Output
#### `output_dir`
Directory to place the output files in.

#### `spatial_resolution`
The spatial resolution to use for the polygon in meters per pixel.

#### `overlap_threshold`
Percentage adjustment for number of overlaps needed for data to contribute to footprint. Must be between 0 and 1.

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