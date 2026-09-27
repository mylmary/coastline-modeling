```import os
import zipfile

zip_path = "/kaggle/input/coastline-data/GMTED2010S10E030_075.zip"
extract_dir = "/kaggle/working/gmted"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_file:
    zip_file.extractall(extract_dir)

print("Extraction complete.")

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        print(os.path.join(root, file))
```


This extracts the terrain data into /kaggle/working/gmted. We then need to identify the actual elevation raster, because GMTED is a terrain-elevation dataset, not a CFD dataset. We are going to use that elevation as the geometry that Transolver sees.

```import glob

tif_files = glob.glob(
    "/kaggle/working/gmted/**/*.tif",
    recursive=True
)

print("GeoTIFF files found:")

for file in tif_files:
    print(file)
```

If the output shows your .tif file, we can load it using rasterio. This gives us the elevation at each location in the terrain.

```
!pip install -q rasterio
import rasterio
import numpy as np

raster_path = tif_files[0]

with rasterio.open(raster_path) as src:
    elevation = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

print("Original terrain shape:", elevation.shape)
print("Coordinate system:", crs)
print(
    "Elevation range:",
    np.nanmin(elevation),
    "to",
    np.nanmax(elevation)
)
```
At this point we have the actual terrain. However, GMTED can contain a very large number of spatial points, so we should not immediately give every point to Transolver. We first reduce the raster to a manageable resolution while preserving the overall coastline/topography.
```
!pip install -q scikit-image
from skimage.transform import resize

target_shape = (128, 256)

elevation_small = resize(
    elevation,
    target_shape,
    order=1,
    preserve_range=True,
    anti_aliasing=True
).astype(np.float32)

print("Reduced terrain shape:", elevation_small.shape)
print("Number of spatial points:", elevation_small.size)
```
Now we have 128 × 256 = 32,768 spatial points. Each point will eventually contain its location and elevation. We create normalized x and y coordinates so that Transolver can work with the spatial domain.
```
ny, nx = elevation_small.shape

x = np.linspace(0, 1, nx, dtype=np.float32)
y = np.linspace(0, 1, ny, dtype=np.float32)

X, Y = np.meshgrid(x, y)

coordinates = np.stack(
    [
        X.reshape(-1),
        Y.reshape(-1),
        elevation_small.reshape(-1)
    ],
    axis=1
)

print("Coordinate array shape:", coordinates.shape)

So each spatial point now has the form:

[x, y, elevation]
```
 We don't need to give Transolver every individual wave as an input. We can describe the environmental state using variables such as wind speed, wind direction, water level, season, tide, and eventually storm/event information. The exact variables will depend on what is available alongside your radar observations.

For example, you can initially define several environmental conditions like this:
```
conditions = np.array([
    [5.0,  90.0, 1.0, 1],
    [8.0,  90.0, 1.0, 1],
    [12.0, 90.0, 1.0, 1],

    [5.0, 180.0, 1.0, 2],
    [8.0, 180.0, 1.0, 2],
    [12.0, 180.0, 1.0, 2],
], dtype=np.float32)

print("Environmental conditions shape:", conditions.shape)
```
Here the four columns could represent:

[wind speed, wind direction, water level, season]

These numbers are only placeholders for now. You should eventually replace them with the actual environmental information associated with each radar observation.

Each environmental condition applies to the entire spatial domain, so we repeat the condition for every terrain point.
```
n_cases = conditions.shape[0]
n_points = coordinates.shape[0]

geometry_features = np.repeat(
    coordinates[None, :, :],
    n_cases,
    axis=0
)

condition_features = np.repeat(
    conditions[:, None, :],
    n_points,
    axis=1
)

print("Geometry:", geometry_features.shape)
print("Conditions:", condition_features.shape)
```
Conceptually, one training example now looks like:
```
                    ┌── x
Terrain geometry ───┼── y
                    └── elevation

                    ┌── wind speed
Environmental ──────┼── wind direction
conditions           ├── water level
                    └── season

                         ↓

                    Transolver++

                         ↓

                water movement field
```
The missing part is the target. We require u, v, or water-depth values just to make the model train. If we're using radar, the radar observations should provide the measured water motion that Transolver learns to reproduce.


