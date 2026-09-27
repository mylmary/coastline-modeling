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


Load the data
DATASETS
```
!pip install -q rasterio

import numpy as np
import rasterio

# Direct path to your Mean Elevation file on Kaggle

import os
import glob

# 1. This automatically finds any file ending with 'mea075.tif' inside the input directory
search_pattern = "/kaggle/input/**/*mea075.tif"
found_files = glob.glob(search_pattern, recursive=True)

if not found_files:
    raise FileNotFoundError("Could not find the file. Double check if the dataset is attached correctly to the notebook.")

file_path = found_files[0]
print(f"Success! Found exact file path: {file_path}")

# 2. Proceed with loading the data safely
with rasterio.open(file_path) as src:
    elevation_grid = src.read(1)
    transform = src.transform

print("Elevation grid loaded successfully! Shape:", elevation_grid.shape)


with rasterio.open(file_path) as src:
    elevation_grid = src.read(1)
    transform = src.transform

# Generate coordinates matching your mesh nodes
rows, cols = elevation_grid.shape
cols_idx, rows_idx = np.meshgrid(np.arange(cols), np.arange(rows))
longitudes, latitudes = rasterio.transform.xy(transform, rows_idx, cols_idx)

# Flatten into the (N, 3) unstructured input shape PhysicsNeMo expects
inputs = np.stack([
    np.array(longitudes).flatten(), 
    np.array(latitudes).flatten(), 
    elevation_grid.flatten()
], axis=-1)

# Save to your working directory
np.save("/kaggle/working/coastline_inputs.npy", inputs)
print("PhysicsNeMo Inputs Ready! Data Array Shape:", inputs.shape)

```


Transolver training
```
# ============================================================
# TRANSOLVER++ — TRAINING
# ============================================================

!pip install -q physicsnemo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from physicsnemo.models.transolver import Transolver


# ------------------------------------------------------------
# 1. Use GPU
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# 2. Reduce number of points for first experiment
# ------------------------------------------------------------

# Your full terrain:
# 21,056 points
#
# Start with 5,000 points.
# Later we can increase this.

N_POINTS = 5000

total_points = X_tensor.shape[1]

torch.manual_seed(42)

point_idx = torch.randperm(total_points)[:N_POINTS]

X_train = X_tensor[:, point_idx, :]
Y_train = Y_tensor[:, point_idx, :]

print("Training input :", X_train.shape)
print("Training target:", Y_train.shape)


# ------------------------------------------------------------
# 3. Train/validation split
# ------------------------------------------------------------

n_cases = X_train.shape[0]

split = int(0.8 * n_cases)

train_x = X_train[:split]
train_y = Y_train[:split]

val_x = X_train[split:]
val_y = Y_train[split:]

print("Training cases  :", train_x.shape[0])
print("Validation cases:", val_x.shape[0])

# ------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------

class CoastalDataset(Dataset):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


train_dataset = CoastalDataset(
    train_x,
    train_y
)

val_dataset = CoastalDataset(
    val_x,
    val_y
)


train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)


# ------------------------------------------------------------
# 5. Initialize Transolver++
# ------------------------------------------------------------
 #Instantiate the model from your compiled blueprint handle
#{model = Model(
    #space_dim=3,        # Coordinates: [Longitude, Latitude, Depth]
    #fun_dim=1,          # Terrain/Bathymetry parameter
    #out_dim=3,          # Target physical flow fields to solve: [fluid_height, u_velocity, v_velocity]
    #n_layers=4,         # Depth of layer attention blocks
    #n_hidden=128,       # Channel width representation scale
    #n_head=4,           # Parallel attention heads
    #mlp_ratio=2,        # Hidden feed-forward expansion step scale
    #slice_num=32,       # Physics attention slice counts
    #unified_pos=False   # Keeps mesh coordinates distinct from feature metrics
#).to(device)

model = Transolver(
    space_dim=2,
    n_layers=4,
    n_hidden=128,
    n_head=8,
    dropout=0.0,
    unified_pos=True,
    fun_dim=6,
    out_dim=3,
    slice_num=32,
    mlp_ratio=1,
    H=32,
    L=32,
    attention_type="galerkin",
    plus=True
).to(device)

print(model)


```

