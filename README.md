# sentinel-tree-segmentation
Code for downloading, pre-processing sentinel images to train model for tree segmentation.

## Development
### Setup
- Sentinel Hub credentials can be found in `.env`

- Linux based OS recommended (Ubunut 20 above), else use WSL (if on windows)
```bash

# Install GDAL in system
$   apt-get update && apt-get install -y libgdal-dev gdal-bin

# install gdal python package
$   pip install GDAL==$(gdalinfo --version)

# Install python dependencies (you can add more according to your work). 
$   pip install -r requirements.txt

```

## Running the scripts
### 1) Generating Fixed Size Square Bounding Boxes
- `meters_to_degrees(meters, latitude)` function: Converts distance in meters to degrees latitude. It uses the `geopy.distance.geodesic` function to calculate the distance per degree at a given latitude.

- `generate_bounding_boxes(tiff_path, box_size)` function: Generates fixed-sized square bounding boxes inside a TIFF file. It reads the TIFF file using `rasterio`, calculates the bounds, and then generates bounding boxes based on the specified `box_size`. It checks if each box is within the TIFF bounds and if more than 50% of the pixels in the box are non-zero. It returns a list of Shapely Polygon objects representing the bounding boxes and a list of corresponding image arrays.

- Bounding Box Generation Loop: Iterates over each TIFF file in the specified directory (`tif_labels_path`). For each TIFF file, it calls the `generate_bounding_boxes` function to obtain bounding boxes and image arrays then saves the bounding boxes as a GeoJSON file (`bounding_boxes.geojson`) while also adding the dimension of the box as an attribute, using geopandas and saves the image arrays as NumPy files in the `labels_folder`.

```bash
# generating square bboxes
$  python generate_square_bbox.py -p [path to the tif labels directory]
```

### 2) Downloading SentinelHub Images
- `getL2A()`: Defines an evaluation script for obtaining Sentinel-2 Level-2A data. It uses true color bands and cloud mask to filter out cloudy pixels. The request is made using the SentinelHubRequest class from the sentinelhub library.

- `getDEM()`: Defines an evaluation script for obtaining Digital Elevation Model (DEM) data. The request is made using the SentinelHubRequest class.

- This script uses a GeoJSON file (`../data/bbox_geojson/bounding_boxes.geojson`) containing bounding boxes and their properties. It reads the GeoJSON file to obtain coordinates, width, and height for each bounding box.

- Iterates over the bounding boxes. For each bounding box, it extracts coordinates, creates a bounding box, and makes requests for Sentinel-2 L2A bands (`getL2A()`) and DEM (`getDEM()`).
- It calculates the Normalized Difference Vegetation Index (NDVI) from the downloaded bands.
Finally concatenates the original Sentinel-2 bands, DEM, and NDVI to create a final image array and saves them as a NumPy file in the specified data folder (`../data/sat_imgs`).

```bash
# downloading data
$  python get_data.py -d YYYY-MM-DD
```
### 3) Training
- `UNet` Pytorch nn.Module Class : Basic U-Net architecture with a simple encoder-decoder structure. The encoder downsamples the input, and the decoder upsamples it.

- `CustomDataset` is a custom dataset class for loading images and corresponding masks.
It performs preprocessing such as normalization and resizing.
using interpolation

- Images and corresponding binary masks are loaded using the CustomDataset class. The dataset is split into 70% training and 30% testing sets. 

- Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`) is used as the loss function. Adam optimizer is used for training.

- The script then enters the training loop for the specified number of epochs. For each epoch, it iterates through the training set, computes loss, and performs backpropagation.
After each epoch, the model is evaluated on the validation set. Metrics are also saved to a CSV file (`training_metrics.csv`) for later analysis.

- After training, the model's state dictionary is saved to a file (`unet_model.pth`) in the current directory.

```bash
# training
$  python train.py
```

### 4) Prediction
- Loads the trained model and predicts on square bounding boxes extracted from a geojson file and saves the predictions in `preds/labels` in `tif` format

```bash
# prediction
$  python predict.py -d YYYY-MM-DD -p [path to the geojson file]
```

### Resources
- Max size of bbox for requests [link](https://docs.sentinel-hub.com/api/latest/api/overview/processing-unit/).

