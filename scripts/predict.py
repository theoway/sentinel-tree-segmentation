import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
from scipy.ndimage import zoom
from tqdm import tqdm
import os
import re

from shapely.geometry import box
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.ops import unary_union
from fiona.crs import from_epsg
from geopy.distance import geodesic
import pandas as pd
import argparse

from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import geojson

parser = argparse.ArgumentParser(description = "Date")
parser.add_argument("-d", "--date", type = str, required = True, help = "The date in YYYY-MM-DD format")
parser.add_argument("-p", "--path", type = str, required = True, help = "Path to GeoJSON file")
args = parser.parse_args()
path = args.path
dt = [int(d) for d in args.date.split('-')]
date = datetime(*dt)
start_date = date - timedelta(days = 60)
end_date = date + timedelta(days = 60)

time_interval = start_date, end_date

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = UNet(13, 1)
model.load_state_dict(torch.load("unet_model.pth"))


#generate square boxes from geojson

def meters_to_degrees(meters, latitude):
    # Calculate the distance per degree at the specified latitude
    meters_per_degree = geodesic((latitude, 0), (latitude + 1, 0)).meters

    # Convert meters to degrees
    degrees = meters / meters_per_degree

    return degrees

def get_bounds(geometry):
    coords = np.array(list(geojson.utils.coords(geometry)))
    return coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max()

def generate_bounding_boxes(geojson_path, box_size):
    """
    Generate fixed-sized square bounding boxes inside the polygons of a GeoJSON file.

    Parameters:
    - geojson_path: Path to the GeoJSON file
    - box_size: Size of the square bounding boxes

    Returns:
    - List of Shapely Polygon objects representing the bounding boxes
    """
    # Open the GeoJSON file and get its polygons
    bounding_boxes = []
    

    gdf = gpd.read_file(geojson_path)
    
    for i, polygon in enumerate(gdf.geometry):
        avg_lat = polygon.centroid.y
        box_size_deg = meters_to_degrees(box_size, avg_lat)

        # Calculate the bounding box
        bounds = polygon.bounds
        num_boxes_x = int((bounds[2] - bounds[0]) // box_size_deg)
        num_boxes_y = int((bounds[3] - bounds[1]) // box_size_deg)
        
        for j in range(num_boxes_x):
            for k in range(num_boxes_y):
                min_x = bounds[0] + j * box_size_deg
                min_y = bounds[1] + k * box_size_deg
                max_x = min_x + box_size_deg
                max_y = min_y + box_size_deg

                box_geometry = box(min_x, min_y, max_x, max_y)

                # Check if the box is completely within the polygon
                if polygon.contains(box_geometry):
                    
                    bounding_boxes.append(box_geometry)

    return bounding_boxes

path = "../data/bbox/bbox_1.json"
bounding_boxes = generate_bounding_boxes(path, 1000)
target_crs = from_epsg(4326)    
gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs=target_crs)
gdf.to_file('bounding_boxes_for_preds.geojson', driver='GeoJSON') 

size = (512, 512)
load_dotenv()

config = SHConfig()
config.sh_client_id = os.getenv("CLIENT_ID")
config.sh_client_secret = os.getenv("CLIENT_SECRET")

def getL2A():
    evalscript_true_color ="""//VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "CLM", "dataMask"],
            }],
            output: {
                bands: 11
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask != 1) {
            return [99, 99, 99]
        }
        return [
            2.5 * sample.B02,
            2.5 * sample.B03,
            2.5 * sample.B04,
            2.5 * sample.B05,
            2.5 * sample.B06,
            2.5 * sample.B07,
            2.5 * sample.B8A,
            2.5 * sample.B09,
            2.5 * sample.B11,
            2.5 * sample.B12,
            sample.CLM,
        ];
    }"""


    request = SentinelHubRequest(
        data_folder = data_folder,
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC",
    
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    image = request.get_data()[0]
    
    return image

def getDEM():
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["DEM"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32" 
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.DEM];
    }
    """

    request = SentinelHubRequest(
        data_folder = data_folder,
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM,
                time_interval=time_interval,
                mosaicking_order="leastCC"

            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    dem = request.get_data()[0]
    return dem

with open("bounding_boxes_for_preds.geojson") as file:
    data = json.load(file)
    
print("Downloading Sentinel 2 Bands...")
root = "../preds"
if not os.path.isdir(root):
    os.mkdir(root)
data_folder = "../preds/bboxes"
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for i in tqdm(range(len(data['features']))):
    coords = data['features'][i]['geometry']['coordinates'][0]
    bl = (min(row[0] for row in coords), min(row[1] for row in coords))
    tr = (max(row[0] for row in coords), max(row[1] for row in coords))
    
    bbox = BBox((bl, tr), crs=CRS.WGS84)  
    
    image = getL2A()
    dem = getDEM()
    sub = (image[..., 3] - image[..., 2]).astype(float)
    add = (image[..., 3] + image[..., 2]).astype(float)
    ndvi = np.divide(sub, add, out = np.zeros_like(add) - 1, where = add != 0)
    ndvi = np.clip(ndvi, -1, 1)
    dem3 = dem[:, :, np.newaxis]
    ndvi3 = ndvi[:, :, np.newaxis]
    
    final_image = np.concatenate((image, dem3, ndvi3), axis = -1)
    np.save(data_folder + "/{iid}.npy".format(iid = i), final_image)
    
preds_bboxes_path = '../preds/bboxes/'
preds_bboxes_npy_path = [os.path.join(preds_bboxes_path, file) for file in os.listdir(preds_bboxes_path) if file.endswith(".npy")]
preds_bboxes_npy_path = sorted(preds_bboxes_npy_path)

def interpolate(array, target_size):
    zoom_factors = [target_size[i] / array.shape[i] for i in range(2)]
    zoom_factors.append(1)

    interpolated_array = zoom(array, zoom_factors, order=1, mode='nearest')

    return interpolated_array

model.eval()


data_folder = "../preds/labels"
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

def save_as_tif(array, output_path, metadata):
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(array, 1)    
    
for i, path in enumerate(preds_bboxes_npy_path):    
    im = np.load(path)
    im = im.astype(np.float32)
    im = interpolate(im, (512, 512))
    im[..., 0 : 3] /= 255
    im = np.reshape(im, (1, 13, 512, 512))
    im = torch.from_numpy(im)
    with torch.no_grad():
        predicted_output = model(im)

    res = predicted_output.numpy()
    res = np.reshape(res, (512, 512))
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': 'float32',
        'width': res.shape[1],
        'height': res.shape[0],
        'crs': 'EPSG:4326',  
        'transform': from_origin(0, 0, 1, 1),  
    }
    save_as_tif(res, f"../preds/labels/{i}.tif", metadata)   

print()    
print("Saved the predictions to " + data_folder)     