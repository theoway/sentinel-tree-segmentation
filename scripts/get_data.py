from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv
import argparse
import os
import json
from datetime import datetime, timedelta
from utils import calcSlope, process_dem

import numpy as np

#gets the bottom left and the top right coords for the bounding box
bbox_id = 1
with open("../data/bbox/bbox_{id}.json".format(id = bbox_id)) as file:
    data = json.load(file)
    
coords = data['geometry']['coordinates'][0]
bl = (min(row[0] for row in coords), min(row[1] for row in coords))
tr = (max(row[0] for row in coords), max(row[1] for row in coords))

load_dotenv()

config = SHConfig()
config.sh_client_id = os.getenv("CLIENT_ID")
config.sh_client_secret = os.getenv("CLIENT_SECRET")

parser = argparse.ArgumentParser(description = "Date")
parser.add_argument("-d", "--date", type = str, required = True, help = "The date in YYYY-MM-DD format")
args = parser.parse_args()

dt = [int(d) for d in args.date.split('-')]
date = datetime(*dt)
start_date = date - timedelta(days = 60)
end_date = date + timedelta(days = 60)
bbox = BBox((bl, tr), crs=CRS.WGS84)

size = (512, 512)
time_interval = start_date, end_date
data_folder = "../sat_imgs"

evalscript_true_color ="""//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12"],
        }],
        output: {
            bands: 10
        }
    };
}

function bilinearInterpolation(value, fromResolution, toResolution) {
    // Perform bilinear interpolation manually
    return value * (fromResolution / toResolution);
}

function evaluatePixel(sample) {
    return [
        bilinearInterpolation(sample.B02, 10, 10),
        bilinearInterpolation(sample.B03, 10, 10),
        bilinearInterpolation(sample.B04, 10, 10),
        bilinearInterpolation(sample.B05, 20, 10),
        bilinearInterpolation(sample.B06, 20, 10),
        bilinearInterpolation(sample.B07, 20, 10),
        bilinearInterpolation(sample.B8A, 20, 10),
        bilinearInterpolation(sample.B09, 20, 10),
        bilinearInterpolation(sample.B11, 20, 10),
        bilinearInterpolation(sample.B12, 20, 10),
    ];
}"""

request = SentinelHubRequest(
    data_folder = data_folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="leastCC"
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=size,
    config=config,
)

evalscript_dem = """
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

request_dem = SentinelHubRequest(
    data_folder = data_folder,
    evalscript=evalscript_dem,
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


path = os.path.join(data_folder, str(bbox_id))
if not os.path.isdir(path):
    os.mkdir(path)
print("Downloading L2A...")
# image = request.get_data(save_data = True, show_progress=True)
image = request.get_data()[0]
np.save(path + "/L2A", image)
print("Downloading DEM...")
# image_dem = request_dem.get_data(save_data = True, show_progress = True)
image_dem = request_dem.get_data()[0]
image_dem = process_dem(image_dem)
np.save(path + "/DEM", image_dem)
