from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv
import argparse
import os
import json
from datetime import datetime, timedelta
from utils import calcSlope, process_dem
from tqdm import tqdm
import numpy as np

#gets the bottom left and the top right coords for the bounding box
with open("../data/bbox_geojson/bounding_boxes.geojson") as file:
    data = json.load(file)

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

parser = argparse.ArgumentParser(description = "Date")
parser.add_argument("-d", "--date", type = str, required = True, help = "The date in YYYY-MM-DD format")
args = parser.parse_args()

dt = [int(d) for d in args.date.split('-')]
date = datetime(*dt)
start_date = date - timedelta(days = 60)
end_date = date + timedelta(days = 60)

time_interval = start_date, end_date
data_folder = "../data/sat_imgs"
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
    

print("Downloading Sentinel 2 Bands...")

for i in tqdm(len(data['features'])):
    coords = data['features'][i]['geometry']['coordinates'][0]
    bl = (min(row[0] for row in coords), min(row[1] for row in coords))
    tr = (max(row[0] for row in coords), max(row[1] for row in coords))
    size = (data['features'][i]['properties']['width'], data['features'][i]['properties']['height'])
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

print("Saved to " + os.path.abspath(data_folder))

    
