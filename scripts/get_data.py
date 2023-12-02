from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv
import argparse
import os
import json

#gets the bottom left and the top right coords for the bounding box
with open("../data/bbox/bbox_1.json") as file:
    data = json.load(file)
coords = data['geometry']['coordinates'][0]
bl = (min(row[0] for row in coords), min(row[1] for row in coords))
tr = (max(row[0] for row in coords), max(row[1] for row in coords))

load_dotenv()

config = SHConfig()
config.sh_client_id = os.getenv("CLIENT_ID")
config.sh_client_secret = os.getenv("CLIENT_SECRET")

parser = argparse.ArgumentParser(description = "Time Interval")
parser.add_argument("-s", "--start", type = str, required = True, help = "The start date in YYYY-MM-DD format")
parser.add_argument("-e", "--end", type = str, required = True, help = "The end date in YYYY-MM-DD format")

args = parser.parse_args()

bbox = BBox((bl, tr), crs=CRS.WGS84)
size = (700, 466)
time_interval = args.start, args.end
data_folder = "../sat_imgs"

evalscript_true_color = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04"]
        }],
        output: {
            bands: 3
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""

request = SentinelHubRequest(
    data_folder = data_folder,
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=bbox,
    size=size,
    config=config,
)

image = request.get_data(save_data = True, show_progress=True)