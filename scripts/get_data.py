from sentinelhub import SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv
import argparse
import os
import json
from datetime import datetime, timedelta

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

parser = argparse.ArgumentParser(description = "Date")
parser.add_argument("-d", "--date", type = str, required = True, help = "The date in YYYY-MM-DD format")
args = parser.parse_args()

dt = [int(d) for d in args.date.split('-')]
date = datetime(*dt)
start_date = date - timedelta(days = 60)
end_date = date + timedelta(days = 60)
bbox = BBox((bl, tr), crs=CRS.WGS84)

size = (700, 466)
time_interval = start_date, end_date
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