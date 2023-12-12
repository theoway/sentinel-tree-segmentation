from shapely.geometry import box
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.ops import unary_union
from fiona.crs import from_epsg
import os
import numpy as np
from geopy.distance import geodesic
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description = "Tiff Labels Path")
parser.add_argument("-p", "--path", type = str, required = True, help = "The path to the Tiff Labels Directory")
args = parser.parse_args()

tiff_path = args.path
box_size = 5120

tif_labels_path = [os.path.join(tiff_path, file) for file in os.listdir(tiff_path) if file.endswith(".tif")]

def meters_to_degrees(meters, latitude):
    # Calculate the distance per degree at the specified latitude
    meters_per_degree = geodesic((latitude, 0), (latitude + 1, 0)).meters

    # Convert meters to degrees
    degrees = meters / meters_per_degree

    return degrees

def generate_bounding_boxes(tiff_path, box_size):
    """
    Generate fixed-sized square bounding boxes inside a given TIFF file.

    Parameters:
    - tiff_path: Path to the TIFF file
    - box_size: Size of the square bounding boxes

    Returns:
    - List of Shapely Polygon objects representing the bounding boxes
    """
    # Open the TIFF file and get its bounds
    bounding_boxes = []
    imgs = []
    
    with rasterio.open(tiff_path) as src:
        
        bounds = box(*src.bounds)
        avg_lat = (bounds.bounds[1] + bounds.bounds[2]) / 2
        box_size = meters_to_degrees(box_size, avg_lat)
        
        # Calculate the number of boxes in the x and y directions
        num_boxes_x = int((bounds.bounds[2] - bounds.bounds[0]) // box_size)
        num_boxes_y = int((bounds.bounds[3] - bounds.bounds[1]) // box_size)
        
        # Generate the bounding boxes
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                # Calculate the coordinates of the box
                min_x = bounds.bounds[0] + i * box_size
                min_y = bounds.bounds[1] + j * box_size
                max_x = min_x + box_size
                max_y = min_y + box_size

                # Create a Shapely box
                box_geometry = box(min_x, min_y, max_x, max_y)

                # Check if the box is completely within the TIFF bounds
                if bounds.contains(box_geometry):                    
                    window = src.window(min_x, min_y, max_x, max_y)
                    image_array = src.read(window=window)
                    pixels = image_array.shape[0] * image_array.shape[1]
                    
                    if np.count_nonzero(image_array) >= pixels * 0.5:  
                        imgs.append(image_array)                    
                        bounding_boxes.append(box_geometry)
        
    return bounding_boxes, imgs

geojson_folder = '../data/bbox_geojson'
if not os.path.isdir(geojson_folder):
    os.mkdir(geojson_folder)
    
labels_folder = '../data/labels'
if not os.path.isdir(labels_folder):
    os.mkdir(labels_folder)    

target_crs = from_epsg(4326)    
all_gdf = gpd.GeoDataFrame()
print("Generating Fixed Size Square Bboxes...")
target_crs = from_epsg(4326)
for i, path in enumerate(tif_labels_path):    
    # Generate bounding boxes
    bounding_boxes, imgs = generate_bounding_boxes(path, box_size)
    
    # Save the bounding boxes to a GeoJSON file
    gdf = gpd.GeoDataFrame(geometry=bounding_boxes, crs=target_crs)
    gdf['height'] = (imgs[0].shape[1])
    gdf['width'] = (imgs[0].shape[2])
    all_gdf = pd.concat([all_gdf, gdf], ignore_index = True)

all_gdf.to_file(geojson_folder + '/bounding_boxes.geojson', driver='GeoJSON')    
print("Generated geojson and saved to " + os.path.abspath(geojson_folder))

for i in range(100):
    np.save(labels_folder + "/{iid}.npy".format(iid = i), imgs[i])      
print("Saved the Fixed Size Square Bboxes as numpy arrays to " + os.path.abspath(labels_folder))
