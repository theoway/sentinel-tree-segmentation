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

- BBoxes in: `./data/bbox/*.json`
- Time period of images: 15 September, 2023 (or the closest date image available)

### Running the scripts
```bash
# downloading data
$  python get_data.py -d YYYY-MM-DD
```

### Resources
- Max size of bbox for requests [link](https://docs.sentinel-hub.com/api/latest/api/overview/processing-unit/).
