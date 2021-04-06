# gis_x_vision: Case study - Johannesburg, South Africa

Given that you have georeferenced satellite images and polygons depicting the extent of buildings on overlapping satellite images, this repository will walk you through the details of how you can create ground truth masks using the polygons of buildings and then further train a model to predict building pixels on satellite images.

## Testing the Pipeline
1. To run the code, first download the compressed dataset file from [here](https://drive.google.com/file/d/1mJ2pA4OK-3QZDZpbmVq1OCwRY6xMMcgH/view?usp=sharing) that consists of one large satellite image of a part of Johannesburg, South Africa and a corresponding dataset showing the extent of the real estate as registered with the government.
2. Unzip this file and place the contents in the data folder.
3. Run the [create_masks.py](https://github.com/sefalab/gis_x_vision/blob/main/create_masks.py) code to convert the building vector layer into a corresponding PNG mask that has white pixels (255, 255, 255) where there are buildings and then black pixels (0, 0, 0) where there are no pixels.
    - This pipeline starts by first checking whether the satellite image overlapse with the vector layer.
    - Using Rasterio, creates a mask by covering all the non-overlapping parts(background) with black pixels and then cover the overlapping parts with white pixels.
    - Since we do not need the geographical information for the machine learning part, we convert the satellite image and correspoding mask to a PNG format(PNGs conserve pixel values unlike in JPEGs).
    - Convert the 21688 x21688 pixel satellite image into smaller tiles and store them in folders.
    - At this point you should have an image dataset and a masks dataset both in PNG format that you can use to train a semantic segmentation model.
4. Run the [train.py](https://github.com/sefalab/gis_x_vision/blob/main/train.py) file to train a unet model to segment out pixels of buildings from background pixels.
   -split the data into train-val-test datasets, load the data and then train a semantic segmentation model and then store the weights.
5. Run the [eval.ipynb](https://github.com/sefalab/gis_x_vision/blob/main/eval.ipynb) notebook to evaluate the model using the test set.
   
## These are some of the results from [these](https://github.com/sefalab/gis_x_vision/blob/main/saved_weights/model.h5) saved weights.
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_1.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_2.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_4.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_7.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_6.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_9.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_12.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_13.png)
![](https://github.com/sefalab/gis_x_vision/blob/main/visualizations/gis_8.png)

## Some observations
- Given that the vector(building) dataset is supposed to show the extent of real estate that is registered with the Government, some of these buildings may have not been built yet. We see this case in the groundtruth of the last 2 images above. In these cases, the model performs really well comparatively.

