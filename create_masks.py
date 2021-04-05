import rasterio
import fiona
import os
from rasterio.mask import mask
from PIL import Image
from osgeo import gdal, osr, ogr
Image.MAX_IMAGE_PIXELS = None


class Create_Mask:
    
    def __init__(self,tile_size):
        self.tile_size=tile_size

    ## Check if polygon and georeferenced satellite image overlap
    def in_polygon(self,raster,vector):
        raster = gdal.Open(raster)
        vector = ogr.Open(vector)

        # Get raster geometry
        transform = raster.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        cols = raster.RasterXSize
        rows = raster.RasterYSize

        xLeft = transform[0]
        yTop = transform[3]
        xRight = xLeft+cols*pixelWidth
        yBottom = yTop+rows*pixelHeight

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xLeft, yTop)
        ring.AddPoint(xLeft, yBottom)
        ring.AddPoint(xRight, yBottom)
        ring.AddPoint(xRight, yTop)
        ring.AddPoint(xLeft, yTop)
        rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
        rasterGeometry.AddGeometry(ring)
        # Get vector geometry
        layer = vector.GetLayer()
        feature = layer.GetFeature(0)
        vectorGeometry = feature.GetGeometryRef()
        return rasterGeometry.Intersect(vectorGeometry)

    def create_mask(self,image,image_size,color):
        for i in range(image_size):
            for j in range(image_size):
                if image[0,i,j] >0 and image[1,i,j] >0 and image[2,i,j] >0:
                    image[:,i,j] = color
        return image

    #Convert Geotiff to png
    def tif_to_png(self,tif_path,png_path):

        options_list = ['-ot Byte','-of PNG','-scale']  

        options_string = " ".join(options_list)

        gdal.Translate(png_path,tif_path,options=options_string)

    #Convert image of large size into smaller tiles    
    def tile_image(self,image_path,mask_path):
        #Create Directories to store image tiles
        if not os.path.exists('./data/images/'):
            os.makedirs('./data/images/')

        if not os.path.exists('./data/masks/'):
            os.makedirs('./data/masks/')

        img = Image.open(image_path)
        msk = Image.open(mask_path)

        width, height = img.size

        # Save Chops of original image
        for x0 in range(0, width, self.tile_size):
            for y0 in range(0, height, self.tile_size):
                box = (x0, y0,
                     x0 + self.tile_size if x0 + self.tile_size < width else width - 1,
                     y0 + self.tile_size if y0 + self.tile_size < height else height - 1)
                if img.crop(box).size ==(self.tile_size, self.tile_size):

                    img.crop(box).save("".join(['./data/images/', 'sa','_',str(x0),'_', str(y0),'.png']))
                    msk.crop(box).save("".join(['./data/masks/', 'sa','_',str(x0),'_', str(y0),'.png']))


if not os.path.exists('./data/'):
    os.makedirs('./data/')
        
        
image_w_geo = '2628A.tif'
polygon = '2628A_buildings.shp'

mask_w_geo = './data/2628A_mask.tif'

final_mask = './data/2628A_mask.png'
final_image= './data/2628A_image.png'
image_size =21688
mask_color= [255,255,255]

data = Create_Mask(256)


## Check if the image overlapse with the polygon
if data.in_polygon(image_w_geo,polygon): 
    print(image_w_geo + ' overlaps with polygon')

    #read the polygon geometry
    with fiona.open(polygon, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]

    #Mask out the non-overlapping parts without cropping
    with rasterio.open(image_w_geo) as src:
        out_image, out_transform = rasterio.mask.mask(src, geoms,crop= False)
        out_meta = src.meta.copy()

    #color the overlapping parts with a white color so that the background is 
    #black and the foreground is white    
    im = data.create_mask(out_image,image_size,mask_color)

    #save masked image as a Geotiff
    with rasterio.open(mask_w_geo, "w", **out_meta) as dest:
        dest.write(im) 

    ##Convert image and mask from Geotiffs to pngs
    data.tif_to_png(image_w_geo,final_mask)
    data.tif_to_png(mask_w_geo,final_image)

    data.tile_image(final_image,final_mask)