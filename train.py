from tensorflow.keras.optimizers import Adam
import os
import tensorflow as tf
import random
import re
import matplotlib.pyplot as plt
from keras_unet.models.custom_unet import conv2d_block, custom_unet
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
import numpy as np
from tensorflow.keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

DATA_PATH = './data/'
FRAME_PATH = DATA_PATH+'/images/'
MASK_PATH = DATA_PATH+'/masks/'
seed =42
model_path='./saved_weights/model_v1.h5'
log_path = './log.out'
epochs = 35
bs = 16
num_classes =2

label_codes=[(255, 255, 255), (0, 0, 0)]
label_names=['building', 'background']

# Create folders to hold images and masks per data split
folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val', 'test_frames/test', 'test_masks/test']
            
for folder in folders:
    if not os.path.exists(DATA_PATH + folder):
        os.makedirs(DATA_PATH + folder)

# #Get all frames and masks, sort them, shuffle them to generate data splits.
# all_frames = os.listdir(FRAME_PATH)
# all_masks = os.listdir(MASK_PATH)


# all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x 
#                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])
# all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
#                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

# random.seed(seed)
# random.shuffle(all_frames)

# # Generate train, val, and test sets for frames train-val-test:60-20-20
# train_split = int(0.6*len(all_frames))
# val_split = int(0.8 * len(all_frames))

# train_frames = all_frames[:train_split]
# val_frames = all_frames[train_split:val_split]
# test_frames = all_frames[val_split:]


# # Generate corresponding mask lists for masks
# train_masks = [f for f in all_masks if f in train_frames]
# val_masks = [f for f in all_masks if f in val_frames]
# test_masks = [f for f in all_masks if f in test_frames]


# #Add train, val, test frames and masks to relevant folders
# def add_frames(dir_name, image):
  
#     img = Image.open(FRAME_PATH+image)
#     img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

# def add_masks(dir_name, image):
  
#     img = Image.open(MASK_PATH+image)
#     img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)
    
    
# frame_folders = [(train_frames, 'train_frames/train'), (val_frames, 'val_frames/val'), 
#                  (test_frames, 'test_frames/test')]

# mask_folders = [(train_masks, 'train_masks/train'), (val_masks, 'val_masks/val'), 
#                 (test_masks, 'test_masks/test')]

# # Add frames
# for folder in frame_folders:
  
#     array = folder[0]
#     name = [folder[1]] * len(array)

#     list(map(add_frames, name, array))

# # Add masks
# for folder in mask_folders:
  
#     array = folder[0]
#     name = [folder[1]] * len(array)

#     list(map(add_masks, name, array))
    

##Helper functions for converting label codes to names
id2code = {k:v for k,v in enumerate(label_codes)}

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)

# Normalizing only frame images, since masks contain label info
data_gen_args = dict(rescale=1./255)
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
val_frames_datagen = ImageDataGenerator(**data_gen_args)
val_masks_datagen = ImageDataGenerator(**mask_gen_args)

def TrainAugmentGenerator(seed=seed, batch_size = bs):
    '''Train Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    train_image_generator = train_frames_datagen.flow_from_directory(
    DATA_PATH + 'train_frames/',
    batch_size = batch_size, seed = seed)

    train_mask_generator = train_masks_datagen.flow_from_directory(
    DATA_PATH + 'train_masks/',
    batch_size = batch_size, seed = seed)

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)

def ValAugmentGenerator(seed=seed, batch_size = bs):
    '''Validation Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    val_image_generator = val_frames_datagen.flow_from_directory(
    DATA_PATH + 'val_frames/',
    batch_size = batch_size, seed = seed)


    val_mask_generator = val_masks_datagen.flow_from_directory(
    DATA_PATH + 'val_masks/',
    batch_size = batch_size, seed = seed)


    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)
        


model = custom_unet(
    (256,256,3),
    num_classes=num_classes,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=True,
    use_attention=True,
    filters=16,
    num_layers=4,
    output_activation="sigmoid")
   

model.compile(optimizer='adam',loss='binary_crossentropy',
              metrics=['accuracy',tf.keras.metrics.MeanIoU(num_classes=num_classes)])

plot_model(model,to_file='./visualizations/model_unet_w_att.png',show_shapes=True)

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger(log_path, append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'val_accuracy', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks = [checkpoint, csv_logger, earlystopping]

vs = (len(os.listdir('./data/val_frames/val/'))//bs)
ts = (len(os.listdir('./data/train_frames/train/'))//bs)
model.summary()


result = model.fit_generator(TrainAugmentGenerator(seed,bs), epochs=epochs, 
                          steps_per_epoch = ts,
                          validation_data=ValAugmentGenerator(seed,bs), 
                          validation_steps=vs, 
                          callbacks=callbacks)

model.save(model_path)
# Get actual number of epochs model was trained for
N = len(result.history['loss'])

#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))

fig.add_subplot(1,2,1)
plt.title("Loss")
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.ylim(0, 1)


# fig.add_subplot(1,3,2)
# plt.title("Mean Intersection Over Union")
# plt.plot(np.arange(0, N), result.history["mean_io_u_6"], label="train_miou")
# plt.plot(np.arange(0, N), result.history["val_mean_io_u_6"], label="val_miou")
# plt.ylim(-1,1)

fig.add_subplot(1,2,2)
plt.title("Accuracy")
plt.plot(np.arange(0, N), result.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), result.history["val_accuracy"], label="val_accuracy")
plt.ylim(0, 1)

plt.xlabel("Epoch #")
#plt.ylabel("")
plt.legend(loc="lower left")
plt.savefig('./visualizations/eval_during_train.png')
#plt.show()