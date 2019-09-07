import keras
import numpy as np
import tensorflow as tf

# constants
IMG_SHAPE = (400, 300)
CROP_SIZE = (333, 250)
BATCH_SIZE = 8
EPOCHS = 1

# data generator
def data_flow():
  datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=None,
    shear_range=0.25,
    zoom_range=0.25,
    channel_shift_range=0.,
    fill_mode="nearest",
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255,
    preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input,
    data_format=None,
    validation_split=0.2,
    dtype=np.uint8
  )

  image_generator = datagen.flow_from_directory(
    "/home/ryan/Documents/food-datasets/downloaded-datasets/food-101/images",
    # "/home/ryan/PycharmProjects/food-404/images",
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
  )

  # return add_random_crops(image_generator)
  return image_generator

def val_data_flow():
  datagen = keras.preprocessing.image.ImageDataGenerator()

  image_generator = datagen.flow_from_directory(
    "/home/ryan/Documents/food-datasets/downloaded-datasets/upmc-food-101/images/test",
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
  )

  return image_generator

def add_random_crops(flow_from_dir, crop_size = CROP_SIZE):
  #
  # def random_crop(img, size):
  #   height, width = img.shape[0], img.shape[1]
  #   change_y, change_x = size
  #   x = np.random.randint(0, width - change_x + 1)
  #   y = np.random.randint(0, height - change_y + 1)
  #   return img[y:(y + change_y), x:(x + change_x), :]

  while True:
    batch, labels = next(flow_from_dir)
    cropped_batches = np.empty((batch.shape[0], *crop_size, 3))
    for index, img in enumerate(cropped_batches.shape[0]):
      cropped_batches[index] = tf.random_crop(img, crop_size) # random_crop(img, crop_size)
    yield (cropped_batches, labels)