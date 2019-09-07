import keras
import numpy as np

# constants
IMG_SHAPE = (400, 300)
BATCH_SIZE = 8
EPOCHS = 1

# creating FoodNet
def FoodNet(train_type = "classifier", num_classes = 404):
  input_layer = keras.layers.Input(shape=(*IMG_SHAPE, 3))
  base = keras.applications.inception_v3.InceptionV3(include_top = False,
                                                     weights = "imagenet",
                                                     input_tensor = input_layer,
                                                     pooling = "avg")
  x = base.output
  predictions = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

  food_net = keras.Model(input=base.input, output=predictions)
  if train_type == "classifier":
    for layer in food_net.layers[:-1]:
      layer.trainable = False
  return food_net

# data generator
def data_flow():
  datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=None,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=0.,
    fill_mode="nearest",
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    preprocessing_function=keras.applications.inception_v3.preprocess_input,
    data_format=None,
    validation_split=0.,
    dtype=None
  )

  image_generator = datagen.flow_from_directory(
    "/home/ryan/Documents/food-datasets/food-404-copy/images",
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
  )

  return image_generator

if __name__ == "__main__":
  image_generator = data_flow()

  # first pass: training the output layer
  # food_net = FoodNet(train_type="classifier", num_classes=len(image_generator.class_indices))
  food_net = keras.models.load_model("/home/ryan/Documents/food-net/foodnet_v1.h5")
  food_net.summary()
  food_net.compile(optimizer=keras.optimizers.RMSprop(), loss="categorical_crossentropy",
                   metrics=["categorical_accuracy"])
  food_net.fit_generator(image_generator, steps_per_epoch=int(len(image_generator.classes) / BATCH_SIZE),
                         epochs=EPOCHS, verbose=2)
  food_net.save("/home/ryan/Documents/food-net/foodnet_v1.h5")

  # second pass: fine-tuning whole network
  # del food_net
  # food_net = keras.models.load_model("/home/ryan/Documents/food-net/foodnet_v1.h5")
  # for layer in food_net.layers:
  #   layer.trainable = True
  # food_net.summary()
  # food_net.compile(optimizer=keras.optimizers.RMSprop(), loss="categorical_crossentropy",
  #                  metrics=["accuracy"])
  # food_net.fit_generator(image_generator, steps_per_epoch=int(len(image_generator.classes) / BATCH_SIZE), epochs=EPOCHS)