import keras
import numpy as np

from data_processing import BATCH_SIZE, IMG_SHAPE, EPOCHS, data_flow, val_data_flow

# creating FoodNet
def FoodNet(train_type = "classifier", num_classes = 404):
  base = keras.applications.inception_v3.InceptionV3(include_top = False,
                                                     weights = "imagenet",
                                                     input_shape = (*IMG_SHAPE, 3),
                                                     pooling="avg")
  x = base.output
  x = keras.layers.Dropout(0.4)(x)
  x = keras.layers.Dense(512, activation="relu")(x)
  x = keras.layers.Dense(256)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)
  predictions = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

  food_net = keras.Model(inputs=base.input, outputs=predictions)
  if train_type == "classifier":
    for layer in food_net.layers[:-60]:
      layer.trainable = False

  print(len(food_net.layers))
  return food_net

if __name__ == "__main__":
  image_generator = data_flow()

  # first pass: training the output layer
  food_net = FoodNet(train_type="classifier", num_classes=len(image_generator.class_indices))
  food_net.summary()

  food_net.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), loss="categorical_crossentropy",
                   metrics=["accuracy"])
  food_net.fit_generator(image_generator, steps_per_epoch=int(len(image_generator.classes) / BATCH_SIZE), epochs=EPOCHS,
                         validation_data=val_data_flow(),
                         validation_steps=int(len(image_generator.classes) / BATCH_SIZE), verbose=1)
  food_net.save("/home/ryan/PycharmProjects/food-net/foodnet_v1.h5")

  # second pass: fine-tuning whole network
  del food_net
  food_net = keras.models.load_model("/home/ryan/Documents/food-net/foodnet_v1.h5")
  for layer in food_net.layers:
    layer.trainable = True
  food_net.summary()
  food_net.compile(optimizer=keras.optimizers.RMSprop(), loss="categorical_crossentropy", metrics=["accuracy"])
  food_net.fit_generator(image_generator, steps_per_epoch=int(len(image_generator.classes) / BATCH_SIZE), epochs=EPOCHS)
  food_net.save("/home/ryan/PycharmProjects/food-net/foodnet_v1_3.h5")