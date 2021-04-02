import numpy as np 
import pickle
import tensorflow as tf
from keras import layers, Sequential


# load training and testing data
train_imgs = pickle.load(open("train_imgs.pickle", "rb"))
train_labels = pickle.load(open("train_labels.pickle", "rb"))
test_imgs = pickle.load(open("test_imgs.pickle", "rb"))
test_labels = pickle.load(open("test_labels.pickle", "rb"))

# Create model. Uses are encouraged make modifications to improve accuracy and decrease loss
model = Sequential([
    layers.Conv2D(128, (3, 3), input_shape=train_imgs.shape[1:]),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1),
    layers.Activation('sigmoid')
])

NAME = "hotdog-not-hotdog-classifier"

# Compile and fit/train model
model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
model.fit(train_imgs, train_labels, batch_size=32, epochs=4, validation_split=0.1)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_imgs, test_labels)
print(f"Test Accuracy: {test_acc}")

# Save model
model.save(NAME)