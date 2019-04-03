from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import csv

# hyper parameter
training_epoches = 5000
batch_size = 512
num_classes = 12

# load data
train_images = np.load('training_data.npy')
train_labels = to_categorical(np.load('training_labels.npy'), num_classes=num_classes)
print("Data loaded successfully...")

# data split
x_train, x_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
print(x_valid.shape)

'''
# create image data generator
imgGen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    cval=0,
    channel_shift_range=0,
    )
'''u

# create VGG16 model
model1 = VGG16(weights='imagenet', include_top = False, input_shape=(224,224,3))
print("VGG16 model instantiated successfully...")

# predict for validation set
x_valid = model1.predict(x_valid)
x_valid = x_valid.reshape(len(x_valid), -1)
print(x_valid.shape)

# training
features_train = model1.predict(x_train)
features_train = features_train.reshape(len(features_train),-1)
print(features_train.shape)

# create MLP model
model2 = Sequential()
model2.add(Dense(512, activation='relu', input_dim=features_train.shape[1], kernel_regularizer=regularizers.l2(0.001))) #, kernel_regularizer=regularizers.l2(0.01)
model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model2.summary()) # cannot produce summary if the input shape of model2 is not specified
print("MLP model created successfully...")

print("Training...")
history = model2.fit(features_train,y_train, epochs=training_epoches,batch_size=32, verbose=1, validation_data=(x_valid,y_valid))

# initialize record
record = open('epoch record.csv', 'w')
writer = csv.writer(record)
writer.writerow(['Train loss', 'Val loss', 'Train acc', 'Val acc'])

# save history
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

for i in range(len(train_acc)):
	writer.writerow([train_loss[i], val_loss[i], train_acc[i], val_acc[i]])

record.close()

# serialize model to JSON
model_json = model2.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model2.save_weights("model.h5")
print("Saved model to disk")