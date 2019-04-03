from telegram_bot import telegram_bot_sendtext
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import numpy as np
import csv, os
from Plotter import Plotter
import pandas as pd


# hyperparamter
input_shape = (224, 224, 3)
training_epoch = 1000
batch_size = 128
validation_ratio = 0.2
num_classes = 12


def load_data(data_file, label_file):
    # load data
    train_images = np.load(data_file)
    train_labels = to_categorical(np.load(label_file), num_classes=num_classes)
    train_images, train_labels = shuffle(train_images, train_labels)
    print("Data loaded successfully...")

    return train_images, train_labels


def create_model():
    # create model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # default 128
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def train(Train_images, Train_labels, Test_images, Test_label):
    model = create_model()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print("model created successfully...")
    print(model.summary())

    es = [EarlyStopping(monitor='val_loss', patience=50),
          ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    print("Training...")
    telegram_bot_sendtext("Training Start")

    history = model.fit(Train_images, Train_labels,
                        batch_size=batch_size,
                        epochs=training_epoch,
                        callbacks=es,
                        validation_data=(Test_images, Test_label),
                        # validation_split=validation_ratio,
                        shuffle=True)

    telegram_bot_sendtext("Training Completed")

    return history, model


def load_trained_model(filename):
    # Model reconstruction from JSON file
    with open(filename + '.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(filename + '.h5')
    return model


def save_accuracy(history, filename):

    # initialize csv to record loss and accuracy
    record = open(filename, 'w', newline='')
    writer = csv.writer(record)
    writer.writerow(['Train loss', 'Val loss', 'Train acc', 'Val acc'])

    # summarize history for accuracy
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    for i in range(len(train_loss)):
        writer.writerow([train_loss[i], val_loss[i], train_acc[i], val_acc[i]])

    record.close()


def save_trained_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':

    test_images, test_labels = load_data('test_data.npy', 'test_labels.npy')
    print("Test Data loaded successfully...")
    plt = Plotter(no_epochs=training_epoch, label='Seed Classifier', extension='.png', file_name='Seed_Classifier')

    if os.path.exists('model.h5'):
        test_labels = np.argmax(test_labels, axis=1)
        _model = load_trained_model('model')
        y = _model.predict(test_images)
        y = np.argmax(y, axis=1)
        plt.plot_confusion_matrix(y_test=test_labels, y_pred=y, classes=set(test_labels))

    else:
        train_images, train_labels = load_data('training_data.npy', 'training_labels.npy')
        print("Training Data loaded successfully...")
        _history, _model = train(train_images, train_labels, test_images, test_labels)
        save_accuracy(_history, '5_layer_CNN_record.csv')
        save_trained_model(_model)

    if os.path.exists('5_layer_CNN_record.csv'):
        data = pd.read_csv('5_layer_CNN_record.csv')

        loss = np.array(data['Train loss'])
        acc = np.array(data['Val acc'])

        plt.plot_graph(loss, 'Training Loss')
        plt.plot_graph(acc, 'Training Accuracy')

