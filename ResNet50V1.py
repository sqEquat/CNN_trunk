import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D


# Off fitting on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_path = "resources/TRUNK12_test/Train"
val_path = "resources/TRUNK12_test/Val"
batch_size = 10


def resnet50_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None))
    model.add(Dense(12, activation='softmax'))

    return model


def get_data_generators():
    train_datagen = image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                             vertical_flip=True, preprocessing_function=preprocess_input)

    test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

    train = train_datagen.flow_from_directory(train_path, batch_size=batch_size,
                                              class_mode='categorical', target_size=(224, 224))
    validation = test_datagen.flow_from_directory(val_path, batch_size=batch_size,
                                                  class_mode='categorical', target_size=(224, 224))

    return train, validation


base_model = resnet50_model()
base_model.summary()
base_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

train_gen, val_gen = get_data_generators()

resnet_history = base_model.fit(train_gen, validation_data=val_gen,
                                steps_per_epoch=16, epochs=100)

base_model.save("models/ResNet50V1_12_32921")

print('\nHistory dict: ', resnet_history.history)
print("'loss': ", resnet_history.history['loss'])
print("'acc': ", resnet_history.history['acc'])
print("'val_loss': ", resnet_history.history['val_loss'])
print("'val_acc': ", resnet_history.history['val_acc'])
