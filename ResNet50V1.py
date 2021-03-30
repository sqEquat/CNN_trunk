import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# Off fitting on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_path = "resources/TRUNK12_test/Train"
val_path = "resources/TRUNK12_test/Val"
checkpoint_path = "models/resnet50/trunk12_{epoch:02d}_{val_acc:.4f}.h5"
batch_size = 10
img_shape = (224, 224)
lr_rate = 0.001
step_per_epoch = 16
epochs = 100


def resnet50_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None))
    model.add(Dense(12, activation='softmax'))

    return model


def get_data_generators():
    train_datagen = image.ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, shear_range=0.3,
                                             vertical_flip=True, preprocessing_function=preprocess_input)

    val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

    train = train_datagen.flow_from_directory(train_path, batch_size=batch_size,
                                              class_mode='categorical', target_size=img_shape)
    validation = val_datagen.flow_from_directory(val_path, batch_size=batch_size,
                                                 class_mode='categorical', target_size=img_shape)

    return train, validation


def training(model, train, val):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['acc'])
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    history = base_model.fit(train, validation_data=val, steps_per_epoch=step_per_epoch,
                             epochs=epochs, callbacks=[checkpoint, reduce_lr])

    return history, model


if __name__ == '__main__':
    base_model = resnet50_model()
    base_model.summary()
    train_gen, val_gen = get_data_generators()

    fit_history, base_model = training(base_model, train_gen, val_gen)

    print('\nHistory dict: ', fit_history.history)
    print("'loss': ", fit_history.history['loss'])
    print("'acc': ", fit_history.history['acc'])
    print("'val_loss': ", fit_history.history['val_loss'])
    print("'val_acc': ", fit_history.history['val_acc'])
