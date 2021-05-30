import tensorflow as tf
import csv
# import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet101V2
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# Off fitting on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_path = "resources/TRUNK12_test/Train"
val_path = "resources/TRUNK12_test/Val"

checkpoint_path = "models/resnet50/resnet50_trunk3_053021_{epoch:02d}_{val_acc:.4f}.h5"
fit_result_csv = "stat/resnet50/053021_50.csv"

img_shape = (224, 224)
train_samples_num = 261
val_samples_num = 63

batch_size = 10
lr_rate = 0.001
epochs = 100



def resnet50_transfer_model(model_path, class_num):
    pretrained_model = keras.models.load_model(model_path)
    pretrained_model.summary()
    base_model = Sequential()
    for layer in pretrained_model.layers[:-1]:
        layer.trainable = False
        base_model.add(layer)

    base_model.add(Dense(class_num, activation='softmax'))
    base_model.summary()

    return base_model


def transfer_learning_fit_gen(dataset_path):
    train_datagen = image.ImageDataGenerator(rescale=1./255, width_shift_range=0.3, height_shift_range=0.3,
                                             shear_range=0.3, vertical_flip=True, validation_split=0.2,
                                             preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(dataset_path, target_size=img_shape, batch_size=batch_size,
                                                  class_mode='categorical', subset='training')

    val_gen = train_datagen.flow_from_directory(dataset_path, target_size=img_shape, batch_size=batch_size,
                                                class_mode='categorical', subset='validation')

    return train_gen, val_gen


def resnet50_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None))
    model.add(Dense(12, activation='softmax'))
    return model


def get_data_generators():
    train_datagen = image.ImageDataGenerator(rescale=1./255, width_shift_range=0.3, height_shift_range=0.3,
                                             shear_range=0.3, vertical_flip=True,
                                             preprocessing_function=preprocess_input)

    val_datagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

    train = train_datagen.flow_from_directory(train_path, batch_size=batch_size,
                                              class_mode='categorical', target_size=img_shape)
    validation = val_datagen.flow_from_directory(val_path, batch_size=batch_size,
                                                 class_mode='categorical', target_size=img_shape)

    return train, validation


def training(model_base, train_gen, val_gen):
    model_base.compile(optimizer=tf.keras.optimizers.Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['acc'])
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')

    history = model_base.fit(train_gen, validation_data=val_gen, steps_per_epoch=train_samples_num // batch_size,
                             validation_steps=val_samples_num // batch_size, epochs=epochs,
                             callbacks=[checkpoint, reduce_lr])

    return history


if __name__ == '__main__':
    # train, val = get_data_generators()
    # base_model = resnet50_model()
    # fit_history = training(base_model, train, val)
    base_model_path = "models/resnet50/resnet50_trunk12_74_0.9250.h5"
    dataset_path = "resources/prediction/trunkmo"
    model_classes_num = 3
    model = resnet50_transfer_model(base_model_path, model_classes_num)
    train_gen, val_gen = transfer_learning_fit_gen(dataset_path)
    fit_history = training(model, train_gen, val_gen)

    print('\nTraining metrics: ')
    print("'loss': ", fit_history.history['loss'])
    print("'acc': ", fit_history.history['acc'])
    print("'val_loss': ", fit_history.history['val_loss'])
    print("'val_acc': ", fit_history.history['val_acc'])

    with open(fit_result_csv, 'w') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(fit_history.keys())
        writer.writerows(zip(*fit_history.values()))
