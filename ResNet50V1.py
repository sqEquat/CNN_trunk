import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D


# Disable training on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def resnet50_model():
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None))
    model.add(Dense(12, activation='softmax'))

    return model


batch_size = 10

train_datagen = image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                         vertical_flip=True, preprocessing_function=preprocess_input)

test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory("resources/TRUNK12_test/Train", batch_size=batch_size,
                                                    class_mode='categorical', target_size=(224, 224))
validation_generator = test_datagen.flow_from_directory("resources/TRUNK12_test/Val", batch_size=batch_size,
                                                        class_mode='categorical', target_size=(224, 224))

base_model = resnet50_model()

base_model.summary()

base_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

resnet_history = base_model.fit(train_generator, validation_data=validation_generator,
                                steps_per_epoch=16, epochs=100)

base_model.save("models/ResNet50V1_12_32821")

print('\nHistory dict: ', resnet_history.history)
print("'loss': ", resnet_history.history['loss'])
print("'acc': ", resnet_history.history['acc'])
print("'val_loss': ", resnet_history.history['val_loss'])
print("'val_acc': ", resnet_history.history['val_acc'])
