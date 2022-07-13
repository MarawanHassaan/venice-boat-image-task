import os
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

TRAIN_SET_PATH = 'train'
TEST_SET_PATH = 'test'

def read_train_set():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_SET_PATH, 
        target_size=(150, 150), 
        batch_size=16, 
        color_mode='rgb', 
        class_mode='categorical') 

    return train_generator


def read_test_set():
    df = pd.DataFrame()
    df = pd.read_csv(TEST_SET_PATH + '/ground_truth.txt', sep=';',
                     names=['filename', 'class'])

    df['class'] = df['class'].apply(
        lambda x: x.strip().replace(' ', '').replace(':', ''))
    df_filtered = df[df['class'].isin(os.listdir(TRAIN_SET_PATH))]
    df_filtered = df_filtered.reset_index(drop=True)

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    test_generator = test_datagen.flow_from_dataframe(
        df_filtered,
        TEST_SET_PATH,
        target_size=(150, 150),
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical')

    return test_generator, df_filtered['class']
