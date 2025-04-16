#!/usr/bin/env python3
"""
Profesyonel eğitim scripti: CPU üzerinde ses komut tanıma işlemleri için
ses dosyalarından özellik çıkarımı, veri arttırma, CNN tabanlı derin öğrenme modelinin
oluşturulması ve eğitilmesi işlemlerini gerçekleştirir.
"""

import os
import logging
import random
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout,
    Flatten, Dense, add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# TensorFlow'un sadece CPU kullanması için zorla
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametreler
COMMANDS = ['dur', 'duz_devam_et', 'geri_don', 'sag_don', 'sola_don']
N_MFCC = 40
MAX_PAD_LEN = 44  # Ses süresine göre ayarlayın
DATA_PATH = 'data'  # Her komut için alt klasörler içeren veri yolu
SR = 16000  # Örnekleme hızı

def augment_pitch(audio, sr, n_steps):
    """Pitch shift (perde değiştirme) veri arttırma uygulaması."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def augment_time_stretch(audio, rate):
    """Zaman germe (time stretch) veri arttırma uygulaması."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def extract_features(audio, sr=SR):
    """Sesten normalize edilmiş MFCC özelliklerini çıkarır."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc

def process_file(file_path, sr=SR, augment=True):
    """Ses dosyasını yükler ve opsiyonel veri arttırma ile özellik çıkarımı yapar."""
    audio, _ = librosa.load(file_path, sr=sr)
    features = [extract_features(audio, sr)]
    if augment:
        n_steps = random.uniform(-1, 1)
        audio_pitch = augment_pitch(audio, sr, n_steps)
        features.append(extract_features(audio_pitch, sr))
        rate = random.uniform(0.9, 1.1)
        audio_stretch = augment_time_stretch(audio, rate)
        features.append(extract_features(audio_stretch, sr))
    return features

def load_data(data_path=DATA_PATH, commands=COMMANDS, augment=True):
    """Alt dizinlerden veri yükler ve veri arttırma uygular."""
    data, labels = [], []
    for label, command in enumerate(commands):
        folder = os.path.join(data_path, command)
        logging.info("'%s' komutu için '%s' klasöründen işleniyor", command, folder)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder, filename)
                feats = process_file(file_path, augment=augment)
                for feat in feats:
                    data.append(feat)
                    labels.append(label)
    logging.info("Veri yükleme tamamlandı. İşlenen örnek sayısı: %d", len(data))
    data = np.array(data)[..., np.newaxis]  # Şekil: (örnekler, n_mfcc, MAX_PAD_LEN, 1)
    labels = tf.keras.utils.to_categorical(np.array(labels), num_classes=len(commands))
    return train_test_split(data, labels, test_size=0.2, random_state=42)

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """İki konvolüsyonel katman içeren residual bloğu tanımlar."""
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters or strides != (1, 1):
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    return Activation('relu')(x)

def build_model(input_shape, num_classes):
    """Residual blokları içeren CNN modelini oluşturur ve derler."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    logging.info("Veri hazırlığı başlatılıyor...")
    x_train, x_val, y_train, y_val = load_data()

    logging.info("Model oluşturuluyor...")
    model = build_model(x_train.shape[1:], num_classes=len(COMMANDS))
    model.summary(print_fn=logging.info)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('sound_command_model_cpu_professional.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    logging.info("Eğitim başlatılıyor...")
    history = model.fit(
        x_train, y_train, epochs=100, batch_size=16,
        validation_data=(x_val, y_val), callbacks=callbacks, verbose=1
    )
    logging.info("Eğitim tamamlandı.")

if __name__ == '__main__':
    main()
