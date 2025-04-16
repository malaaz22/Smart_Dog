#!/usr/bin/env python3
"""
Profesyonel çoklu komut test scripti: CPU üzerinde ses komut tanıma için.
Bu script, bir ses dosyasını sessizliklere göre segmentlere ayırır ve her segment için komut tahmini yapar.
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametreler (eğitim ayarlarıyla uyumlu olmalı)
COMMANDS = ['dur', 'duz_devam_et', 'geri_don', 'sag_don', 'sola_don']
N_MFCC = 40
MAX_PAD_LEN = 44  # Eğitimde kullanılan uzunluk
SR = 16000  # Örnekleme hızı

def extract_features(audio, sr=SR):
    """
    Ses verisinden normalize edilmiş MFCC özelliklerini çıkarır ve MAX_PAD_LEN'e göre pad/truncate işlemi yapar.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc

def predict_commands(model, audio_path):
    """
    Verilen ses dosyasını yükler, sessizliklere göre segmentlere ayırır ve her segment için komut tahmini yapar.
    """
    logging.info("Ses dosyası yükleniyor: %s", audio_path)
    audio, sr = librosa.load(audio_path, sr=SR)
    
    # Sessizliklere göre ses dosyasını böl. Gerekirse top_db değeri ayarlanabilir.
    intervals = librosa.effects.split(audio, top_db=20)
    logging.info("Toplam %d segment bulundu.", len(intervals))
    
    predictions = []
    
    for i, (start, end) in enumerate(intervals):
        # Çok kısa segmentleri atla (eşik değeri gerektiğinde ayarlanabilir)
        if end - start < 200:
            continue
        segment = audio[start:end]
        features = extract_features(segment, sr)
        features = features[np.newaxis, ..., np.newaxis]  # Modelin beklediği forma dönüştür: (1, n_mfcc, MAX_PAD_LEN, 1)
        
        pred = model.predict(features)[0]
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx]
        start_time = start / sr
        end_time = end / sr
        
        predictions.append({
            'segment': i + 1,
            'command': COMMANDS[pred_idx],
            'confidence': confidence,
            'start_time': start_time,
            'end_time': end_time
        })
    return predictions

def main():
    # Komut satırında ses dosyası yolu verilmiş olmalı
    if len(sys.argv) < 2:
        print("Usage: python test_multi_command.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    
    # Eğitilmiş modeli yükle (dosya adının kaydedilen model ile eşleştiğinden emin olun)
    model_path = 'sound_command_model_cpu_professional.h5'
    logging.info("Model yükleniyor: %s", model_path)
    model = tf.keras.models.load_model(model_path)
    
    predictions = predict_commands(model, audio_path)
    
    if not predictions:
        logging.info("Geçerli ses segmenti bulunamadı.")
        return

    logging.info("Tahmin edilen komutlar:")
    for pred in predictions:
        print(f"Segment {pred['segment']}: {pred['command']} (güven: {pred['confidence']:.2f}) "
              f"{pred['start_time']:.2f}s ile {pred['end_time']:.2f}s arası")
        
if __name__ == '__main__':
    main()
