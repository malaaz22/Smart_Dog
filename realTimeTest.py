#!/usr/bin/env python3
"""
Mikrofondan gerçek zamanlı çoklu komut tanıma.
SPACE tuşuna basılı tutarak kayıt yapın; tuş bırakıldığında, kayıt 0.5 saniye daha devam eder,
sonrasında ses sessizliklere göre segmentlere ayrılır ve her segment için komut tahmini yapılır.
Tahmin edilen komutlar boşluklarla ayrılmış şekilde yazdırılır.
"""

import time
import numpy as np
import sounddevice as sd
import librosa
import keyboard
import tensorflow as tf
import logging

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametreler (eğitim ayarlarıyla uyumlu olmalı)
SR = 16000         # Örnekleme hızı
N_MFCC = 40
MAX_PAD_LEN = 44   # Eğitimde kullanılan ayar ile eşleşmeli
COMMANDS = ['dur', 'duz_devam_et', 'geri_don', 'sag_don', 'sola_don']
MODEL_PATH = 'sound_command_model_cpu_professional.h5'
SILENCE_TOP_DB = 20  # Sessizlik tespiti için eşik değeri
MIN_SEGMENT_LENGTH = 200  # Geçerli bir segment için minimum örnek sayısı

# Kayıt için global değişkenler
recording = False
audio_frames = []
stream = None

# Eğitilmiş modeli yükle
logging.info("Model yükleniyor: %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
logging.info("Model başarıyla yüklendi.")

def audio_callback(indata, frames, time_info, status):
    """
    Mikrofon verilerini toplamak için callback fonksiyonu.
    """
    global audio_frames
    if status:
        logging.warning("Ses akışı durumu: %s", status)
    audio_frames.append(indata.copy())

def extract_features(audio, sr=SR):
    """
    Sesten normalize edilmiş MFCC özelliklerini çıkarır ve MAX_PAD_LEN'e göre pad/truncate işlemi yapar.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc

def process_and_predict(audio_data):
    """
    Kaydedilen sesi işler, sessizliklere göre segmentlere ayırır ve her segment için komut tahmini yapar.
    """
    # Tüm ses verilerini birleştir ve normalize et
    audio = np.concatenate(audio_data, axis=0).flatten()
    logging.info("Toplam ses süresi: %.2f saniye", len(audio) / SR)
    
    # Sessizlik tespiti kullanılarak sesi segmentlere ayır
    intervals = librosa.effects.split(audio, top_db=SILENCE_TOP_DB)
    logging.info("Tespit edilen segment sayısı: %d", len(intervals))
    
    predictions = []
    for start, end in intervals:
        if end - start < MIN_SEGMENT_LENGTH:
            continue
        segment = audio[start:end]
        features = extract_features(segment, SR)
        features = features[np.newaxis, ..., np.newaxis]  # Modelin beklediği forma dönüştür: (1, n_mfcc, MAX_PAD_LEN, 1)
        
        pred = model.predict(features)[0]
        pred_idx = np.argmax(pred)
        predictions.append(COMMANDS[pred_idx])
    
    if predictions:
        result = " ".join(predictions)
        logging.info("Tahmin edilen komutlar: %s", result)
        print("Tahmin edilen komutlar:", result)
    else:
        logging.info("Geçerli konuşma segmenti tespit edilmedi.")
        print("Geçerli konuşma segmenti tespit edilmedi.")

def start_recording():
    """
    Mikrofondan ses kaydını başlatır.
    """
    global recording, audio_frames, stream
    audio_frames = []
    recording = True
    stream = sd.InputStream(samplerate=SR, channels=1, callback=audio_callback)
    stream.start()
    logging.info("Kayıt başladı...")

def stop_recording():
    """
    Ek 0.5 saniye kayıttan sonra ses kaydını durdurur.
    """
    global recording, stream
    time.sleep(0.5)
    stream.stop()
    recording = False
    logging.info("Kayıt durduruldu. Ses işleniyor...")
    process_and_predict(audio_frames)

def on_space_press(e):
    """
    SPACE tuşuna basıldığında kayıt başlatma işlemini tetikler.
    """
    global recording
    if not recording:
        start_recording()

def on_space_release(e):
    """
    SPACE tuşunun bırakılmasıyla kayıt durdurma işlemini tetikler.
    """
    if recording:
        stop_recording()

def main():
    logging.info("Gerçek zamanlı çoklu komut tanıma başladı.")
    logging.info("Kayıt için SPACE tuşuna basılı tutun; komutları işlemek için SPACE tuşunu bırakın.")
    
    # SPACE tuşu için event handler'ları kaydet
    keyboard.on_press_key("space", on_space_press)
    keyboard.on_release_key("space", on_space_release)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Gerçek zamanlı tanıma sonlandırılıyor.")
    finally:
        keyboard.unhook_all()

if __name__ == '__main__':
    main()
    