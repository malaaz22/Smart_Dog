# test_model.py
import sys
import numpy as np
import librosa
import tensorflow as tf

# Kullanılacak komutlar listesi
commands = ['dur', 'duz_devam_et', 'geri_don', 'sag_don', 'sola_don']
max_pad_len = 44  # Eğitimde kullanılan padding uzunluğu ile eşleşmelidir

def preprocess_audio(file_path, sr=16000):
    """
    Ses dosyasını yükler, MFCC özelliklerini çıkarır ve gerekli padding işlemini yapar.
    """
    audio, _ = librosa.load(file_path, sr=sr)  # Ses dosyasını belirtilen örnekleme hızıyla yükle
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # MFCC özelliklerini çıkar
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]  # Gerekli padding miktarını hesapla
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')  # Padding uygula
    else:
        mfcc = mfcc[:, :max_pad_len]  # Fazla olan kısımları kes
    return mfcc

if __name__ == "__main__":
    # Argüman kontrolü: Ses dosyası yolu sağlanmalıdır
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]  # Komut satırından alınan ses dosyası yolu
    mfcc = preprocess_audio(audio_file)  # Ses dosyasını ön işle
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Veriyi modelin beklediği forma getir: (1, n_mfcc, max_pad_len, 1)

    # Eğitilmiş modeli yükle
    model = tf.keras.models.load_model('sound_command_model_cpu.h5')
    prediction = model.predict(mfcc)  # Model ile tahmin yap
    predicted_index = np.argmax(prediction)  # En yüksek olasılığa sahip indeksi belirle
    print("Predicted command:", commands[predicted_index])  # Tahmin edilen komutu yazdır
