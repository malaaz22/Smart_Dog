import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import serial
import serial.tools.list_ports
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from threading import Thread
from datetime import datetime

# === Model yükleniyor ===
model = tf.keras.models.load_model("sound_command_model_cpu_professional.h5")
labels = ['dur', 'duz_devam_et', 'geri_don', 'sag_don', 'sola_don']
komut_kodlari = {
    'dur': 'S',
    'duz_devam_et': 'F',
    'geri_don': 'B',
    'sag_don': 'R',
    'sola_don': 'L'
}

ser = None  # Global seri port nesnesi

def log_yaz(mesaj):
    zaman = datetime.now().strftime("%H:%M:%S")
    log_text.configure(state="normal")
    log_text.insert(tk.END, f"[{zaman}] {mesaj}\n")
    log_text.configure(state="disabled")
    log_text.see(tk.END)

def seri_portlari_yenile():
    global ser
    ports = serial.tools.list_ports.comports()
    portlar = [port.device for port in ports]
    port_combo['values'] = portlar
    if portlar:
        port_combo.current(0)
        try:
            ser = serial.Serial(portlar[0], 9600, timeout=1)
            log_yaz(f"✅ {portlar[0]} portu açıldı.")
        except serial.SerialException as e:
            log_yaz(f"❌ Seri port açılamadı: {e}")
    else:
        log_yaz("❗ Seri port bulunamadı.")

def sesi_kaydet():
    fs = 16000
    saniye = 2
    log_yaz("🎙 Kayıt başladı...")
    ses = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
    sd.wait()
    log_yaz("🎙 Kayıt bitti.")
    return ses.flatten(), fs

def mfcc_ozellikleri(sinyal, sr):
    mfcc = librosa.feature.mfcc(y=sinyal, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-10)

    if mfcc.shape[1] < 44:
        pad = 44 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :44]

    mfcc = mfcc.reshape(1, 40, 44, 1)
    return mfcc

def ses_tanima_ve_gonder():
    global ser
    etiket.config(text="🎤 Dinleniyor...")
    ses, sr = sesi_kaydet()
    ozellik = mfcc_ozellikleri(ses, sr)
    tahmin = model.predict(ozellik)
    index = np.argmax(tahmin)
    komut = labels[index]

    log_yaz(f"✅ Tanınan komut: {komut}")
    etiket.config(text=f"✅ Komut: {komut}")
    veri = komut_kodlari.get(komut, '')

    if veri:
        if ser and ser.is_open:
            ser.write(veri.encode())
            log_yaz(f"📤 Gönderildi: {veri}")
            son_karakter_etiketi.config(text=f"🟢 Gönderilen: {veri}")
            etiket.config(text=f"📤 Seri porta gönderildi: {veri}")
        else:
            log_yaz("❗ Seri port açık değil.")
            etiket.config(text="❗ Port kapalı.")
            son_karakter_etiketi.config(text="🔴 Gönderilemedi")
    else:
        etiket.config(text="❗ Komut eşleşmedi.")
        log_yaz(f"❗ '{komut}' komutu eşleşmedi.")
        son_karakter_etiketi.config(text="⚪ Tanınmayan komut")

def butona_basildi():
    Thread(target=ses_tanima_ve_gonder).start()

# GUI
pencere = tk.Tk()
pencere.title("Akıllı Köpek Sesli Kontrol")
pencere.geometry("500x600")

etiket = tk.Label(pencere, text="Butona basıp komut söyleyin.", font=("Arial", 14))
etiket.pack(pady=20)

buton = tk.Button(pencere, text="🎙 Komut Söyle", command=butona_basildi, font=("Arial", 12), width=20)
buton.pack(pady=10)

port_frame = tk.Frame(pencere)
port_frame.pack(pady=10)

port_label = tk.Label(port_frame, text="🔌 Seri Port Seç:", font=("Arial", 11))
port_label.pack(side=tk.LEFT, padx=5)

port_combo = ttk.Combobox(port_frame, width=20, state="readonly")
port_combo.pack(side=tk.LEFT)

yenile_buton = tk.Button(port_frame, text="🔄 Yenile", command=seri_portlari_yenile)
yenile_buton.pack(side=tk.LEFT, padx=5)

log_label = tk.Label(pencere, text="📜 Komut Geçmişi:", font=("Arial", 11))
log_label.pack()

log_text = ScrolledText(pencere, width=60, height=10, state="disabled", font=("Consolas", 10))
log_text.pack(pady=5)

son_karakter_etiketi = tk.Label(pencere, text="⚪ Henüz komut gönderilmedi", font=("Arial", 12, "bold"), fg="blue")
son_karakter_etiketi.pack(pady=10)

seri_portlari_yenile()
pencere.mainloop()
