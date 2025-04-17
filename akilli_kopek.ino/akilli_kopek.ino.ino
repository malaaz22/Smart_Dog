// Dosya Adı: arduino_kod.ino
// Açıklama: Seri port üzerinden gelen komutlara göre robot köpeği hareket ettiren Arduino kodu

#include <Servo.h>

Servo sag_on_ayak_diz;
Servo sag_on_ayak_omuz;
Servo sag_arka_ayak_diz;
Servo sag_arka_ayak_omuz;
Servo sol_on_ayak_diz;
Servo sol_on_ayak_omuz;
Servo sol_arka_ayak_diz;
Servo sol_arka_ayak_omuz;

int z;
int bekle = 4;
int data;

void setup() {
  sag_on_ayak_diz.attach(11);
  sag_on_ayak_omuz.attach(10);
  sag_arka_ayak_diz.attach(9);
  sag_arka_ayak_omuz.attach(6);
  sol_on_ayak_diz.attach(5);
  sol_on_ayak_omuz.attach(4);
  sol_arka_ayak_diz.attach(3);
  sol_arka_ayak_omuz.attach(2);

  kalibirasyon();
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    data = Serial.read();

    if (data == 'F') { ileri_ok(); }
    if (data == 'B') { geri_ok(); }
    if (data == 'L') { sola(); }
    if (data == 'R') { saga(); }
    if (data == 'S') { dur(); }
  }
}

void kalibirasyon() {
  sag_on_ayak_diz.write(90);
  sag_on_ayak_omuz.write(90);
  sag_arka_ayak_diz.write(90);
  sag_arka_ayak_omuz.write(90);
  sol_on_ayak_diz.write(90);
  sol_on_ayak_omuz.write(90);
  sol_arka_ayak_diz.write(90);
  sol_arka_ayak_omuz.write(90);
}

void dur() {
  for (int i = 90 ; i > 30 ; i--) {
    z = 90 - i;  delay(15);
    sag_on_ayak_diz.write(i);
    sag_arka_ayak_diz.write(i);
    sol_arka_ayak_diz.write(z + 90);
    sol_on_ayak_diz.write(z + 90);
  }
  delay(500);
  for (int i = 30 ; i < 90; i++) {
    z = i - 30;
    delay(15);
    sag_on_ayak_diz.write(i);
    sag_arka_ayak_diz.write(i);
    sol_arka_ayak_diz.write(150 - z);
    sol_on_ayak_diz.write(150 - z);
  }
  delay(500);
}

void ileri_ok() {
  sol_arka_ayak_omuz.write(90);
  sag_arka_ayak_omuz.write(40);
  sag_on_ayak_omuz.write(110);
  sol_on_ayak_omuz.write(100);
  delay(100);
  sol_arka_ayak_diz.write(50);
  sag_arka_ayak_diz.write(80);
  sag_on_ayak_diz.write(130);
  sol_on_ayak_diz.write(90);
  delay(100);
  sol_arka_ayak_omuz.write(130);
  sag_arka_ayak_omuz.write(90);
  sag_on_ayak_omuz.write(80);
  sol_on_ayak_omuz.write(70);
  delay(100);
  sol_arka_ayak_diz.write(90);
  sag_arka_ayak_diz.write(110);
  sag_on_ayak_diz.write(90);
  sol_on_ayak_diz.write(50);
  delay(100);
}

void geri_ok() {
  sol_arka_ayak_omuz.write(90);
  sag_arka_ayak_omuz.write(40);
  sag_on_ayak_omuz.write(110);
  sol_on_ayak_omuz.write(100);
  delay(100);
  sol_arka_ayak_diz.write(90);
  sag_arka_ayak_diz.write(110);
  sag_on_ayak_diz.write(90);
  sol_on_ayak_diz.write(50);
  delay(100);
  sol_arka_ayak_omuz.write(130);
  sag_arka_ayak_omuz.write(90);
  sag_on_ayak_omuz.write(80);
  sol_on_ayak_omuz.write(70);
  delay(100);
  sol_arka_ayak_diz.write(50);
  sag_arka_ayak_diz.write(80);
  sag_on_ayak_diz.write(130);
  sol_on_ayak_diz.write(90);
  delay(100);
}

void sola() {
  for (int i = 90 ; i > 80 ; i--) {
    delay(bekle);
    sag_on_ayak_diz.write(i);
  }
  for (int i = 90 ; i < 150; i++) {
    delay(bekle);
    sag_on_ayak_omuz.write(i);
  }
  for (int i = 80 ; i < 110; i++) {
    delay(bekle);
    sag_on_ayak_diz.write(i);
  }
  for (int i = 150 ; i > 90 ; i--) {
    sag_on_ayak_omuz.write(i);
    delay(bekle);
  }
  for (int i = 110 ; i > 90 ; i--) {
    sag_on_ayak_diz.write(i);
    delay(bekle);
  }
}

void saga() {
  for (int i = 90 ; i > 50 ; i--) {
    delay(bekle);
    sol_on_ayak_diz.write(i);
  }
  for (int i = 90 ; i < 110; i++) {
    sol_on_ayak_omuz.write(i);
    delay(bekle);
  }
  for (int i = 50 ; i < 110; i++) {
    sol_on_ayak_diz.write(i);
    delay(bekle);
  }
  for (int i = 110 ; i > 90 ; i--) {
    sol_on_ayak_omuz.write(i);
    delay(bekle);
  }
  for (int i = 110 ; i > 90 ; i--) {
    sol_on_ayak_diz.write(i);
    delay(bekle);
  }
}
