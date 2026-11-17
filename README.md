# webgaze

Tobii benzeri göz takibi, sadece web kamerasıyla (Windows). MediaPipe Face Mesh ile iris konumu bulunur,
kişiye özel kalibrasyonla ekran koordinatına çevrilir.

## Özellikler
- 13 noktalı kalibrasyon, 9 noktalı doğrulama (hata px / cm / derece, titreme, bölge isabet oranı)
- İlgi alanları (`--aoi`): `grid:3x3`, elle tanımlı JSON (`aoi_ornek.json`) veya ekrandan otomatik (`auto`)
- Bölgeye oturtma ve titreme bastırma; CSV kaydına bölge adı yazılır
- Tıklamalarla sürekli kalibrasyon
- Ekran üstünde tıklanamaz bakış göstergesi, bakışla fare kontrolü, OpenTrack'e baş takibi (UDP)
- `analyze.py`: kaydedilen kalibrasyon verisinde farklı model ayarlarını karşılaştırır

## Kurulum ve çalıştırma
Python 3.10–3.12 gerekir. `run.bat` ilk açılışta ortamı kendisi kurar.

```bat
run.bat
run.bat --aoi grid:3x3
run.bat --camera 1 --width 1280 --height 720 --distance 45
```

## Kısayollar
F5 bölgeler · F6 baş merkez · F7 gösterge · F8 bakışla fare · F9 yeniden kalibrasyon · F10 çıkış

## Doğruluk
Web kamerasıyla tipik hata 2–3°'dir (Tobii ~0.5–1°). 21.5" ekran, 45 cm uzaklık ve dahili kamerayla
ölçülen: ~63 px (~2°), 3×3 ekran bölgesinde ~%93 isabet. Baş sabit tutulmalı; çene desteği ve önden ışık
doğruluğu belirgin artırır.
