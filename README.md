# InclusiveHAR — Rastgele Orman ile İnsan Aktivitesi Tanıma

**BLM0463 Veri Madenciliğine Giriş — Dönem Projesi**

**Sınıflandırma:** *Decision Tree based Methods* ödev kategorisi kapsamında **yalnızca Rastgele Orman** — `proje.py` tek dosya.  
**Veri seti:** **InclusiveHAR** — *Data in Brief*, 2026; CSV: `DisabledHAR_dataset_v1.csv` (Mendeley DOI [10.17632/r78dn3f6nc.4](https://doi.org/10.17632/r78dn3f6nc.4)).

---

## 1. Özet

Akıllı telefon **24 IMU** özniteliği ile **6 sınıflı HAR**. Değerlendirme: **5 katlı StratifiedKFold** + `cross_val_predict` OOF; kaynak makaledeki örnek düzeyi train/test bölünmesiyle uyumlu, **UserID gruplu** katlama yok.

Son üretilen özet: `outputs/metrics/07_ozet.json` — örnek çalıştırmada **~0,982** OOF doğruluk ve makro-F1, **~146 s** eğitim süresi; üst öznitelikler: `motionYaw`, `magnetometerZ`, `magnetometerY`, `motionMagneticFieldZ`, `magnetometerX`.

---

## 2. Veri seti

- 396.602 örnek, 30 sensör + `label` + `UserID` + `disabled`
- 20 katılımcı, 6 aktivite
- Ayrıntı: [RAPOR.md](RAPOR.md)

---

## 3. Çıktı dosyaları

**`outputs/metrics/`** — `train` / `eda` ile yeniden üretilir; şu anki set:

| Dosya | İçerik |
|-------|--------|
| `00_veri_ozeti.txt` | Örnek sayısı, IMU sütun sayısı, kişi sayısı, eksik |
| `01_model_karsilastirma.csv` | RF + StratifiedKFold tek satır özet metrikler |
| `02_rf_rastgele_sinif_metrikleri.csv` | Sınıf başına P, R, Sp, F1 + makro/ağırlıklı |
| `04_engelli_kirilim.csv` | Engelli / engelsiz alt küme doğruluğu ve makro-F1 |
| `05_oznitelik_onemi.csv` | IMU öznitelik önem sıralaması |
| `06_kisi_bazli_dogruluk.csv` | UserID başına OOF doğruluk |
| `07_ozet.json` | Özet sayılar ve `protocol` |

**`outputs/figures/`** — `eda_01` … `eda_07` EDA; `fig_07` … `fig_17` eğitim ve sonuç grafikleri. Dosya adları `proje.py` içinde sabitlenmiştir.

---

## 4. Çalıştırma

```bash
pip install -r requirements.txt
# CSV: DisabledHAR_dataset_v1.csv proje kökünde olmalı

python proje.py              # explore → eda → train
python proje.py all
python proje.py explore
python proje.py eda
python proje.py train        # uzun sürebilir

python tools/export_rapor_docx.py   # isteğe bağlı
```

---

## 5. Yöntem

- **RF:** `n_estimators=120`, `min_samples_leaf=2`, Gini, `random_state=42`, `n_jobs=-1`.
- **OOF:** Tek `cross_val_predict(..., method='predict_proba')`; sınıf tahmini olasılık argmax ile; aynı vektör ROC / PR için kullanılır.
- **Görselleştirme:** EDA; özet çubuklar, karmaşıklık, F1, P/R, engelli kırılımı, kişi bazlı çubuklar, öznitelik önemi, GPS sızıntısı demosu, örnek ağaç, ROC, P–R, ısı haritası.

---

## 6. Sonuçlar

Güncel sayıları **`01_model_karsilastirma.csv`** ve **`07_ozet.json`** dosyalarından okuyun. Son kayıtlı OOF doğruluk **~%98,2**; engelli/engelsiz kırılımı **`04_engelli_kirilim.csv`**. Ayrıntılı metin: **[RAPOR.md](RAPOR.md)**.
