# Zaman Serisi Anomali Tespiti

Bu proje, LSTM tabanlı bir otomatik kod çözücü (autoencoder) kullanarak zaman serisi verisindeki anomali örneklerini tespit etmeyi amaçlar. Verideki olağandışı davranışlar, yeniden oluşturma hatası (reconstruction error) ile belirlenir.

## Proje Amacı

Zaman serisi verilerinde normal davranıştan sapma gösteren örnekleri bulmak için aşağıdaki akış kullanılır:

- Sentetik zaman serisi verisi oluşturma
- Veriyi ölçeklendirme
- Kayma pencereleri (sliding windows) oluşturma
- LSTM tabanlı autoencoder ile model eğitme
- Öğrenilen normal davranıştan sapmaları anomali olarak işaretleme

## Kullanılan Teknolojiler

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- joblib

## Proje Yapısı

- `load_and_explore.py` : Sentetik zaman serisi verisini üretir ve görselleştirir.
- `preprocessing.py` : Veriyi normalize eder, kayan pencere dizilerini oluşturur ve `x_train.npy` ile `scaler.save` dosyalarını üretir.
- `model.py` : LSTM autoencoder modelini eğitir ve `autoEncoder.h5` kaydeder.
- `test.py` : Eğitilmiş modeli kullanarak anomali tespiti yapar ve sonuçları CSV olarak kaydeder.
- `synhetic_series.csv` : Üretilmiş zaman serisi verisi.
- `x_train.npy` : Model eğitimi için hazırlanmış pencereli veri.
- `scaler.save` : MinMaxScaler modeli.
- `autoEncoder.h5` : Eğitilmiş LSTM autoencoder.
- `anomaly_results.csv` : Tespit edilen anomali sonuçları.

## Gereksinimler

Aşağıdaki paketlerin kurulu olması gerekir:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn joblib
```

## Çalıştırma Adımları

Sıra önemli olduğu için aşağıdaki adımları izleyin:

### 1) Sentetik veri üretimi

```bash
python .\load_and_explore.py
```

Bu komut `synhetic_series.csv` dosyasını oluşturur.

### 2) Veri ön işleme

```bash
python .\preprocessing.py
```

Bu adım:

- veriyi normalize eder,
- kayan pencereleri oluşturur,
- `x_train.npy` ve `scaler.save` dosyalarını üretir.

### 3) Model eğitimi

```bash
python .\model.py
```

Bu adım:

- LSTM autoencoder modelini oluşturur,
- modeli eğitir,
- eğitim kaybı grafiğini gösterir,
- `autoEncoder.h5` dosyasını kaydeder.

### 4) Anomali tespiti

```bash
python .\test.py
```

Bu adım:

- eğitilmiş modeli yükler,
- aynı ölçeklendirme işlemini uygular,
- tüm pencere örneklerini modelden geçirir,
- yeniden oluşturma hatasına göre eşik değeri belirler,
- anomali pozisyonlarını işaretler,
- sonuçları `anomaly_results.csv` olarak kaydeder.

## Model Mantığı

Bu proje, normal davranış örneklerini öğrenen bir otomatik kod çözücü kullanır. Modelin eğitim sırasında verdiği hata, öğrenilen normal örüntüden sapma gösteren veriler için artar.

Bu nedenle:

- normal örnekler düşük hata ile yeniden üretilir,
- anomali örnekleri yüksek yeniden oluşturma hatası üretir,
- 95. persentil eşik değeri kullanılarak anomali tespit edilir.

## Çıktılar

### Eğitim çıktısı

- `autoEncoder.h5`
- Eğitim kaybı grafiği (terminal veya matplotlib penceresi üzerinden)

### Tespit çıktısı

- `anomaly_results.csv`
- Anomalilerin işaretlendiği zaman serisi grafiği

## Notlar

- Bu proje sentetik bir veri seti üzerinde çalışmaktadır.
- Gerçek zaman serisi verileri için `synhetic_series.csv` yerine kendi veriniz kullanılır.
- Anomaly eşik değeri `np.percentile(mse_list, 95)` ile hesaplanmaktadır; gerekirse bu eşik değeri farklı veri setleri için ayarlanabilir.

## Geliştirme Önerileri

- Farklı zaman serisi verileri ile test edin.
- Eşik değeri için IQR, z-score veya dinamik eşik yöntemleri ekleyin.
- Daha karmaşık veri için GRU veya Transformer tabanlı modeller deneyin.
- Anomali sonuçlarını görselleştirmek için web tabanlı dashboard ekleyin.

---

Proje, zaman serisi tabanlı anomali tespiti için basit ve anlaşılır bir örnek olarak tasarlanmıştır. İsterseniz ileride bunu daha profesyonel bir README’ye dönüştürebilir veya proje için kullanım ve açıklama eklemelerini Türkçe/İngilizce olarak da düzenleyebilirim.
