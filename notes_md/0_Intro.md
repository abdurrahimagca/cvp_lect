# Bilgisayarlı Görme ve Makine Öğrenmesine Giriş

## Ders Bilgileri

- Vize %30 + Proje (Final) %70
- Gereksinimler: Python, lineer cebir, kalkülüs
- İletişim: <ismail.parlak@ibu.edu.tr>, Oda: 335

### Kaynaklar

- «Computer Vision: Algorithms and Applications» - Richard Szeliski (<https://szeliski.org/Book/>)
- «Computer Vision: Foundations and Applications» - Ranjay Krishna
- <https://www.youtube.com/@patloeber/videos>
- <https://pytorch.org>

## Bilgisayarlı Görme Temelleri

### Görme Teorisi

İbnü'l Heysem (1040):
> "Gözışın Kuramı'na göre gözden ışık çıkmakta, nesneye ulaşabilmesi için saydam ortamdan geçerek görme eylemi gerçekleşmektedir. Oysa bütün ihtimaller dikkate alındığında, gözden ışığın çıkmasıyla değil, göz ışınlarının bakılan nesneye gidip ondan geri gelmesiyle görme gerçekleşir."

### Bilgisayarlı Görme Tanımı

- **Görü**: Nesnelerden yansıyan elektromanyetik dalgaların algılanması ve anlamlandırılması ile oluşturulan yaklaşık çevre modeli (Vural, ODTÜ).
- **Bilgisayarlı Görme**: İnsan görme sisteminin karmaşıklığının parçalarını kopyalamaya ve bilgisayarların, insanların yaptığı gibi görüntü ve videolardaki nesneleri tanımlayıp işlemesine odaklanan bilgisayar bilimi alanıdır (IBM).

### Dijital Görüntü Temelleri

- RGB (Kırmızı, Yeşil, Mavi) renk uzayı
- Piksel tabanlı temsil
- Çözünürlük ve bit derinliği kavramları

### Bilgisayarlı Görme Zorlukları

1. Aydınlatma değişimleri
2. Pozlama farklılıkları
3. Sınıf içi çeşitlilik
4. Engeller ve kısmi görünürlük
5. Anlamsal karmaşa
6. Arka plan karmaşıklığı
7. Hareketlilik

## Makine Öğrenmesine Giriş

### Makine Öğrenmesi Nedir?

- Veriden öğrenme yeteneği
- Açık programlanmadan pattern tanıma
- Deneyimden öğrenme ve iyileşme

### Öğrenme Türleri

#### 1. Denetimli Öğrenme (Supervised Learning)

- Etiketli veri kullanımı
- Giriş-çıkış eşleştirmesi
- Örnek: Sınıflandırma, regresyon

#### 2. Denetimsiz Öğrenme (Unsupervised Learning)

- Etiketsiz veri
- Pattern keşfi
- Örnek: Kümeleme, boyut indirgeme

#### 3. Pekiştirmeli Öğrenme (Reinforcement Learning)

- Ödül/ceza mekanizması
- Deneme-yanılma
- Örnek: Oyun AI, robot kontrolü

## Temel Problemler

### 1. Sınıflandırma (Classification)

- Kategorik çıktı
- İkili/çoklu sınıflandırma
- Örnekler:
  - Spam tespiti
  - Görüntü tanıma
  - Hastalık teşhisi

### 2. Regresyon (Regression)

- Sürekli çıktı
- Sayısal tahmin
- Örnekler:
  - Fiyat tahmini
  - Sıcaklık tahmini
  - Satış tahmini

### 3. Kümeleme (Clustering)

- Benzer grupları bulma
- Otomatik kategorilendirme
- Örnekler:
  - Müşteri segmentasyonu
  - Döküman gruplandırma
  - Görüntü bölütleme

## Model Değerlendirme

### 1. Performans Metrikleri

- Accuracy
- Precision/Recall
- F1 Score
- ROC/AUC
- MSE/RMSE/MAE

### 2. Cross-Validation

- K-fold CV
- Hold-out validation
- Time series split

### 3. Overfitting/Underfitting

- Bias-Variance trade-off
- Regularization
- Model karmaşıklığı

## Veri Hazırlama

### 1. Veri Temizleme

- Missing values
- Outlier detection
- Noise reduction

### 2. Feature Engineering

- Feature extraction
- Feature selection
- Feature scaling

### 3. Veri Augmentasyonu

- Veri çoğaltma
- Noise ekleme
- Transformasyonlar

## Pratik Uygulamalar

### 1. Python Kütüphaneleri

- NumPy: Sayısal işlemler
- Pandas: Veri manipülasyonu
- Scikit-learn: ML algoritmaları
- TensorFlow/PyTorch: Derin öğrenme

### 2. Proje Akışı

```python
# 1. Veri yükleme
import pandas as pd
data = pd.read_csv('data.csv')

# 2. Veri önişleme
X = preprocess_data(data)

# 3. Model eğitimi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 4. Model değerlendirme
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```

### 3. Best Practices

- Version control
- Pipeline automation
- Documentation
- Model monitoring
