6607115282023

Bilgisayarlı Görme Uygulamaları

ismail.parlak@ibu.edu.tr
Oda: 335

Ders Bilgileri

• Vize %30 + Proje (Final) %70
• Python, lineer cebir, kalkülüs
• Kaynaklar

• «Computer Vision: Algorithms and Applications» -

Richard Szeliski ( https://szeliski.org/Book/ )

• «Computer Vision: Foundations and Applications» -

Ranjay Krishna

• https://www.youtube.com/@patloeber/videos
• https://pytorch.org

1

BG (CV) Uygulamaları

Mihajlovic, towardsdatascience

2

BG (CV) Uygulamaları

Rizzoli, v7labs

3

BG (CV) Uygulamaları

Rizzoli, v7labs

4

BG (CV) Uygulamaları

Rizzoli, v7labs

5

BG (CV) Uygulamaları

Rizzoli, v7labs

6

BG (CV) Uygulamaları

Rizzoli, v7labs

7

BG (CV) Uygulamaları

Rizzoli, v7labs

8

BG (CV) Uygulamaları

Rizzoli, v7labs

9

BG (CV) Uygulamaları

Rizzoli, v7labs

10

BG (CV) Uygulamaları

Rizzoli, v7labs

11

BG (CV) Uygulamaları

Fei Fei, Stanford

12

Görme

İbnü’l Heysem (1040)

"Gözışın Kuramı'na göre gözden ışık çıkmakta,
nesneye ulaşabilmesi için saydam ortamdan
geçerek görme eylemi gerçekleşmektedir. Oysa
bütün ihtimaller dikkate alındığında, gözden
ışığın çıkmasıyla değil, göz ışınlarının bakılan
nesneye gidip ondan geri gelmesiyle görme
gerçekleşir."

13

Camera Obscura (Karanlık Oda)

De Radio Astronomica et Geometrica, Gemma Frisius, 1545

14

Görü ve Bilgisayarlı Görme

• Görü: Nesnelerden yansıyan elektromanyetik
dalgaların algılanması ve anlamlandırılması ile
oluşturulan yaklaşık çevre modeli (Vural, ODTÜ).
• Bilgisayarlı Görme (Bilgisayarlı Görü, Computer

Vision - CV): İnsan görme sisteminin karmaşıklığının
parçalarını kopyalamaya ve bilgisayarların,
insanların yaptığı gibi görüntü ve videolardaki
nesneleri tanımlayıp işlemesine odaklanan
bilgisayar bilimi alanıdır (IBM).

15

Dijital Resim

16

Dijital Resim

(R, G, B)

commons.wikimedia.org

17

Karakter Tanıma

code.org

18

Karakter Tanıma

code.org

19

Karakter Tanıma

code.org

20

Karakter Tanıma

code.org

21

Zorluklar - Aydınlatma

Fridman, MIT

22

Zorluklar - Pozlama

Fridman, MIT

23

Zorluklar - Sınıf içi Çeşitlilik

Fridman, MIT

24

Zorluklar - Engeller

Fridman, MIT

25

Zorluklar - Anlamsal Karmaşa

Fridman, MIT

26

Zorluklar - Arka Plan

Duygulu, Bilkent

27

Zorluklar - Hareketlilik

Duygulu, Bilkent

28

ImageNet

22K kategori (sınıf), 15M resim (Fei Fei)

​NASNET-A(6)
NASNET-A(6)

​Inception V3
Inception V3

​CoAtNet-7
CoAtNet-7

​87.79
87.79

​87.73
87.73

​SPPNet
SPPNet

​Five Base + Five HiRes
Five Base + Five HiRes

​AlexNet
AlexNet

​SIFT + FVs
SIFT + FVs

Y
C
A
R
U
C
C
A
1

P
O
T

100

90

80

70

60

50

40

2012

2014

2016

2018

2020

2022

Other models

State-of-the-art models

https://paperswithcode.com/sota/image-classification-on-imagenet

29

ImageNet - Sınıflandırma Hataları

İnsan seviyesi hata: %5

https://www.kaggle.com/getting-started/149448

30

İnsan vs. BG

http://karpathy.github.io/2012/10/22/state-of-computer-vision/

31

# Makine Öğrenmesine Giriş

## Temel Kavramlar

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


