# K-En Yakın Komşu (K-Nearest Neighbors)

## Algoritma Temelleri

### Çalışma Prensibi
1. Yeni örnek için tüm eğitim örnekleriyle uzaklık hesapla
2. En yakın K komşuyu bul
3. Çoğunluk oylaması ile sınıf tahmin et

### Uzaklık Metrikleri

#### 1. Öklid Uzaklığı
$$d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

#### 2. Manhattan Uzaklığı
$$d(x,y) = \sum_{i=1}^n |x_i - y_i|$$

#### 3. Minkowski Uzaklığı
$$d(x,y) = (\sum_{i=1}^n |x_i - y_i|^p)^{\frac{1}{p}}$$

## Algoritma Karmaşıklığı

### Eğitim Aşaması
- Zaman Karmaşıklığı: $O(1)$
- Uzay Karmaşıklığı: $O(n)$
- Lazy learning algoritması

### Tahmin Aşaması
- Zaman Karmaşıklığı: $O(nd)$
  - $n$: eğitim örneği sayısı
  - $d$: özellik sayısı
- Uzay Karmaşıklığı: $O(k)$

## Hiperparametreler

### 1. K Değeri Seçimi
- Çok küçük K: Overfitting
- Çok büyük K: Underfitting
- Genelde tek sayı tercih edilir
- Cross-validation ile optimize edilir

### 2. Ağırlıklı Oylama
$$w_i = \frac{1}{d(x,x_i)^2}$$
- Uzaklığa bağlı ağırlıklandırma
- Yakın komşulara daha fazla önem

## Veri Ön İşleme

### 1. Feature Scaling
$$x_{norm} = \frac{x - \mu}{\sigma}$$
- Standart scaler
- Min-max scaler
- Robust scaler

### 2. Boyut İndirgeme
- PCA
- t-SNE
- UMAP

## Optimizasyon Teknikleri

### 1. Ball Tree
- Uzay bölümleme
- Küresel sınırlar
- Logaritmik arama zamanı

### 2. KD Tree
- Uzay bölümleme
- Dikdörtgen sınırlar
- Dengeli ağaç yapısı

### 3. Approximate Nearest Neighbors
- LSH (Locality Sensitive Hashing)
- Tahmini en yakın komşular
- Hız-doğruluk ödünleşimi

## Uygulama Örnekleri

### 1. Sınıflandırma
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=k)
```

### 2. Regresyon
```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=k)
```

### 3. Anomali Tespiti
```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=k)
```


