# Evrişimsel Sinir Ağları (Convolutional Neural Networks)

## Temel Kavramlar

### Evrişim İşlemi (Convolution)
- 2B görüntü üzerinde filtre/kernel gezdirme
- Matematiksel ifade: $(f * g)(x,y) = \sum_{i}\sum_{j} f(i,j)g(x-i,y-j)$

### CNN Katmanları

#### 1. Evrişim Katmanı (Convolutional Layer)
- Filtre boyutu: $k \times k$
- Stride: $s$
- Padding: $p$
- Çıktı boyutu: $\lfloor \frac{n + 2p - k}{s} + 1 \rfloor$

#### 2. Aktivasyon Katmanı
- Genelde ReLU kullanılır
- $f(x) = \max(0,x)$

#### 3. Havuzlama Katmanı (Pooling Layer)
- Max Pooling
- Average Pooling
- Çıktı boyutu: $\lfloor \frac{n}{s} \rfloor$

#### 4. Tam Bağlantılı Katman (Fully Connected Layer)
- Klasik yapay sinir ağı katmanı
- Sınıflandırma için son katmanlarda kullanılır

## CNN Mimarisi Örnekleri

### 1. LeNet-5 (1998)
```
Input -> Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output
```

### 2. AlexNet (2012)
- 5 konvolüsyon katmanı
- 3 tam bağlantılı katman
- ReLU aktivasyonu
- Dropout regularization

### 3. VGG-16 (2014)
- 13 konvolüsyon katmanı
- 3 tam bağlantılı katman
- 3x3 filtreler
- Max pooling

## İleri Düzey Kavramlar

### 1. Transfer Learning
- Önceden eğitilmiş model kullanımı
- Feature extraction
- Fine-tuning

### 2. Data Augmentation
- Görüntü döndürme
- Yatay/dikey çevirme
- Zoom in/out
- Parlaklık/kontrast ayarı

### 3. Bottleneck Features
- Son katmanlar öncesi özellikler
- Transfer learning'de kullanışlı
- Hesaplama verimliliği

## Uygulama Alanları

### 1. Görüntü Sınıflandırma
- ImageNet yarışması
- Nesne tanıma
- Yüz tanıma

### 2. Nesne Tespiti
- YOLO
- R-CNN
- SSD

### 3. Semantic Segmentation
- U-Net
- FCN
- DeepLab

## Optimizasyon ve Regularizasyon

### Batch Normalization
$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}$$

### Dropout
- Feature detektorlerin co-adaptation'ını önler
- Test zamanında scaling gerektirir

### Learning Rate Scheduling
- Step decay
- Exponential decay
- Cosine annealing

## Best Practices

### 1. Model Seçimi
- Problem karmaşıklığına göre
- Veri seti büyüklüğüne göre
- Hesaplama kaynaklarına göre

### 2. Hiperparametre Optimizasyonu
- Learning rate
- Batch size
- Optimizer seçimi
- Regularization parametreleri

### 3. Veri Hazırlığı
- Normalize etme
- Augmentation
- Train/Val/Test split


