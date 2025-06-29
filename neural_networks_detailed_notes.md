# Sinir Ağları ve Derin Öğrenme - Detaylı Notlar

## 1. Temel Yapay Sinir Ağları

### Yapay Nöron (Perceptron)
Yapay nöron, biyolojik nörondan esinlenilmiş temel yapı taşıdır:

**Matematiksel Model:**
$$y = f(\sum_{i=1}^n w_ix_i + b)$$

**Bileşenler:**
- **$x_i$**: Giriş değerleri (özellikler)
- **$w_i$**: Ağırlıklar (parametreler)
- **$b$**: Bias değeri (sapma)
- **$f$**: Aktivasyon fonksiyonu

### Aktivasyon Fonksiyonları

#### 1. Sigmoid Fonksiyonu
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Çıkış aralığı: (0, 1)
- Lojistik regresyon için ideal
- Gradient vanishing problemi

#### 2. Tanh Fonksiyonu
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- Çıkış aralığı: (-1, 1)
- Sigmoid'den daha iyi performans
- Yine gradient vanishing sorunu

#### 3. ReLU (Rectified Linear Unit)
$$f(x) = \max(0, x)$$
- Basit ve hızlı hesaplama
- Gradient vanishing yok
- Dying ReLU problemi

## 2. İleri ve Geri Yayılım Algoritmaları

### İleri Yayılım (Forward Propagation)
Her katman için ardışık hesaplamalar:

**Katman l için:**
$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

Burada:
- $z^{[l]}$: Katman l'nin doğrusal çıkışı
- $a^{[l]}$: Katman l'nin aktivasyon çıkışı
- $W^{[l]}$: Katman l'nin ağırlık matrisi
- $b^{[l]}$: Katman l'nin bias vektörü

### Geri Yayılım (Backpropagation)
Hata fonksiyonu gradyanlarının hesaplanması:

**Temel Kural:**
$$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial y_j} \frac{\partial y_j}{\partial net_j} \frac{\partial net_j}{\partial w_{jk}}$$

**Zincir Kuralı Uygulaması:**
1. Son katmandan başlayarak hata hesaplama
2. Gradyanları önceki katmanlara yayma
3. Parametreleri güncelleme

## 3. Optimizasyon Stratejileri

### Gradient Descent Türleri

#### 1. Batch Gradient Descent
```
Tüm veri seti → Gradient hesaplama → Parametre güncelleme
```
- **Avantaj**: Kararlı öğrenme
- **Dezavantaj**: Yüksek bellek kullanımı

#### 2. Stochastic Gradient Descent (SGD)
```
Tek örnek → Gradient hesaplama → Parametre güncelleme
```
- **Avantaj**: Hızlı, düşük bellek
- **Dezavantaj**: Gürültülü öğrenme

#### 3. Mini-batch Gradient Descent
```
Batch_size örnek → Gradient hesaplama → Parametre güncelleme
```
- **Avantaj**: Batch ve SGD'nin dengesi
- **En yaygın kullanılan yöntem**

### Gelişmiş Optimizasyon Algoritmaları

#### 1. Momentum
$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - v_t$$
- Önceki gradyanların ağırlıklı ortalaması
- Salınımları azaltır
- Hızlanma etkisi

#### 2. Adam (Adaptive Moment Estimation)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta J(\theta)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta J(\theta))^2$$
- Momentum ve RMSprop birleşimi
- Adaptif öğrenme hızı
- Genellikle iyi performans

## 4. Düzenlileştirme (Regularization) Teknikleri

### 1. L1 Regularization (Lasso)
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n |\theta_i|$$
- Sparsity (seyreklik) sağlar
- Özellik seçimi etkisi
- Bazı ağırlıkları sıfırlar

### 2. L2 Regularization (Ridge)
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n \theta_i^2$$
- Ağırlıkları küçültür
- Overfitting'i önler
- Yaygın kullanım

### 3. Dropout
- Eğitim sırasında rastgele nöronları devre dışı bırakma
- Co-adaptation'ı önler
- Test zamanında scaling gerektirir
- Overfitting'e karşı güçlü koruma

## 5. Model Mimarisi ve Tasarım

### Katman Sayısı Seçimi
- **Shallow Networks**: Basit problemler
- **Deep Networks**: Karmaşık pattern'ler
- **Gradient Vanishing/Exploding** riski

### Nöron Sayısı Stratejisi
```
Giriş Boyutu → [Büyük] → [Orta] → [Küçük] → Çıkış Boyutu
```
- Piramit yapısı tercih edilir
- Capacity kontrolü
- Overfitting dengesi

## 6. Evrişimsel Sinir Ağları (CNN)

### Temel Konseptler

#### Evrişim İşlemi
$$$(f * g)(x,y) = \sum_{i}\sum_{j} f(i,j)g(x-i,y-j)$$
- 2D görüntü üzerinde filtre/kernel gezdirme
- Lokal pattern detection
- Parameter sharing

#### CNN Katman Türleri

**1. Evrişim Katmanı (Convolutional Layer)**
$$\text{Output Size} = \lfloor \frac{n + 2p - k}{s} + 1 \rfloor$$
- **n**: Giriş boyutu
- **k**: Kernel boyutu
- **s**: Stride
- **p**: Padding

**2. Havuzlama Katmanı (Pooling Layer)**
$$\text{Output Size} = \lfloor \frac{n}{s} \rfloor$$
- **Max Pooling**: Maksimum değer seçimi
- **Average Pooling**: Ortalama değer hesaplama
- Boyut azaltma ve translation invariance

**3. Tam Bağlantılı Katman (Fully Connected)**
- Klasik MLP katmanı
- Sınıflandırma için son katmanlarda
- Feature'ları birleştirme

### Ünlü CNN Mimarileri

#### 1. LeNet-5 (1998)
```
Input → Conv → Pool → Conv → Pool → FC → FC → Output
```
- İlk başarılı CNN
- El yazısı rakam tanıma
- MNIST veri seti

#### 2. AlexNet (2012)
- 5 konvolüsyon + 3 tam bağlantılı katman
- ReLU aktivasyon fonksiyonu
- Dropout regularization
- ImageNet yarışması kazananı

#### 3. VGG-16 (2014)
- 13 konvolüsyon + 3 tam bağlantılı katman
- Sadece 3x3 filtreler
- Daha derin ağ yapısı
- Basit ve tutarlı mimari

### İleri Düzey CNN Teknikleri

#### 1. Transfer Learning
- **Feature Extraction**: Önceden eğitilmiş modelin feature'larını kullanma
- **Fine-tuning**: Tüm parametreleri yeniden eğitme
- Az veri durumunda etkili
- Hesaplama maliyeti düşürme

#### 2. Data Augmentation
- Görüntü döndürme (rotation)
- Yatay/dikey çevirme (flipping)
- Zoom in/out
- Parlaklık/kontrast ayarı
- Veri çeşitliliği artırma

#### 3. Batch Normalization
$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}$$
- Internal covariate shift azaltma
- Daha hızlı öğrenme
- Regularization etkisi

## 7. Transposed Convolution (Deconvolution)

### Temel Konsept
$$y = W^T x$$
- Normal konvolüsyonun tersi işlem
- Upsampling (yukarı örnekleme)
- Encoder-Decoder mimarilerinde kullanım

### Boyut Hesaplama
```
Normal Konvolüsyon:      n×n * f×f → (n-f+1)×(n-f+1)
Transposed Konvolüsyon:  n×n → (n+f-1)×(n+f-1)
```

### Hiperparametreler
$$\text{output\_size} = (input\_size - 1) \times stride + kernel\_size - 2 \times padding$$

### Uygulama Alanları

#### 1. Otoenkoderler
```
Input → Encoder → Latent Space → Decoder → Output
      ↓           ↓              ↓         ↓
    Conv        Conv        TransConv  TransConv
```

#### 2. Semantic Segmentation
- Pixel-wise sınıflandırma
- U-Net mimarisi
- Skip connections

#### 3. Super Resolution
- Düşük çözünürlükten yüksek çözünürlüğe
- Detail generation
- Quality enhancement

### Checkerboard Artifacts Problemi
**Problem**: Düzensiz örtüşme ve görsel artifaktlar

**Çözümler**:
1. **Resize + Convolution**
2. **Pixel Shuffle**
3. **Bilinear Upsampling**

## 8. Pratik Uygulama Önerileri

### Model Seçimi Kriterleri
1. **Problem Karmaşıklığı**: Basit → MLP, Karmaşık → CNN
2. **Veri Seti Boyutu**: Az veri → Transfer Learning
3. **Hesaplama Kaynakları**: Sınırlı → Daha küçük modeller

### Hiperparametre Optimizasyonu
- **Learning Rate**: 0.001-0.01 arası başlangıç
- **Batch Size**: 32, 64, 128 (GPU belleğine göre)
- **Optimizer**: Adam (genellikle iyi başlangıç)
- **Regularization**: Dropout 0.2-0.5

### Veri Hazırlığı
1. **Normalization**: [0,1] veya [-1,1] aralığına getirme
2. **Standardization**: Ortalama=0, Standart sapma=1
3. **Train/Val/Test Split**: 70%/15%/15% tipik oran
4. **Cross-validation**: K-fold (k=5 veya k=10)

### Performans Değerlendirme
- **Sınıflandırma**: Accuracy, Precision, Recall, F1-Score
- **Regresyon**: MSE, MAE, R²
- **Confusion Matrix**: Detaylı analiz
- **Learning Curves**: Overfitting tespiti

## 9. Yaygın Problemler ve Çözümleri

### Overfitting
- **Belirtiler**: Train accuracy yüksek, validation accuracy düşük
- **Çözümler**: Dropout, L1/L2 regularization, veri artırma

### Underfitting
- **Belirtiler**: Hem train hem validation accuracy düşük
- **Çözümler**: Model kapasitesi artırma, daha uzun eğitim

### Gradient Vanishing
- **Belirtiler**: Derin ağlarda öğrenme yavaşlaması
- **Çözümler**: ReLU aktivasyon, Batch Normalization, ResNet

### Gradient Exploding
- **Belirtiler**: Loss değerlerinde ani artışlar
- **Çözümler**: Gradient clipping, learning rate azaltma

Bu detaylı notlar, temel sinir ağlarından modern CNN mimarilerine kadar geniş bir yelpazede derin öğrenme konseptlerini kapsamaktadır.