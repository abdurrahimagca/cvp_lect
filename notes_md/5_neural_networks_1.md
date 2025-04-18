# Yapay Sinir Ağları

## Temel Kavramlar

### Yapay Nöron (Perceptron)
Biyolojik nörondan esinlenilmiş yapay nöron modeli:

$y = f(\sum_{i=1}^n w_ix_i + b)$

Burada:
- $x_i$: Giriş değerleri
- $w_i$: Ağırlıklar
- $b$: Bias değeri
- $f$: Aktivasyon fonksiyonu

### Aktivasyon Fonksiyonları

1. **Sigmoid**:
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

2. **Tanh**:
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

3. **ReLU** (Rectified Linear Unit):
   $$f(x) = \max(0, x)$$

### İleri Yayılım (Forward Propagation)

Her katman için:
$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

### Geri Yayılım (Backpropagation)

Hata fonksiyonu $E$ için:
$$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial y_j} \frac{\partial y_j}{\partial net_j} \frac{\partial net_j}{\partial w_{jk}}$$

## Eğitim Stratejileri

### 1. Batch Gradient Descent
- Tüm veri seti üzerinden gradient hesaplanır
- Bellek kullanımı yüksek
- Daha kararlı öğrenme

### 2. Stochastic Gradient Descent (SGD)
- Her seferinde tek örnek
- Hızlı ama gürültülü öğrenme
- Düşük bellek kullanımı

### 3. Mini-batch Gradient Descent
- Batch ve SGD arası bir yaklaşım
- Batch_size örnek kullanılır
- Pratikte en çok kullanılan yöntem

## Optimizasyon Algoritmaları

### 1. Momentum
$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - v_t$$

### 2. Adam (Adaptive Moment Estimation)
Momentum ve RMSprop'un birleşimi:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta J(\theta)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta J(\theta))^2$$

## Düzenlileştirme (Regularization)

### 1. L1 Regularization
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n |\theta_i|$$

### 2. L2 Regularization
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n \theta_i^2$$

### 3. Dropout
- Eğitim sırasında rastgele nöronları devre dışı bırakma
- Overfitting'i önlemeye yardımcı olur

## Model Mimarisi Seçimi

### 1. Katman Sayısı
- Problem karmaşıklığına göre belirlenir
- Çok derin ağlar gradient vanishing/exploding problemi yaşayabilir

### 2. Nöron Sayısı
- Genelde piramit yapısı tercih edilir
- Giriş boyutu > Ara katmanlar > Çıkış boyutu

### 3. Aktivasyon Fonksiyonu Seçimi
- Gizli katmanlar: ReLU (genelde)
- Çıkış katmanı: Problem tipine göre
  - Sınıflandırma: Sigmoid/Softmax
  - Regresyon: Linear


