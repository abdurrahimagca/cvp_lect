# Lojistik Regresyon (Logistic Regression)

## Temel Kavramlar

### Sınıflandırma Problemi
- İkili sınıflandırma (Binary Classification)
- Çıktı: 0 veya 1
- Örnek: Spam tespiti, hastalık teşhisi

### Sigmoid Fonksiyonu
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Özellikleri:
- $(0,1)$ aralığında değer alır
- Türevi: $\sigma'(z) = \sigma(z)(1-\sigma(z))$

### Model
$$h_\theta(x) = \sigma(\theta^T x)$$
$$P(y=1|x;\theta) = h_\theta(x)$$
$$P(y=0|x;\theta) = 1-h_\theta(x)$$

## Hata Fonksiyonu

### Cross Entropy Loss
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

### Gradyan
$$\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

## Model Eğitimi

### Gradient Descent
$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$$

### Mini-batch Gradient Descent
1. Veriyi mini-batch'lere böl
2. Her mini-batch için:
   - Forward pass
   - Gradient hesapla
   - Parametreleri güncelle

## Model Değerlendirme

### Metrikler

#### 1. Accuracy
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

#### 2. Precision
$$Precision = \frac{TP}{TP + FP}$$

#### 3. Recall
$$Recall = \frac{TP}{TP + FN}$$

#### 4. F1 Score
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

### ROC Eğrisi
- True Positive Rate vs False Positive Rate
- AUC (Area Under Curve)
- Model karşılaştırması için kullanılır

## Regularization

### L1 Regularization (Lasso)
$$J_{new}(\theta) = J(\theta) + \lambda\sum_{j=1}^n |\theta_j|$$

### L2 Regularization (Ridge)
$$J_{new}(\theta) = J(\theta) + \lambda\sum_{j=1}^n \theta_j^2$$

## Çok Sınıflı Lojistik Regresyon

### One-vs-All Yaklaşımı
- Her sınıf için binary classifier
- $k$ sınıf için $k$ model
- Tahmin: $\arg\max_i(h_\theta^{(i)}(x))$

### Softmax Regresyon
$$P(y=j|x;\theta) = \frac{e^{\theta_j^T x}}{\sum_{k=1}^K e^{\theta_k^T x}}$$

## Uygulama İpuçları

### 1. Feature Engineering
- Normalizasyon
- Polynomial features
- Feature scaling

### 2. Hiperparametre Optimizasyonu
- Learning rate seçimi
- Regularization parametresi
- Mini-batch boyutu

### 3. Model Seçimi
- Linear separability kontrolü
- Kernel trick ihtiyacı
- Veri boyutu ve karmaşıklığı


