# Doğrusal Regresyon (Linear Regression)

## Temel Kavramlar

### Veri seti ve Hipotez
- Makine öğrenmesi modelimiz ev fiyatlarını tahmin etmeye çalışacak
- Girdi: Ev alanı
- Çıktı: Tahmini Fiyat

### Matematiksel Model

Doğrusal regresyon modeli şu şekilde ifade edilir:

- Basit form: $Fiyat = a \times Alan + b$
- Matematiksel form: $y = ax + b$
- Makine öğrenmesi notasyonu: $h_\theta(x) = \theta_0 + \theta_1x$

Burada $\theta_i$ parametrelerdir.

### Hata Fonksiyonu (Cost Function)

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

Hedefimiz: $J(\theta_0, \theta_1)$ fonksiyonunu minimize etmek

### Dereceli Alçalma (Gradient Descent)

Basitleştirilmiş model için ($\theta_0 = 0$):
- Hipotez: $h_\theta(x) = \theta_1x$
- Hata fonksiyonu: $J(\theta_1) = \frac{1}{2m} \sum_{i=1}^m (\theta_1x^{(i)} - y^{(i)})^2$

Güncelleme kuralı:
$$\theta_1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1}J(\theta_1)$$

Burada $\alpha$ öğrenme hızıdır (learning rate).

### Öğrenme Hızı (Learning Rate)

Öğrenme hızı $\alpha$ seçimi önemlidir:
- Çok büyük olursa: Yakınsama olmayabilir
- Çok küçük olursa: Çok yavaş öğrenme

### Tam Gradient Descent Algoritması

Genel model için güncelleme kuralları:

$$\frac{\partial}{\partial\theta_0}J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^m (\theta_0 + \theta_1x^{(i)} - y^{(i)})$$

$$\frac{\partial}{\partial\theta_1}J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^m (\theta_0 + \theta_1x^{(i)} - y^{(i)}) \cdot x^{(i)}$$

## Model Eğitimi Stratejileri

### 1. Tüm Veri ile Eğitim
- Tüm veriyi eğitim için kullanma

### 2. Train-Validation Split
- Eğitim (Train)
- Doğrulama (Validation)

### 3. Train-Validation-Test Split
- Eğitim (Train)
- Doğrulama (Validation)
- Test

### 4. Çapraz Doğrulama (Cross Validation)
K-fold cross validation yaklaşımı:
- Veri K parçaya bölünür
- Her seferinde bir parça test için ayrılır
- Kalan parçalar eğitim için kullanılır
- K farklı model eğitilir

## Makine Öğrenmesi Genel Akışı

```python
# Veri hazırlama
X, y = veriyi_hazirla(veri_seti)

# Veri bölme
X_train, X_val, X_test, y_train, y_val, y_test = veriyi_bol(X, y, 0.7, 0.2, 0.1)

# Model oluşturma ve eğitme
model = DogrusalRegresyonModeli()
model.fit(X_train, y_train)

# Doğrulama
y_tahmin = model.predict(X_val)
basari = basari_olc(y_tahmin, y_val)

# Test
nihai_basari = basari_olc(model.predict(X_test), y_test)


