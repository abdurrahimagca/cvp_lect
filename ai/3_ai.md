# Lojistik Regresyon (Logistic Regression) - Teorik ve Uygulamalı Anlatım

Bu belge, lojistik regresyon algoritmasının hem teorik temellerini hem de Python ile pratik uygulamasını bir araya getirmektedir. `/home/apo/Code/tmp/cvp_lect/notes_md/3_log_reg.md` dosyasındaki teorik bilgiler ve `/home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb` notebook'undaki MNIST veri seti ile 3 ve 7 rakamlarını sınıflandırma örneği kullanılmıştır.

## 1. Temel Kavramlar (Teori)

### 1.1. Sınıflandırma Problemi
Doğrusal regresyonun aksine, lojistik regresyon bir **sınıflandırma (classification)** algoritmasıdır. Amacı, girdileri önceden tanımlanmış kategorilere veya sınıflara ayırmaktır.

-   **İkili Sınıflandırma (Binary Classification):** En yaygın türüdür. Çıktı sadece iki olası sınıftan biridir (genellikle 0 veya 1 olarak kodlanır).
    -   Örnekler: Bir e-postanın spam olup olmadığını belirleme (1: Spam, 0: Spam değil), bir hastanın belirli bir hastalığa sahip olup olmadığını teşhis etme (1: Hasta, 0: Sağlıklı), bir müşterinin bir ürünü satın alıp almayacağını tahmin etme (1: Satın alır, 0: Satın almaz).
-   **Çok Sınıflı Sınıflandırma (Multi-class Classification):** İkiden fazla olası sınıf vardır.
    -   Örnekler: El yazısı rakamları tanıma (0-9 arası sınıflar), bir resimdeki nesneyi tanıma (kedi, köpek, araba vb.).

Bu belgede öncelikle ikili sınıflandırma üzerine odaklanacağız.

### 1.2. Sigmoid (Lojistik) Fonksiyonu
Doğrusal regresyonda hipotez fonksiyonu $h_\theta(x) = \theta^T x$ idi ve çıktı herhangi bir reel sayı olabilirdi. Ancak sınıflandırmada, çıktının genellikle bir olasılık değeri (0 ile 1 arasında) olması istenir. Bu dönüşümü sağlamak için **sigmoid** veya **lojistik fonksiyon** kullanılır:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Burada $z$, genellikle doğrusal bir kombinasyondur ($z = \theta^T x$).

**Sigmoid Fonksiyonunun Özellikleri:**
-   Girdi olarak herhangi bir reel sayıyı ($z$) alır.
-   Çıktı her zaman $(0, 1)$ aralığındadır.
-   $z=0$ iken $\sigma(z) = 0.5$'tir.
-   $z$ çok büyük pozitif değerler aldığında $\sigma(z)$ 1'e yaklaşır.
-   $z$ çok büyük negatif değerler aldığında $\sigma(z)$ 0'a yaklaşır.
-   Bu özellikleri sayesinde, sigmoid fonksiyonunun çıktısı bir olasılık olarak yorumlanabilir (örneğin, $y=1$ olma olasılığı).
-   Türevi kolayca hesaplanabilir: $\sigma'(z) = \sigma(z)(1-\sigma(z))$. Bu, gradient descent hesaplamalarında önemlidir.

```python
# Sigmoid fonksiyonunun Python implementasyonu
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

### 1.3. Model (Hipotez Fonksiyonu)
Lojistik regresyonda hipotez fonksiyonu, doğrusal modelin çıktısını sigmoid fonksiyonuna sokarak elde edilir:

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Bu hipotez fonksiyonu $h_\theta(x)$, verilen $x$ girdisi için $y=1$ olma olasılığını tahmin eder:

$$P(y=1|x;\theta) = h_\theta(x)$$

$y=0$ olma olasılığı ise:

$$P(y=0|x;\theta) = 1 - h_\theta(x)$$

Tahmin yapmak için genellikle bir eşik değeri (threshold) kullanılır (varsayılan olarak 0.5):
-   Eğer $h_\theta(x) \ge 0.5$ ise, tahmin $y=1$'dir.
-   Eğer $h_\theta(x) < 0.5$ ise, tahmin $y=0$'dır.

(Not: $\sigma(z) \ge 0.5$ olması, $z = \theta^T x \ge 0$ olmasıyla eşdeğerdir.)

## 2. Hata Fonksiyonu (Cost Function)

Doğrusal regresyonda kullanılan Ortalama Kare Hata (MSE) fonksiyonu, lojistik regresyonda (sigmoid fonksiyonu nedeniyle) konveks olmayan bir yapıya yol açar. Bu, gradient descent'in global minimum yerine yerel minimumlara takılmasına neden olabilir. Bu yüzden lojistik regresyon için farklı bir hata fonksiyonu kullanılır: **Çapraz Entropi Kaybı (Cross-Entropy Loss)** veya **Log Kaybı (Log Loss)**.

**Sezgisel Anlatım:** Çapraz entropi, iki olasılık dağılımı arasındaki farkı ölçer. Bizim durumumuzda, bu dağılımlar gerçek etiketler (0 veya 1) ve modelimizin tahmin ettiği olasılıklardır ($h_\theta(x)$). Modelin tahmini gerçek etiketten uzaklaştıkça (örn. gerçek 1 iken tahmin 0'a yakınsa), kayıp hızla artar ve modeli doğru yöne doğru daha güçlü bir şekilde iter.

**Maksimum Olabilirlik Tahmini (Maximum Likelihood Estimation - MLE) Bağlantısı:** Lojistik regresyonun çapraz entropi hata fonksiyonunu minimize etmek, aslında veri setinin gözlemlenme olasılığını maksimize etmekle (MLE prensibi) eşdeğerdir.

Tek bir örnek için kayıp:
$$Cost(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1-h_\theta(x))$$

Bu fonksiyonun özellikleri:
-   Eğer gerçek $y=1$ ise, kayıp $- \log(h_\theta(x))$ olur. Model $y=1$'i doğru tahmin ederse ($h_\theta(x) \approx 1$), kayıp 0'a yaklaşır. Yanlış tahmin ederse ($h_\theta(x) \approx 0$), kayıp sonsuza gider.
-   Eğer gerçek $y=0$ ise, kayıp $- \log(1-h_\theta(x))$ olur. Model $y=0$'ı doğru tahmin ederse ($h_\theta(x) \approx 0$), kayıp 0'a yaklaşır. Yanlış tahmin ederse ($h_\theta(x) \approx 1$), kayıp sonsuza gider.

Tüm veri seti ($m$ örnek) için ortalama hata fonksiyonu $J(\theta)$:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m Cost(h_\theta(x^{(i)}), y^{(i)})$$ 
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

Bu hata fonksiyonu konvekstir ve gradient descent ile global minimumu bulmak daha kolaydır.

## 3. Gradyan ve Model Eğitimi

### 3.1. Gradyan Hesaplama
Hata fonksiyonu $J(\theta)$'nın $\theta_j$ parametresine göre kısmi türevi (gradyanı) şaşırtıcı bir şekilde doğrusal regresyondaki gradyan formülüyle aynı çıkar (ancak $h_\theta(x)$ tanımı farklıdır):

$$\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

Vektörel formda:
$$\nabla J(\theta) = \frac{1}{m} X^T (\sigma(X\theta) - y)$$

### 3.2. Gradient Descent ile Eğitim
Gradyan hesaplandıktan sonra, parametreler $\theta$ yine gradient descent algoritması kullanılarak güncellenir:

$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)$$ 
Veya vektörel olarak:
$$\theta := \theta - \alpha \nabla J(\theta)$$ 
$$\theta := \theta - \alpha \frac{1}{m} X^T (\sigma(X\theta) - y)$$

Burada $\alpha$ öğrenme hızıdır.

**Ağırlıkların (Theta) Yorumlanması:**
Doğrusal regresyonda $\theta_j$, $x_j$'deki bir birimlik artışın $y$'deki beklenen artışını gösterirken, lojistik regresyonda $\theta_j$'nin yorumu biraz daha farklıdır. $\theta_j$, $x_j$'deki bir birimlik artışın **log-odds (logit)** üzerindeki etkisini gösterir.
-   **Odds:** Bir olayın olma olasılığının olmama olasılığına oranı: $Odds = \frac{P(y=1)}{P(y=0)} = \frac{h_\theta(x)}{1-h_\theta(x)}$
-   **Log-Odds (Logit):** Odds'un doğal logaritması: $Logit = \log(\frac{h_\theta(x)}{1-h_\theta(x)}) = \theta^T x$

Dolayısıyla, $\theta_j$'nin pozitif olması, $x_j$ arttıkça $y=1$ olma olasılığının log-odds'unun arttığını (yani olasılığın arttığını), negatif olması ise azaldığını gösterir. $e^{\theta_j}$ değeri ise **odds oranı (odds ratio)** olarak yorumlanır; $x_j$'deki bir birimlik artışın odds'u kaç kat değiştirdiğini gösterir.

### 3.3. Mini-batch Gradient Descent
Büyük veri setlerinde, her adımda tüm veri setini kullanmak (Batch Gradient Descent) yerine, veriyi daha küçük alt kümelere (mini-batch) bölerek eğitimi hızlandırmak yaygındır.

**Algoritma:**
1.  Veri setini karıştır (shuffle).
2.  Veriyi belirli boyuttaki mini-batch'lere böl (örn. 32, 64, 128 örnek).
3.  Her bir mini-batch için:
    a.  İleri Yayılım (Forward Pass): Mini-batch'teki $X_{batch}$ için tahminleri hesapla: $h = \sigma(X_{batch}\theta)$.
    b.  Gradyan Hesaplama: Sadece o mini-batch'i kullanarak gradyanı hesapla: $\nabla J_{batch}(\theta) = \frac{1}{m_{batch}} X_{batch}^T (h - y_{batch})$.
    c.  Parametre Güncelleme: $\theta := \theta - \alpha \nabla J_{batch}(\theta)$.
4.  Tüm mini-batch'ler işlendiğinde bir **epoch** tamamlanmış olur. Bu işlem belirli sayıda epoch boyunca veya hata belirli bir seviyeye düşene kadar tekrarlanır.

## 4. MNIST ile Lojistik Regresyon Uygulaması (Python)

Şimdi, MNIST veri setinden 3 ve 7 rakamlarını ayırt etmek için lojistik regresyon modelini Python'da uygulayalım.

### 4.1. Kütüphaneler ve Veri Yükleme
Gerekli kütüphaneleri içe aktaralım ve MNIST verisini yükleyelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 3
import numpy as np
import idx2numpy # MNIST IDX formatını okumak için
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Veriyi bölmek için
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 7
MNIST_DIR = "mnist/" # Veri setinin bulunduğu dizin
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 9
# Eğitim verilerini ve etiketlerini yükle
train_arr = idx2numpy.convert_from_file(MNIST_DIR + "train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file(
    MNIST_DIR + "train-labels-idx1-ubyte")
```

### 4.2. Veri Keşfi ve Ön İşleme
Verinin boyutlarını ve piksel değer aralığını kontrol edelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 11
print(train_arr.shape, train_labels.shape)
# Çıktı: (60000, 28, 28) (60000,)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 13
print(np.min(train_arr), np.max(train_arr))
# Çıktı: 0 255
```

**Ön İşleme Adımları:**
1.  **Düzleştirme (Flattening):** 28x28 boyutundaki görüntüleri 784 elemanlı vektörlere dönüştürme.
2.  **Normalizasyon:** Piksel değerlerini 0-255 aralığından 0-1 aralığına ölçeklendirme.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 19
# 1. Düzleştirme
X_train = train_arr.reshape(60000, -1) # (60000, 784)
print(X_train.shape)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 23
# 2. Normalizasyon
X_train = X_train / 255.0
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 27
# Etiketleri kopyala
y_train = np.copy(train_labels)
```

### 4.3. Özel Veri Seti Oluşturma (3 ve 7 Rakamları)
Problemimizi ikili sınıflandırmaya indirgemek için sadece 3 ve 7 rakamlarını içeren örnekleri seçelim ve etiketleri yeniden düzenleyelim (3 -> 0, 7 -> 1).

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 31
# 3 rakamlarını filtrele
X_3 = X_train[y_train == 3]
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 36
# 7 rakamlarını filtrele
X_7 = X_train[y_train == 7]
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 40
# Yeni ikili etiketleri oluştur (3 için 0, 7 için 1)
y_3 = np.zeros(X_3.shape[0])
y_7 = np.ones(X_7.shape[0])
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 45
# Veri setlerini birleştir
X_train_subset = np.append(X_3, X_7, axis=0)
y_train_subset = np.append(y_3, y_7)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 49
print(X_train_subset.shape, y_train_subset.shape)
# Örnek Çıktı: (12396, 784) (12396,)
```

### 4.4. Eğitim ve Test Veri Setlerine Ayırma
Oluşturduğumuz 3 ve 7 rakamlarından oluşan veri setini eğitim ve test kümelerine ayıralım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 52
X_train, X_test, y_train, y_test = train_test_split(
    X_train_subset, y_train_subset, test_size=0.2, random_state=42)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 54
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Örnek Çıktı:
# (9916, 784) (9916,)
# (2480, 784) (2480,)
```

### 4.5. Lojistik Regresyon Sınıfı ve Eğitimi
Gradient descent kullanarak lojistik regresyon modelini eğiten bir sınıf oluşturalım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 58
class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr # Öğrenme hızı
        self.n_iters = n_iters # İterasyon sayısı
        self.weights = None # Model ağırlıkları (theta)
        # Bias terimi için ayrı bir parametre yerine,
        # X matrisine 1'lerden oluşan bir sütun ekleyip
        # weights vektörünün ilk elemanını bias olarak kullanacağız.

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Bias terimi için X'e 1'lerden oluşan sütun ekle
        ones_column = np.ones((num_samples, 1))
        X = np.append(ones_column, X, axis=1) # Boyut: (num_samples, num_features + 1)

        # Ağırlıkları (theta) sıfırla başlat
        self.weights = np.zeros(num_features + 1)
        history = [] # Hata değerlerini kaydetmek için

        # Gradient Descent
        for epoch in range(self.n_iters):
            # 1. Doğrusal model: z = X * theta
            # (num_samples, num_features + 1) * (num_features + 1,) -> (num_samples,)
            linear_model = np.dot(X, self.weights)

            # 2. Sigmoid ile olasılıkları hesapla: h = sigma(z)
            y_hat = sigmoid(linear_model)

            # 3. Hata (Cross-Entropy Loss) hesapla
            # Küçük değerlerde log(0) hatasını önlemek için epsilon eklenebilir
            epsilon = 1e-15
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            history.append(loss)

            # 4. Gradyanı hesapla: dw = (1/m) * X^T * (h - y)
            # (num_features + 1, num_samples) * (num_samples,) -> (num_features + 1,)
            dw = (1 / num_samples) * np.dot(X.T, (y_hat - y))

            # 5. Ağırlıkları güncelle: theta := theta - alpha * dw
            self.weights = self.weights - (self.lr * dw)

            # Belirli aralıklarla hatayı yazdır
            if (epoch + 1) % 50 == 0:
                print(f"e: {epoch + 1:04} \t loss: {loss}")

        return history # Eğitim sırasındaki hata geçmişini döndür

    def predict(self, X):
        num_samples, _ = X.shape

        # Tahmin için de X'e bias sütununu ekle
        ones_column = np.ones((num_samples, 1))
        X = np.append(ones_column, X, axis=1)

        # Doğrusal model ve sigmoid ile olasılıkları hesapla
        linear_model = np.dot(X, self.weights)
        y_predicted = sigmoid(linear_model)

        # Olasılıkları sınıflara dönüştür (eşik değeri 0.5)
        y_predicted_classes = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted_classes
```

Modeli eğitelim:

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 62
regressor = LogisticRegression(lr=0.001, n_iters=1000)
history = regressor.fit(X_train, y_train)

# Hata grafiğini çizdir
plt.plot(history)
plt.xlabel("İterasyon")
plt.ylabel("Hata (Cross-Entropy Loss)")
plt.title("Eğitim Sırasında Hata Değişimi")
plt.show()
```
*Hata grafiği burada gösterilir.*

### 4.6. Model Değerlendirme
Modelin performansını test seti üzerinde ölçelim. Bunun için **doğruluk (accuracy)** metriğini kullanabiliriz.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 60
def accuracy(y_true, y_predicted):
    # Doğru tahminlerin toplam örneğe oranı
    acc = np.sum(y_true == y_predicted) / len(y_predicted)
    return acc
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 64
# Test seti üzerinde tahmin yap
y_predictions = regressor.predict(X_test)

# Doğruluğu hesapla ve yazdır
test_accuracy = accuracy(y_test, y_predictions)
print(f"Test Seti Doğruluğu: {test_accuracy:.4f}")
# Örnek Çıktı: Test Seti Doğruluğu: 0.9758
```

Modelimiz, test setindeki 3 ve 7 rakamlarını yaklaşık %97.6 doğrulukla ayırt edebilmektedir.

### 4.7. Tahminlerin Görselleştirilmesi
Test setinden bazı örnekler ve modelin tahminlerini görselleştirelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/3_log_reg.ipynb
# cell 66
plt.figure(figsize=(9, 9))

for i in range(9):
    plt.subplot(3, 3, i+1)
    # Tahmini başlığa yaz (0 -> 3, 1 -> 7)
    plt.title(f"Tahmin: {y_predictions[i]}")
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.axis('off') # Eksenleri kapat

plt.tight_layout()
plt.show()
```
*Tahminleri içeren görseller burada gösterilir.*

## 5. Model Değerlendirme Metrikleri (Teori)

Doğruluk (Accuracy) yaygın bir metrik olsa da, özellikle dengesiz veri setlerinde (sınıflardan birinin diğerinden çok daha fazla örneğe sahip olduğu durumlar) yanıltıcı olabilir. Bu nedenle sınıflandırma problemlerinde başka metrikler de kullanılır:

**Karmaşıklık Matrisi (Confusion Matrix):**

|                   | Tahmin Edilen: 1 | Tahmin Edilen: 0 |
| :---------------- | :--------------- | :--------------- |
| **Gerçek: 1**     | TP (True Positive) | FN (False Negative)| 
| **Gerçek: 0**     | FP (False Positive)| TN (True Negative) |

-   **TP (Doğru Pozitif):** Gerçekte 1 olan ve 1 olarak tahmin edilenler.
-   **TN (Doğru Negatif):** Gerçekte 0 olan ve 0 olarak tahmin edilenler.
-   **FP (Yanlış Pozitif):** Gerçekte 0 olan ancak 1 olarak tahmin edilenler (Tip I Hata).
-   **FN (Yanlış Negatif):** Gerçekte 1 olan ancak 0 olarak tahmin edilenler (Tip II Hata).

**Diğer Metrikler:**

-   **Kesinlik (Precision):** Pozitif olarak tahmin edilen örneklerden ne kadarının gerçekten pozitif olduğu.
    $$Precision = \frac{TP}{TP + FP}$$
    *Yüksek kesinlik, modelin pozitif tahminlerinde daha güvenilir olduğunu gösterir (daha az yanlış pozitif).*

-   **Duyarlılık (Recall / Sensitivity / True Positive Rate - TPR):** Gerçekte pozitif olan örneklerden ne kadarının doğru bir şekilde pozitif olarak tahmin edildiği.
    $$Recall = \frac{TP}{TP + FN}$$
    *Yüksek duyarlılık, modelin gerçek pozitifleri kaçırmadığını gösterir (daha az yanlış negatif).*

-   **F1 Skoru:** Kesinlik ve Duyarlılık metriklerinin harmonik ortalaması. Bu iki metrik arasında bir denge sağlar.
    $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
    *Kesinlik ve Duyarlılık arasında bir ödünleşim (trade-off) vardır. F1 skoru, bu dengeyi tek bir sayıda özetler.*

-   **Özgüllük (Specificity / True Negative Rate - TNR):** Gerçekte negatif olan örneklerden ne kadarının doğru bir şekilde negatif olarak tahmin edildiği.
    $$Specificity = \frac{TN}{TN + FP}$$

-   **False Positive Rate (FPR):** Gerçekte negatif olan örneklerden ne kadarının yanlışlıkla pozitif olarak tahmin edildiği.
    $$FPR = \frac{FP}{TN + FP} = 1 - Specificity$$

**Kesinlik-Duyarlılık Ödünleşimi (Precision-Recall Trade-off):**
-   Genellikle, sınıflandırma eşik değerini (threshold) değiştirerek kesinliği artırmaya çalışırsanız, duyarlılık azalır ve tersi de geçerlidir.
    -   **Eşiği Yükseltmek:** Modelin $y=1$ tahmini yapmak için daha emin olması gerekir. Bu, FP sayısını azaltır (Kesinlik artar) ancak TP sayısını da azaltabilir (Duyarlılık azalır).
    -   **Eşiği Düşürmek:** Model daha kolay $y=1$ tahmini yapar. Bu, FN sayısını azaltır (Duyarlılık artar) ancak FP sayısını da artırabilir (Kesinlik azalır).
-   Hangi metriğin daha önemli olduğu probleme bağlıdır:
    -   **Spam Tespiti:** Yanlışlıkla önemli bir e-postayı spam olarak işaretlemek (FP) kötü olabilir. Kesinlik (Precision) daha önemli olabilir.
    -   **Hastalık Teşhisi:** Hasta birini sağlıklı olarak etiketlemek (FN) çok tehlikeli olabilir. Duyarlılık (Recall) daha önemli olabilir.
-   **Precision-Recall Eğrisi:** Farklı eşik değerleri için Kesinlik ve Duyarlılık değerlerini gösteren bir grafiktir. ROC eğrisine benzer şekilde model karşılaştırması için kullanılır, özellikle dengesiz veri setlerinde ROC'dan daha bilgilendirici olabilir.

**ROC Eğrisi (Receiver Operating Characteristic Curve):**
-   Farklı sınıflandırma eşik değerleri (0.5 yerine) için **True Positive Rate (Recall)**'in **False Positive Rate**'e karşı çizildiği bir grafiktir.
-   Eğrinin sol üst köşeye yakın olması daha iyi bir performansı gösterir.
-   Rastgele bir tahminci köşegen çizgisine (TPR = FPR) karşılık gelir.
-   **AUC (Area Under the Curve - Eğri Altında Kalan Alan):** ROC eğrisinin altında kalan alandır. 0.5 ile 1 arasında bir değer alır.
    -   AUC = 1: Mükemmel sınıflandırıcı.
    -   AUC = 0.5: Rastgele tahminci.
    -   AUC < 0.5: Rastgeleden daha kötü.
-   AUC, farklı modelleri veya aynı modelin farklı hiperparametrelerini karşılaştırmak için kullanışlıdır.

## 6. Düzenlileştirme (Regularization)

Doğrusal regresyonda olduğu gibi, lojistik regresyonda da **aşırı öğrenmeyi (overfitting)** önlemek için düzenlileştirme teknikleri kullanılabilir. Hata fonksiyonuna, ağırlıkların (theta değerlerinin) büyüklüğünü cezalandıran bir terim eklenir.

### 6.1. L1 Düzenlileştirme (Lasso)
Hata fonksiyonuna ağırlıkların mutlak değerlerinin toplamı eklenir:
$$J_{new}(\theta) = J(\theta) + \frac{\lambda}{m}\sum_{j=1}^n |\theta_j|$$
(Not: Genellikle bias terimi $\theta_0$ düzenlileştirmeye dahil edilmez.)
L1, bazı ağırlıkları tam olarak sıfır yapma eğilimindedir, bu da **özellik seçimi (feature selection)** yapmaya yardımcı olabilir.

### 6.2. L2 Düzenlileştirme (Ridge)
Hata fonksiyonuna ağırlıkların karelerinin toplamı eklenir:
$$J_{new}(\theta) = J(\theta) + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$
L2, ağırlıkları sıfıra yaklaştırır ancak genellikle tam sıfır yapmaz. Genellikle L1'den daha yaygın kullanılır.

$\lambda$ (lambda), düzenlileştirme parametresidir ve ne kadar ceza uygulanacağını kontrol eder. $\lambda$ değeri genellikle çapraz doğrulama (cross-validation) ile belirlenir.

## 7. Çok Sınıflı Lojistik Regresyon

İkiden fazla sınıf olduğunda lojistik regresyonu kullanmak için yaygın yaklaşımlar:

### 7.1. One-vs-All (One-vs-Rest) Yaklaşımı
-   $K$ adet sınıf varsa, $K$ tane bağımsız ikili lojistik regresyon modeli eğitilir.
-   $i$-inci model, sınıf $i$'yi diğer tüm sınıflara ($K-1$ sınıf) karşı ayırt etmek için eğitilir (Sınıf $i$ -> 1, diğerleri -> 0).
-   Yeni bir $x$ girdisi geldiğinde, $K$ modelin hepsi çalıştırılır ve en yüksek olasılığı veren sınıf tahmin olarak seçilir:
    $$\text{Tahmin} = \arg\max_i(h_\theta^{(i)}(x))$$
-   Uygulaması basittir ancak modeller bağımsız eğitildiği için olasılıklar tam olarak kalibre olmayabilir.

### 7.2. Softmax Regresyon (Multinomial Logistic Regression)
-   One-vs-All'dan farklı olarak, tüm sınıfları aynı anda ele alan tek bir model eğitilir.
-   Hipotez fonksiyonu, her sınıf $j$ için $P(y=j|x;\theta)$ olasılığını hesaplayan **softmax** fonksiyonunu kullanır:
    $$P(y=j|x;\theta) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} \quad \text{burada } z_j = \theta_j^T x$$
-   Softmax fonksiyonu, her sınıf için hesaplanan skorları ($z_j$) alır ve toplamları 1 olan bir olasılık dağılımına dönüştürür.
-   Hata fonksiyonu genellikle kategorik çapraz entropi (categorical cross-entropy) olur.
-   Genellikle sinir ağlarının çıkış katmanlarında kullanılır.

## 8. Uygulama İpuçları

-   **Özellik Mühendisliği (Feature Engineering):**
    -   **Özellik Ölçeklendirme (Feature Scaling):** Gradient descent'in daha hızlı yakınsaması için özelliklerin benzer ölçeklerde olması önemlidir (örn. Normalizasyon veya Standardizasyon).
    -   **Polinom Özellikler:** Doğrusal olarak ayrılamayan veriler için özelliklerin polinom kombinasyonları (örn. $x_1^2, x_1x_2$) eklenebilir.
-   **Hiperparametre Optimizasyonu:**
    -   **Öğrenme Hızı (Learning Rate $\alpha$):** Uygun bir değer bulmak için denemeler yapılmalıdır.
    -   **Düzenlileştirme Parametresi ($\lambda$):** Çapraz doğrulama ile en iyi değeri bulmak önemlidir.
    -   **Mini-batch Boyutu:** Performans ve eğitim süresi üzerinde etkisi vardır, denenebilir.
-   **Model Seçimi:**
    -   Verinin **doğrusal olarak ayrılabilir (linearly separable)** olup olmadığını kontrol etmek (lojistik regresyon özellik uzayında doğrusal bir karar sınırı - hyperplane - çizer).
    -   Doğrusal olmayan sınırlar gerekiyorsa, polinom özellikler, kernel yöntemleri (SVM gibi) veya daha karmaşık modeller (sinir ağları) düşünülebilir.
-   **Eşik Değeri Seçimi (Threshold Tuning):** Varsayılan 0.5 eşik değeri her zaman en iyisi olmayabilir. Özellikle Kesinlik ve Duyarlılık arasında belirli bir denge hedefleniyorsa veya maliyetler (FP vs FN maliyeti) farklıysa, doğrulama seti üzerinde farklı eşik değerleri denenerek (örn. Precision-Recall eğrisine bakarak) optimum eşik bulunabilir.

## 9. Sıkça Sorulan Sorular (SSS)

**S1: Lojistik Regresyon ile Doğrusal Regresyon arasındaki temel farklar nelerdir? Ne zaman hangisini kullanmalıyım?**
**C1:**
*   **Amaç:** Doğrusal regresyon **sürekli** bir değeri (örn. fiyat) tahmin eder (regresyon problemi). Lojistik regresyon ise bir örneğin belirli bir **sınıfa ait olma olasılığını** tahmin eder ve genellikle sınıflandırma için kullanılır (sınıflandırma problemi).
*   **Çıktı:** Doğrusal regresyonun çıktısı herhangi bir reel sayı olabilir. Lojistik regresyonun çıktısı (sigmoid fonksiyonu sayesinde) 0 ile 1 arasında bir olasılık değeridir.
*   **Hipotez:** Doğrusal regresyon $h_\theta(x) = \theta^T x$ kullanır. Lojistik regresyon $h_\theta(x) = \sigma(\theta^T x)$ kullanır.
*   **Hata Fonksiyonu:** Doğrusal regresyon MSE kullanır. Lojistik regresyon Çapraz Entropi (Log Loss) kullanır.
*   **Kullanım:** Tahmin edilecek hedef değişken sürekli ise doğrusal regresyon, kategorik (özellikle ikili) ise lojistik regresyon uygundur.

**S2: "Doğrusal olarak ayrılabilir" (linearly separable) ne anlama gelir?**
**C2:** İki sınıfın verileri, özellik uzayında tek bir doğru (2D), düzlem (3D) veya hiperdüzlem (>3D) ile tamamen ayrılabiliyorsa, bu veriler doğrusal olarak ayrılabilir demektir. Lojistik regresyon, temel formunda, verilerin doğrusal olarak ayrılabilir olduğunu varsayar veya böyle bir doğrusal sınır bulmaya çalışır. Eğer veriler doğrusal olarak ayrılamıyorsa (örn. iç içe geçmiş daireler gibi), basit lojistik regresyon iyi performans göstermeyebilir ve özellik mühendisliği (örn. polinom özellikler) veya daha karmaşık modeller gerekebilir.

**S3: Dengesiz veri setleriyle (imbalanced datasets) nasıl başa çıkılır?**
**C3:** Bir sınıfın diğerinden çok daha fazla örneğe sahip olduğu durumlarda (örn. %99 sağlıklı, %1 hasta), doğruluk (accuracy) yanıltıcı olabilir. Baş etme yöntemleri:
*   **Daha Uygun Metrikler Kullanma:** Kesinlik (Precision), Duyarlılık (Recall), F1 Skoru, AUC, Precision-Recall Eğrisi gibi metriklere odaklanmak.
*   **Yeniden Örnekleme (Resampling):**
    *   **Azınlık Sınıfını Fazla Örnekleme (Oversampling):** Azınlık sınıfından rastgele örnekleri kopyalayarak sayısını artırmak.
    *   **Çoğunluk Sınıfını Az Örnekleme (Undersampling):** Çoğunluk sınıfından rastgele örnekleri silerek sayısını azaltmak.
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Azınlık sınıfı için sentetik (yapay) yeni örnekler üretmek.
*   **Maliyet Duyarlı Öğrenme (Cost-Sensitive Learning):** Algoritmanın hata fonksiyonunda, azınlık sınıfını yanlış sınıflandırmanın maliyetini artırmak.
*   **Farklı Algoritmalar Deneme:** Bazı algoritmalar (örn. Ağaç tabanlı modeller) dengesizliğe daha dayanıklı olabilir.

**S4: Lojistik Regresyon çok sınıflı problemler için kullanılabilir mi?**
**C4:** Evet, kullanılabilir. İki ana yaklaşım vardır:
*   **One-vs-All (One-vs-Rest):** Her sınıf için ayrı bir ikili lojistik regresyon modeli eğitilir (o sınıf vs. diğer tüm sınıflar). Tahmin, en yüksek olasılığı veren modelin sınıfı olur.
*   **Softmax Regresyon (Multinomial Logistic Regression):** Tüm sınıfları aynı anda ele alan tek bir model eğitilir. Çıkış katmanında sigmoid yerine softmax fonksiyonu kullanılır.

**S5: Lojistik Regresyonun avantajları ve dezavantajları nelerdir?**
**C5:**
*   **Avantajları:**
    *   Basit, hızlı ve yorumlanması kolaydır (katsayılar özelliklerin önemini ve yönünü gösterir).
    *   Olasılık tahmini verir.
    *   Doğrusal olarak ayrılabilir verilerde iyi performans gösterir.
    *   Hesaplama açısından verimlidir.
    *   Düzenlileştirme ile aşırı öğrenmeye karşı dirençli hale getirilebilir.
*   **Dezavantajları:**
    *   Doğrusal karar sınırı varsayımı nedeniyle karmaşık, doğrusal olmayan ilişkileri yakalayamaz (özellik mühendisliği gerekebilir).
    *   Özellikler arasındaki karmaşık etkileşimleri otomatik olarak modellemez.
    *   Aykırı değerlere (outliers) karşı hassas olabilir.
    *   Yüksek boyutlu veri setlerinde (çok fazla özellik) aşırı öğrenmeye eğilimli olabilir (düzenlileştirme önemlidir).

Bu belge, lojistik regresyonun teorik temellerini, Python ile MNIST veri seti üzerinde ikili sınıflandırma uygulamasını, önemli değerlendirme metriklerini, düzenlileştirmeyi, çok sınıflı yaklaşımları, uygulama ipuçlarını ve sıkça sorulan soruları kapsamaktadır.
