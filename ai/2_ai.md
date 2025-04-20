# Doğrusal Regresyon (Linear Regression) - Teorik ve Uygulamalı Anlatım

Bu belge, doğrusal regresyon algoritmasının hem teorik temellerini hem de Python ile pratik uygulamasını bir araya getirmektedir. `/home/apo/Code/tmp/cvp_lect/notes_md/2_lin_reg.md` dosyasındaki teorik bilgiler ve `/home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb` notebook'undaki kod örnekleri kullanılmıştır.

## 1. Temel Kavramlar (Teori)

### 1.1. Veri Seti ve Hipotez
Makine öğrenmesinde amaç, genellikle bir girdi verisine (özellikler) karşılık gelen bir çıktıyı (hedef değişken) tahmin etmektir. Doğrusal regresyon özelinde, bu ilişkiyi doğrusal bir fonksiyonla modellemeye çalışırız.

Örnek: Ev fiyatlarını tahmin etme
- **Girdi (Özellik - Feature):** Evin alanı (metrekare) - $x$
- **Çıktı (Hedef - Target):** Tahmini Fiyat - $y$

### 1.2. Matematiksel Model (Hipotez Fonksiyonu)
Girdi ve çıktı arasındaki doğrusal ilişkiyi temsil eden matematiksel fonksiyona **hipotez fonksiyonu** denir. Tek değişkenli (tek özellikli) doğrusal regresyon için hipotez fonksiyonu şöyledir:

- **Basit form:** $Fiyat = a \times Alan + b$
- **Matematiksel form:** $y = ax + b$
- **Makine öğrenmesi notasyonu:** $h_\theta(x) = \theta_0 + \theta_1x$

Burada:
- $h_\theta(x)$: Verilen $x$ girdisi için modelin tahmini.
- $\theta_0$: Kesişim (intercept) veya bias terimi. Doğrunun y eksenini kestiği noktadır.
- $\theta_1$: Eğim (slope) veya ağırlık (weight). Girdi $x$'in çıktı $y$ üzerindeki etkisini belirler.
- $\theta_0$ ve $\theta_1$: Modelin **parametreleri**dir. Modelin öğrenmesi gereken değerlerdir.

### 1.3. Hata Fonksiyonu (Cost Function)
Modelimizin ne kadar iyi performans gösterdiğini ölçmek için bir **hata fonksiyonu** (veya kayıp fonksiyonu - loss function) kullanırız. Doğrusal regresyonda yaygın olarak **Ortalama Kare Hata (Mean Squared Error - MSE)** fonksiyonunun bir varyasyonu kullanılır. Amaç, modelin tahminleri ($h_\theta(x^{(i)})$) ile gerçek değerler ($y^{(i)}$) arasındaki farkların karelerinin ortalamasını minimize etmektir.

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

Burada:
- $m$: Veri setindeki örnek sayısı.
- $x^{(i)}$: $i$-inci örneğin girdi özelliği.
- $y^{(i)}$: $i$-inci örneğin gerçek çıktı değeri.
- $h_\theta(x^{(i)}) = \theta_0 + \theta_1x^{(i)}$: $i$-inci örnek için modelin tahmini.
- $\frac{1}{2m}$: Ortalama almak ve türev hesaplamalarını basitleştirmek için kullanılan bir skalerdir. (1/2 faktörü türev alındığında kaybolur). Kare alma işlemi, hem pozitif hem de negatif hataların aynı şekilde cezalandırılmasını sağlar ve büyük hataları küçük hatalardan daha fazla vurgular. Ayrıca, MSE fonksiyonu türevlenebilir olduğu için optimizasyon algoritmaları (Gradient Descent gibi) ile uyumludur.

**Hedefimiz:** $J(\theta_0, \theta_1)$ hata fonksiyonunu minimize eden $\theta_0$ ve $\theta_1$ değerlerini bulmaktır. Bu, modelimizin verilere en uygun doğruyu bulması anlamına gelir.

### 1.4. Dereceli Alçalma (Gradient Descent)
Hata fonksiyonunu minimize etmek için kullanılan popüler bir optimizasyon algoritması **Dereceli Alçalma (Gradient Descent)**'dir. Bu algoritma, hata fonksiyonunun parametrelere göre kısmi türevlerini (gradyanını) hesaplayarak parametreleri iteratif olarak günceller ve minimum noktaya doğru adım adım ilerler.

**Sezgisel Anlatım:** Hata fonksiyonunu bir vadinin yüzeyi gibi düşünün. Amacımız vadinin en dip noktasına ulaşmaktır. Gradient Descent, bulunduğumuz noktadaki eğimi (gradyanı) hesaplar ve eğimin en dik olduğu ters yönde küçük bir adım atar. Bu adımları tekrarlayarak yavaş yavaş vadinin dibine (minimum hata noktasına) ineriz.

**Algoritma Adımları:**
1. $\theta_0$ ve $\theta_1$'e başlangıç değerleri ata (genellikle 0).
2. Hata fonksiyonunun minimumuna ulaşana kadar (veya belirli sayıda iterasyon boyunca) tekrarla:
   a. Hata fonksiyonunun $\theta_0$ ve $\theta_1$'e göre kısmi türevlerini hesapla:
      $$\frac{\partial}{\partial\theta_0}J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$$
      $$\frac{\partial}{\partial\theta_1}J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
   b. Parametreleri güncelle:
      $$\theta_0 := \theta_0 - \alpha \frac{\partial}{\partial\theta_0}J(\theta_0, \theta_1)$$
      $$\theta_1 := \theta_1 - \alpha \frac{\partial}{\partial\theta_1}J(\theta_0, \theta_1)$$

Burada $\alpha$, **öğrenme hızı (learning rate)** olarak adlandırılan bir hiperparametredir. Her adımda ne kadar büyük bir güncelleme yapılacağını kontrol eder. Gradyan, hatanın parametrelerdeki küçük bir değişikliğe ne kadar duyarlı olduğunu gösterir.

**Gradient Descent Türleri:**
*   **Batch Gradient Descent:** Her güncelleme adımında tüm eğitim verisi kullanılır (yukarıdaki formüller bunu temsil eder). Kararlı bir yakınsama sağlar ancak büyük veri setlerinde yavaş olabilir.
*   **Stochastic Gradient Descent (SGD):** Her güncelleme adımında rastgele seçilmiş *tek bir* veri örneği kullanılır. Çok daha hızlıdır ancak gürültülü güncellemeler nedeniyle salınım yapabilir.
*   **Mini-Batch Gradient Descent:** Her güncelleme adımında küçük bir grup (batch) veri örneği kullanılır. Batch ve SGD arasında bir denge sunar, pratikte en yaygın kullanılanıdır.

### 1.5. Öğrenme Hızı (Learning Rate - $\alpha$)
Öğrenme hızının doğru seçilmesi kritiktir:
- **Çok büyük $\alpha$:** Algoritma minimum noktayı aşabilir ve yakınsamayabilir (hata artabilir).
- **Çok küçük $\alpha$:** Algoritma çok yavaş yakınsar, minimuma ulaşmak çok uzun sürebilir.

Pratikte, uygun bir öğrenme hızı bulmak için farklı değerler denenebilir.

## 2. Tek Değişkenli Doğrusal Regresyon Uygulaması (Python)

Şimdi, teorik bilgileri kullanarak Python'da tek değişkenli bir doğrusal regresyon modeli oluşturalım ve eğitelim.

### 2.1. Kütüphanelerin İçe Aktarılması
Gerekli kütüphaneleri (numpy ve matplotlib) içe aktaralım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 2
import numpy as np
import matplotlib.pyplot as plt
```

### 2.2. Veri Seti Oluşturma ve Görselleştirme
Modelimizi eğitmek için sentetik bir veri seti oluşturalım. Gerçek dünya verilerini taklit etmek için doğrusal ilişkiye sahip verilere rastgele gürültü ekleyeceğiz.

Kullanacağımız model: $y = \theta_0 + \theta_1 x + \epsilon$
- Gerçek $\theta_0 = -3$ (kesişim)
- Gerçek $\theta_1 = 0.5$ (eğim)
- $\epsilon \sim \mathcal{N}(0, 0.05)$: Ortalama 0, standart sapma 0.05 olan normal dağılımlı gürültü.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 4
ornek_sayisi = 100
theta_0 = -3
theta_1 = 0.5
X = np.linspace(-2, 2, ornek_sayisi)
y = theta_0 + (theta_1 * X) + np.random.normal(0, 0.05, size=ornek_sayisi)
print(X.shape, y.shape)
```

Oluşturulan veriyi görselleştirelim:

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 6
plt.figure(figsize=(6, 6))
plt.scatter(X, y)
plt.xlabel("X", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.show()
```
*Grafik çıktısı burada gösterilir (notebook'taki gibi).*

### 2.3. Eğitim ve Doğrulama Veri Setlerinin Ayrılması
Modelin performansını objektif bir şekilde değerlendirmek için veri setini eğitim (%80) ve doğrulama (%20) kümelerine ayırırız. Model eğitim verisiyle öğrenir, doğrulama verisiyle genelleme yeteneği test edilir.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 8
# verileri %80 - %20 olacak şekilde eğitim ve doğrulama verisi olarak ayır
egitim_ornek_sayisi = int(ornek_sayisi * 0.8)

X_egitim, X_val = X[:egitim_ornek_sayisi], X[egitim_ornek_sayisi:]
y_egitim, y_val = y[:egitim_ornek_sayisi], y[egitim_ornek_sayisi:]

print(f"X egitim: {X_egitim.shape}, X validasyon: {X_val.shape}")
print(f"y egitim: {y_egitim.shape}, y validasyon: {y_val.shape}")
```

### 2.4. Tek Özellik İçin Doğrusal Regresyon Sınıfı
Gradient descent kullanarak $\theta_0$ ve $\theta_1$'i optimize eden bir Python sınıfı oluşturalım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 12

# Hata fonksiyonu (MSE'nin 1/2'si)
def hata_fonksiyonu(y_gercek, y_tahmin):
    ornek_sayisi = len(y_gercek)
    # J(theta) = 1 / (2m) * sum((h_theta(x_i) - y_i)^2)
    hata = 1 / (2 * ornek_sayisi) * np.sum((y_tahmin - y_gercek) ** 2)
    return hata


class LinearRegression_1_feature:
    def __init__(self, ogrenme_hizi):
        self.ogrenme_hizi = ogrenme_hizi
        self.theta_0 = None # Bias / Intercept
        self.theta_1 = None # Weight / Slope

    def fit(self, X, y, tekrar_sayisi):
        ornek_sayisi = len(X)

        # parametrelere ilk değerlerini ata (0)
        self.theta_0 = 0
        self.theta_1 = 0

        # Eğitim sürecini görselleştirmek için grafiği hazırla
        plt.figure(figsize=(8, 8))
        plt.scatter(X, y)
        plt.xlabel("X", fontsize=18)
        plt.ylabel("y", fontsize=18)

        hatalar = [] # Her 50 iterasyonda hatayı kaydetmek için

        # Gradient Descent
        for i in range(tekrar_sayisi):
            # 1. Tahmin yap: h_theta(x) = theta_0 + theta_1 * x
            y_tahmin = self.theta_0 + (self.theta_1 * X)

            # 2. Gradyanları (türevleri) hesapla
            # dJ/d(theta_0) = (1/m) * sum(h_theta(x_i) - y_i)
            d_theta_0 = (1 / ornek_sayisi) * np.sum(y_tahmin - y)
            # dJ/d(theta_1) = (1/m) * sum((h_theta(x_i) - y_i) * x_i)
            d_theta_1 = (1 / ornek_sayisi) * np.dot(X.T, (y_tahmin - y)) # X.T ile çarpım, toplama işlemini yapar

            # 3. Parametreleri güncelle
            # theta_j := theta_j - alpha * dJ/d(theta_j)
            self.theta_0 = self.theta_0 - self.ogrenme_hizi * d_theta_0
            self.theta_1 = self.theta_1 - self.ogrenme_hizi * d_theta_1

            # Her 50 döngüde bir tahmin grafiğini ve hatayı kaydet
            if i % 50 == 0:
                plt.plot(X, y_tahmin, label=f'Iter {i}') # Tahmin doğrusunu çiz
                hata = hata_fonksiyonu(y, y_tahmin)
                hatalar.append(hata)
                print(f"Iterasyon {i}: Hata = {hata:.4f}, theta_0 = {self.theta_0:.4f}, theta_1 = {self.theta_1:.4f}")


        plt.title("Eğitim Süreci Boyunca Model Tahminleri")
        plt.legend()
        plt.show()
        return hatalar

    def predict(self, X):
        # Eğitilmiş parametrelerle tahmin yap
        y_tahmin = self.theta_0 + (self.theta_1 * X)
        return y_tahmin

    def get_params(self):
        # Öğrenilen parametreleri döndür
        return self.theta_0, self.theta_1
```

### 2.5. Model Oluşturma ve Eğitme
Sınıfı kullanarak bir model nesnesi oluşturalım ve `fit` metodu ile eğitelim. Öğrenme hızını 0.01 ve iterasyon sayısını 1000 olarak ayarlayalım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 14
model = LinearRegression_1_feature(0.01)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 16
hatalar = model.fit(X_egitim, y_egitim, 1000)
```
*Eğitim sırasındaki tahmin doğrularını gösteren grafik çıktısı burada gösterilir.*

### 2.6. Hata Eğrisi
Eğitim sırasında kaydedilen hata değerlerini (her 50 iterasyonda bir) çizdirerek modelin zamanla nasıl iyileştiğini görelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 18
plt.figure(figsize=(6, 6))
plt.plot(range(0, 1000, 50), hatalar) # Hatalar her 50 iterasyonda kaydedildi
plt.xlabel("İterasyon (x50)", fontsize=18)
plt.ylabel(r"Hata $J(\theta)$", fontsize=18)
plt.title("Eğitim Sırasında Hata Değişimi")
plt.show()
```
*Hata eğrisi grafiği burada gösterilir.* Grafiğin zamanla azaldığını ve belirli bir seviyede stabilize olduğunu görmeyi bekleriz.

### 2.7. Eğitilmiş Modelin Görselleştirilmesi
Eğitilmiş modelin eğitim verisi üzerindeki son tahminini (kırmızı çizgi) görselleştirelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 20
plt.figure(figsize=(6, 6))
plt.scatter(X_egitim, y_egitim, label='Eğitim Verisi')

plt.xlabel("X", fontsize=18)
plt.ylabel("y", fontsize=18)

y_tahmin_egitim = model.predict(X_egitim)
plt.plot(X_egitim, y_tahmin_egitim, color="red", linewidth=3, label='Model Tahmini')
plt.title("Eğitilmiş Model ve Eğitim Verisi")
plt.legend()
plt.show()
```
*Eğitim verisi ve model tahminini gösteren grafik burada gösterilir.*

### 2.8. Model Doğrulama (Validasyon)
Modelin daha önce görmediği doğrulama verisi üzerindeki performansını test edelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 22
plt.figure(figsize=(6, 6))
plt.scatter(X_val, y_val, label='Doğrulama Verisi')

plt.xlabel("X", fontsize=18)
plt.ylabel("y", fontsize=18)

y_tahmin_val = model.predict(X_val)
plt.plot(X_val, y_tahmin_val, color="red", linewidth=3, label='Model Tahmini')
plt.title("Eğitilmiş Model ve Doğrulama Verisi")
plt.legend()
plt.show()
```
*Doğrulama verisi ve model tahminini gösteren grafik burada gösterilir.*

### 2.9. Model Performansının Değerlendirilmesi
Modelin doğrulama seti üzerindeki performansını Ortalama Kare Hata (MSE) ile sayısal olarak ölçelim. MSE, tahmin hatalarının karelerinin ortalamasıdır. Değer ne kadar küçükse, model o kadar iyidir.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 24
def mean_squared_error(y_gercek, y_tahmin):
    # MSE = (1/m) * sum((y_gercek - y_tahmin)^2)
    return np.mean((y_gercek - y_tahmin) ** 2)
```

Bir diğer yaygın metrik **R-kare (R-squared veya Coefficient of Determination)**'dir. Bağımsız değişken(ler) tarafından açıklanan bağımlı değişkendeki varyans oranını gösterir. 0 ile 1 arasında bir değer alır. 1'e yakın olması modelin veriye iyi uyum sağladığını gösterir.

```python
# R-kare hesaplama fonksiyonu (örnek)
def r_squared(y_gercek, y_tahmin):
    ss_res = np.sum((y_gercek - y_tahmin) ** 2) # Residual sum of squares
    ss_tot = np.sum((y_gercek - np.mean(y_gercek)) ** 2) # Total sum of squares
    return 1 - (ss_res / ss_tot)
```

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 25
mse_val = mean_squared_error(y_val, y_tahmin_val)
r2_val = r_squared(y_val, y_tahmin_val) # R-kareyi de hesaplayalım
print(f"Doğrulama Seti MSE: {mse_val:.4f}")
print(f"Doğrulama Seti R-kare: {r2_val:.4f}")
```

Modelin öğrendiği parametreleri yazdıralım ve veri setini oluştururken kullandığımız gerçek değerlerle ($\theta_0 = -3$, $\theta_1 = 0.5$) karşılaştıralım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 27
ogrenilen_theta_0, ogrenilen_theta_1 = model.get_params()
print(f"Modelin öğrendiği parametreler: theta_0 = {ogrenilen_theta_0:.4f}, theta_1 = {ogrenilen_theta_1:.4f}")
print(f"Gerçek parametreler:          theta_0 = {theta_0}, theta_1 = {theta_1}")
```
Öğrenilen parametrelerin gerçek değerlere oldukça yakın olmasını bekleriz.

## 3. Çok Değişkenli Doğrusal Regresyon

Şimdi modeli birden fazla girdi özelliğine sahip durumlar için genelleştirelim.

### 3.1. Matematiksel Model (Çok Değişkenli)
$n$ adet özelliğimiz olduğunda hipotez fonksiyonu şu şekilde olur:
$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$

Vektörel gösterimle daha kompakt yazılabilir:
- $x = [x_0, x_1, x_2, ..., x_n]^T$ (girdi vektörü, $x_0=1$ bias terimi için eklenir)
- $\theta = [\theta_0, \theta_1, \theta_2, ..., \theta_n]^T$ (parametre vektörü)
- $h_\theta(x) = \theta^T x$ (vektörlerin iç çarpımı)

Hata fonksiyonu aynı kalır, ancak $\theta$ artık bir vektördür:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^m (\theta^T x^{(i)} - y^{(i)})^2$$

Gradient Descent güncelleme kuralı her $\theta_j$ için genelleştirilir ($j=0, ..., n$):
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta)$$
$$\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

Vektörel olarak gradyan:
$$\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)$$
Güncelleme kuralı (vektörel):
$$\theta := \theta - \alpha \nabla J(\theta) = \theta - \alpha \frac{1}{m} X^T (X\theta - y)$$
Burada $X$, her satırı bir örneğin özellik vektörü ($x^{(i)T}$, $x_0=1$ dahil) olan $m \times (n+1)$ boyutlu **tasarım matrisidir**. $y$, gerçek değerleri içeren $m \times 1$ boyutlu vektördür. Vektörel işlemler, özellikle Numpy gibi kütüphanelerle uygulandığında, döngülere göre çok daha verimli çalışır.

### 3.2. Veri Seti Hazırlama (Çok Değişkenli)
İki özellik ($x_1, x_2$) ve bir bias terimi ($x_0=1$) içeren sentetik bir veri seti oluşturalım.
Gerçek parametreler: $\theta = [\theta_0, \theta_1, \theta_2] = [3, -2, 5]$

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 29
ornek_sayisi_mv = 100
# Gerçek theta değerleri [theta_0, theta_1, theta_2]
theta_arr = np.array([3, -2, 5])
n_features = len(theta_arr) # Bias dahil özellik sayısı (3)

# Tasarım matrisi X'i oluştur (m x n+1) -> (100 x 3)
X_mv = np.ones((ornek_sayisi_mv, n_features)) # İlk sütun bias için 1'ler
# İkinci sütun (x1)
X_mv[:,1] = np.linspace(-2, 1, ornek_sayisi_mv)
# Üçüncü sütun (x2)
X_mv[:,2] = np.linspace(1, 2, ornek_sayisi_mv)

# y değerlerini hesapla: y = X * theta^T
# (100x3) * (3x1) -> (100x1)
y_mv = np.dot(X_mv, theta_arr.T) # Gürültü eklemedik bu sefer

print("İlk 5 X_mv değeri (bias dahil):\n", X_mv[:5])
print("\nİlk 5 y_mv değeri:\n", y_mv[:5])
```

Veri boyutlarını kontrol edelim:

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 31
print(f"X_mv boyutu: {X_mv.shape}, y_mv boyutu: {y_mv.shape}")
```

### 3.3. Çok Değişkenli Doğrusal Regresyon Sınıfı
Herhangi bir sayıda özellikle çalışabilen genel bir `LinearRegression` sınıfı tanımlayalım. Bu sınıf vektörel işlemleri kullanacaktır.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 33
class LinearRegression:
    def __init__(self, ogrenme_hizi):
        self.ogrenme_hizi = ogrenme_hizi
        self.thetalar = None # Parametre vektörü theta

    def fit(self, X, y, tekrar_sayisi):
        ornek_sayisi, feature_sayisi = X.shape # m, n+1

        # parametrelere ilk değerlerini ata (sıfır vektörü)
        self.thetalar = np.zeros(feature_sayisi) # (n+1,) boyutlu vektör
        hatalar = []

        # Gradient Descent
        for i in range(tekrar_sayisi):
            # 1. Tahmin yap: y_tahmin = X * theta^T
            # (m x n+1) * (n+1 x 1) -> (m x 1)
            y_tahmin = np.dot(X, self.thetalar) # .T'ye gerek yok thetalar (n+1,) şeklinde

            # 2. Gradyanı hesapla: grad = (1/m) * X^T * (y_tahmin - y)
            # (n+1 x m) * (m x 1) -> (n+1 x 1)
            # y_tahmin ve y (m,) boyutunda, (m,1) yapmak için reshape veya newaxis gerekebilir
            # Ancak numpy broadcastingle genellikle çalışır.
            # Eğer y_tahmin (m,) ve y (m,) ise, (y_tahmin - y) de (m,) olur.
            # X.T (n+1, m) * (y_tahmin - y) (m,) -> (n+1,) boyutlu gradyan vektörü
            d_thetalar = (1 / ornek_sayisi) * np.dot(X.T, (y_tahmin - y))

            # 3. Parametreleri güncelle: theta := theta - alpha * grad
            self.thetalar = self.thetalar - self.ogrenme_hizi * d_thetalar

            # Her 20 döngüde bir hatayı kaydet
            if i % 20 == 0:
                hata = hata_fonksiyonu(y, y_tahmin) # Aynı hata fonksiyonunu kullanabiliriz
                hatalar.append(hata)
                # print(f"Iterasyon {i}: Hata = {hata:.4f}") # İsteğe bağlı

        return hatalar

    def predict(self, X):
        # Eğitilmiş parametrelerle tahmin yap: y_tahmin = X * theta^T
        y_tahmin = np.dot(X, self.thetalar)
        return y_tahmin

    def get_params(self):
        # Öğrenilen parametre vektörünü döndür
        return self.thetalar
```

### 3.4. Çok Değişkenli Model Eğitimi
Modeli oluşturalım ve çok değişkenli sentetik veri setimizle eğitelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 35
lr_model_mv = LinearRegression(0.01) # Öğrenme hızı 0.01
hatalar_mv = lr_model_mv.fit(X_mv, y_mv, 500) # 500 iterasyon
```

### 3.5. Hata Eğrisi (Çok Değişkenli Model)
Çok değişkenli modelin eğitim sırasındaki hata değişimini görselleştirelim.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 37
plt.figure(figsize=(6, 6))
plt.plot(range(0, 500, 20), hatalar_mv) # Hatalar her 20 iterasyonda kaydedildi
plt.xlabel("İterasyon (x20)", fontsize=18)
plt.ylabel(r"Hata $J(\theta)$", fontsize=18)
plt.title("Çok Değişkenli Model Eğitim Sırasında Hata Değişimi")
plt.show()
```
*Çok değişkenli modelin hata eğrisi grafiği burada gösterilir.*

### 3.6. Öğrenilen Parametreler (Çok Değişkenli Model)
Modelin öğrendiği parametre vektörünü yazdıralım ve gerçek değerlerle ([3, -2, 5]) karşılaştıralım.

```python
# filepath: /home/apo/Code/tmp/cvp_lect/Uygulama/2_lin_reg.ipynb
# cell 39
ogrenilen_thetalar = lr_model_mv.get_params()
print(f"Öğrenilen Theta değerleri: {ogrenilen_thetalar}")
print(f"Gerçek Theta değerleri:    {theta_arr}")
```
Öğrenilen değerlerin gerçek değerlere çok yakın olması beklenir.

## 4. Model Eğitimi Stratejileri (Teori)

Modelin performansını güvenilir bir şekilde değerlendirmek ve aşırı öğrenme (overfitting) gibi sorunları tespit etmek için farklı veri bölme stratejileri kullanılır:

### 4.1. Tüm Veri ile Eğitim
- **Yöntem:** Tüm veri seti hem eğitim hem de test için kullanılır.
- **Dezavantaj:** Modelin yeni, görmediği veriler üzerindeki performansını ölçemeyiz. Aşırı öğrenmeyi tespit etmek zordur. Genellikle pratik uygulamalarda tercih edilmez.

### 4.2. Train-Validation Split
- **Yöntem:** Veri seti ikiye ayrılır:
    - **Eğitim (Train) Seti:** Modelin parametrelerini öğrenmek için kullanılır (%70-80 gibi).
    - **Doğrulama (Validation) Seti:** Modelin hiperparametrelerini ayarlamak (örn. öğrenme hızı seçimi) ve farklı modelleri karşılaştırmak için kullanılır (%10-15 gibi). Model bu veriyi eğitim sırasında doğrudan görmez.
- **Avantaj:** Modelin görmediği veri üzerindeki performansına dair bir fikir verir.
- **Dezavantaj:** Verinin sadece küçük bir kısmı doğrulama için ayrıldığından, doğrulama performansı bu küçük kümenin özelliklerine bağlı olabilir.

### 4.3. Train-Validation-Test Split
- **Yöntem:** Veri seti üçe ayrılır:
    - **Eğitim (Train) Seti:** Model parametrelerini öğrenir (%60-70 gibi).
    - **Doğrulama (Validation) Seti:** Hiperparametre ayarı ve model seçimi için kullanılır (%10-15 gibi).
    - **Test Seti:** Model eğitimi ve seçimi tamamlandıktan sonra, modelin **nihai performansını** ölçmek için **sadece bir kez** kullanılır (%10-20 gibi). Bu set, model geliştirme sürecinde hiçbir şekilde kullanılmaz.
- **Avantaj:** En güvenilir performans değerlendirmesini sağlar. Modelin gerçek dünya performansını tahmin etmeye en yakın yöntemdir.
- **Dezavantaj:** Daha fazla veriye ihtiyaç duyar.

### 4.4. Çapraz Doğrulama (Cross Validation)
- **Yöntem (K-Fold Cross Validation):** Veri seti K adet eşit (veya yaklaşık eşit) parçaya (fold) bölünür.
    1. Bir parça **test seti** olarak ayrılır.
    2. Kalan K-1 parça **eğitim seti** olarak kullanılır ve model eğitilir.
    3. Ayrılan test seti üzerinde modelin performansı ölçülür.
    4. Bu işlem, her bir parça tam olarak bir kez test seti olacak şekilde K defa tekrarlanır.
    5. K adet performans metriğinin ortalaması alınarak modelin genel performansı hakkında daha sağlam bir tahmin elde edilir.
- **Avantaj:** Verinin tamamı hem eğitim hem de test için kullanılır (farklı iterasyonlarda). Özellikle veri seti küçük olduğunda daha güvenilir bir performans ölçümü sağlar. Veri bölme şekline daha az duyarlıdır.
- **Dezavantaj:** K adet model eğitildiği için hesaplama maliyeti daha yüksektir.

## 5. Makine Öğrenmesi Genel Akışı (Özet)

Tipik bir denetimli öğrenme projesinin adımları şöyledir:

1.  **Veri Toplama ve Hazırlama:** Gerekli veriyi toplama, temizleme, eksik değerleri doldurma, özellikleri seçme/mühendislik yapma.
2.  **Veri Bölme:** Veriyi Train, Validation ve Test setlerine ayırma (veya Cross Validation hazırlığı).
3.  **Model Seçimi:** Probleme uygun bir veya birkaç makine öğrenmesi modeli seçme (örn. Doğrusal Regresyon, KNN, vb.).
4.  **Model Eğitimi:** Eğitim setini kullanarak modelin parametrelerini öğrenme (örn. Gradient Descent ile $\theta$ değerlerini bulma).
5.  **Hiperparametre Ayarlama ve Doğrulama:** Doğrulama setini kullanarak en iyi hiperparametreleri (örn. $\alpha$, K değeri) bulma ve farklı modelleri karşılaştırma.
6.  **Model Değerlendirme:** Eğitim ve ayarlama bittikten sonra, **test setini kullanarak** modelin nihai performansını ölçme.
7.  **Model Dağıtımı ve İzleme:** Modeli kullanıma sunma ve performansını zamanla izleme.

```python
# Genel Akışın Kaba Kodu (Teorik)
# 1. Veri hazırlama
# X, y = veriyi_hazirla(veri_seti)

# 2. Veri bölme (Train/Val/Test örneği)
# X_train, X_val, X_test, y_train, y_val, y_test = veriyi_bol(X, y, train_oran=0.7, val_oran=0.15, test_oran=0.15)

# 3. Model oluşturma
# model = DogrusalRegresyonModeli(ogrenme_hizi=0.01) # Veya başka bir model

# 4. Model eğitme
# model.fit(X_train, y_train, iterasyon_sayisi=1000)

# 5. Doğrulama (Hiperparametre ayarı için döngü içinde olabilir)
# y_tahmin_val = model.predict(X_val)
# val_basari = basari_olc(y_tahmin_val, y_val)
# print(f"Doğrulama Başarısı: {val_basari}")
# -> Farklı öğrenme hızları vb. deneyip en iyi val_basari'yı veren modeli seç

# 6. Test (En iyi model seçildikten sonra SADECE BİR KEZ)
# nihai_tahmin_test = model.predict(X_test)
# nihai_basari = basari_olc(nihai_tahmin_test, y_test)
# print(f"Nihai Test Başarısı: {nihai_basari}")

# 7. Dağıtım...
```

## 6. Ek Detaylar ve Önemli Notlar

### 6.1. Doğrusal Regresyonun Varsayımları
Doğrusal regresyon modelinin güvenilir sonuçlar vermesi için bazı temel varsayımların karşılanması beklenir:
1.  **Doğrusallık (Linearity):** Bağımsız değişkenler ile bağımlı değişken arasında doğrusal bir ilişki olmalıdır. Saçılım grafikleri ile kontrol edilebilir.
2.  **Bağımsız Hatalar (Independence of Errors):** Gözlemlerin hataları birbirinden bağımsız olmalıdır. Özellikle zaman serisi verilerinde önemlidir (örn. Durbin-Watson testi).
3.  **Homoskedastisite (Homoscedasticity):** Hataların varyansı tüm bağımsız değişken seviyeleri için sabit olmalıdır. Hata terimlerinin saçılım grafiği ile kontrol edilebilir (huni şekli olmamalıdır).
4.  **Normal Dağılan Hatalar (Normality of Errors):** Hata terimleri normal dağılıma sahip olmalıdır. Histogram veya Q-Q plot ile kontrol edilebilir. Özellikle küçük veri setlerinde önemlidir.
5.  **Çoklu Doğrusal Bağlantı Yokluğu (No Multicollinearity):** Çok değişkenli regresyonda, bağımsız değişkenler arasında yüksek korelasyon olmamalıdır. VIF (Variance Inflation Factor) değeri ile kontrol edilebilir.

Bu varsayımların ihlali, modelin tahminlerinin ve katsayılarının güvenilirliğini azaltabilir.

### 6.2. Özellik Ölçeklendirme (Feature Scaling)
Gradient Descent gibi algoritmalarda, farklı ölçeklerdeki özellikler (örn. bir özellik 0-1 arasında, diğeri 1000-50000 arasında) optimizasyon sürecini yavaşlatabilir veya zorlaştırabilir. Çünkü büyük ölçekli özellikler gradyanı domine edebilir. Bunu önlemek için özellik ölçeklendirme teknikleri kullanılır:
*   **Standardizasyon (Standardization):** Veriyi ortalaması 0, standart sapması 1 olacak şekilde dönüştürür ($z = (x - \mu) / \sigma$). Aykırı değerlere karşı daha hassastır.
*   **Normalizasyon (Normalization / Min-Max Scaling):** Veriyi genellikle 0 ile 1 arasına sıkıştırır ($x' = (x - min(x)) / (max(x) - min(x))$).

Özellik ölçeklendirme, özellikle çok değişkenli regresyonda ve diğer birçok makine öğrenmesi algoritmasında önemlidir.

### 6.3. Aşırı Öğrenme (Overfitting) ve Düzenlileştirme (Regularization)
*   **Aşırı Öğrenme:** Modelin eğitim verisindeki gürültüyü ve detayları ezberleyerek eğitim setinde çok iyi performans göstermesi, ancak yeni (görülmemiş) verilerde kötü performans göstermesidir. Modelin karmaşıklığı arttıkça (örn. çok fazla özellik veya yüksek dereceli polinom terimleri) overfitting riski artar.
*   **Düzenlileştirme:** Aşırı öğrenmeyi önlemek için kullanılan bir tekniktir. Hata fonksiyonuna, model parametrelerinin (theta değerlerinin) büyüklüğünü cezalandıran bir terim eklenir. Bu, modelin daha basit olmasını (parametrelerin sıfıra yakın olmasını) teşvik eder.
    *   **L1 Düzenlileştirme (Lasso Regression):** Hata fonksiyonuna parametrelerin mutlak değerlerinin toplamını ekler ($\lambda \sum |\theta_j|$). Bazı parametreleri tam olarak sıfır yapabilir, bu da özellik seçimi için kullanışlıdır.
    *   **L2 Düzenlileştirme (Ridge Regression):** Hata fonksiyonuna parametrelerin karelerinin toplamını ekler ($\lambda \sum \theta_j^2$). Parametreleri sıfıra yaklaştırır ancak genellikle tam sıfır yapmaz.

$\lambda$ (lambda), düzenlileştirme gücünü kontrol eden bir hiperparametredir.

## 7. Sıkça Sorulan Sorular (SSS)

**S1: Veri setimdeki ilişki tam olarak doğrusal değilse ne yapmalıyım?**
**C1:**
*   **Polinom Regresyon:** Doğrusal olmayan ilişkileri modellemek için özelliklerin polinom terimlerini (örn. $x^2, x^3$) modele ekleyebilirsiniz. Bu hala doğrusal regresyon çerçevesinde çözülebilir çünkü katsayılara göre doğrusaldır ($y = \theta_0 + \theta_1 x + \theta_2 x^2$).
*   **Özellik Mühendisliği:** Mevcut özelliklerden yeni, daha anlamlı özellikler türetmek (örn. iki özelliği çarpmak, logaritmasını almak) ilişkiyi doğrusallaştırabilir.
*   **Farklı Modeller:** Karar Ağaçları, Destek Vektör Makineleri (SVM), Sinir Ağları gibi doğrusal olmayan ilişkileri daha iyi yakalayabilen başka makine öğrenmesi algoritmalarını deneyebilirsiniz.

**S2: Öğrenme hızı ($\alpha$) nasıl seçilir?**
**C2:** Genellikle deneme yanılma yoluyla bulunur.
*   Farklı $\alpha$ değerleri (örn. 0.1, 0.01, 0.001, 0.0001) denenir.
*   Hata fonksiyonunun iterasyonlara göre grafiği çizilir.
    *   Hata azalmıyor veya artıyorsa: $\alpha$ çok büyük.
    *   Hata çok yavaş azalıyorsa: $\alpha$ çok küçük.
    *   Hata düzgün ve makul bir hızda azalıyorsa: $\alpha$ uygun olabilir.
*   Gelişmiş teknikler olarak öğrenme hızı zamanla azaltılabilir (learning rate decay).

**S3: Aşırı öğrenme (overfitting) nedir ve nasıl önlenir?**
**C3:** Aşırı öğrenme, modelin eğitim verisini ezberlemesi ve yeni verilerde genelleme yapamamasıdır. Önlemek için:
*   **Daha Fazla Veri:** Mümkünse eğitim verisi miktarını artırmak genellemeyi iyileştirir.
*   **Özellik Seçimi:** Daha az sayıda, daha anlamlı özellik kullanmak model karmaşıklığını azaltır.
*   **Düzenlileştirme (Regularization):** L1 (Lasso) veya L2 (Ridge) düzenlileştirme kullanarak model katsayılarını küçültmek.
*   **Çapraz Doğrulama (Cross Validation):** Modelin performansını daha güvenilir bir şekilde değerlendirmek ve hiperparametreleri ayarlamak için kullanılır.

**S4: Doğrulama (Validation) ve Test setleri arasındaki fark nedir?**
**C4:**
*   **Eğitim (Train) Seti:** Modelin parametrelerini (örn. $\theta$ değerleri) öğrenmek için kullanılır.
*   **Doğrulama (Validation) Seti:** Modelin hiperparametrelerini (örn. öğrenme hızı $\alpha$, düzenlileştirme parametresi $\lambda$, polinom derecesi) ayarlamak ve farklı model yapılarını karşılaştırmak için kullanılır. Model bu veriyi görerek ayarlanır.
*   **Test Seti:** Model eğitimi ve hiperparametre ayarı tamamen bittikten sonra, modelin son, tarafsız performansını ölçmek için **sadece bir kez** kullanılır. Bu veri, model geliştirme sürecinde hiçbir şekilde kullanılmamalıdır.

**S5: Doğrusal Regresyon ne zaman kullanılmalıdır?**
**C5:**
*   Sürekli bir değeri (örn. fiyat, sıcaklık, skor) tahmin etmek istediğinizde.
*   Bağımsız değişkenler ile bağımlı değişken arasında doğrusal bir ilişki olduğundan şüphelendiğinizde veya bu ilişkiyi modellemek istediğinizde.
*   Modelin yorumlanabilirliği önemli olduğunda (katsayılar, her bir özelliğin çıktı üzerindeki etkisini gösterir).
*   Hızlı ve basit bir başlangıç modeli gerektiğinde.

Bu belge, doğrusal regresyonun temellerini ve Python uygulamasını kapsamaktadır. Gradient descent optimizasyonu, tek ve çok değişkenli durumlar, model değerlendirme metrikleri, veri bölme stratejileri, varsayımlar ve sıkça sorulan sorular ele alınmıştır.
