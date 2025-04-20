# Yapay Zeka ve Makine Öğrenmesi Notları

## K-En Yakın Komşu (K-Nearest Neighbors)

### 1. Algoritma Temelleri

KNN, sınıflandırma ve regresyon problemlerinde kullanılan bir "lazy learning" (tembel öğrenme) algoritmasıdır. Eğitim aşamasında sadece örnekleri hafızaya alan algoritma, tahmin aşamasında hesaplamalar yapar.

#### 1.1 Çalışma Prensibi

1. Yeni örnek için tüm eğitim örnekleriyle uzaklık hesaplanır
2. En yakın K komşu belirlenir
3. **Sınıflandırma için**: Çoğunluk oylaması ile sınıf tahmin edilir
4. **Regresyon için**: K komşunun ortalama değeri alınır

#### 1.2 Uzaklık Metrikleri

KNN algoritmasında, veri noktaları arasındaki uzaklık çeşitli metriklerle hesaplanabilir:

##### Öklid Uzaklığı
En yaygın kullanılan metriktir. İki nokta arasındaki düz çizgi mesafesini hesaplar.

$$d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

##### Manhattan Uzaklığı
İki nokta arasındaki toplam ızgara mesafesini hesaplar (yalnızca dik açılı hareketlere izin veren "taksi mesafesi" olarak da bilinir).

$$d(x,y) = \sum_{i=1}^n |x_i - y_i|$$

##### Minkowski Uzaklığı
Öklid ve Manhattan uzaklıklarını genelleştiren bir uzaklık metriğidir. p=2 olduğunda Öklid, p=1 olduğunda Manhattan uzaklığına dönüşür.

$$d(x,y) = (\sum_{i=1}^n |x_i - y_i|^p)^{\frac{1}{p}}$$

### 2. Algoritma Karmaşıklığı

KNN algoritması, çoğu makine öğrenmesi algoritmasından farklı olarak bir eğitim modeli oluşturmaz ve parametreleri optimize etmez.

#### 2.1 Eğitim Aşaması
- **Zaman Karmaşıklığı**: O(1) - sadece verileri depolar
- **Uzay Karmaşıklığı**: O(n) - tüm eğitim verilerini saklar
- "Lazy learning" algoritması olduğu için hesaplama maliyeti tahmin aşamasına ertelenir

#### 2.2 Tahmin Aşaması
- **Zaman Karmaşıklığı**: O(nd) - tüm örneklerle uzaklık hesaplar
  - n: örnek sayısı
  - d: özellik sayısı
- **Uzay Karmaşıklığı**: O(k) - en yakın k komşuyu saklar

Bu karmaşıklık analizi, KNN'in küçük ve orta boyuttaki veri setleri için uygun olduğunu, ancak büyük veri setlerinde performans sorunları yaşayabileceğini gösterir.

### 3. Hiperparametreler

#### 3.1 K Değeri Seçimi

K parametresi, algoritmanın en önemli hiperparametresidir ve model performansını doğrudan etkiler:

- **Çok küçük K** (örneğin K=1): Modelin gürültüye karşı hassas olmasına ve aşırı uyum (overfitting) göstermesine neden olur
- **Çok büyük K**: Modelin yetersiz uyum (underfitting) göstermesine ve karar sınırlarının çok düzgünleşmesine yol açar
- Genelde oy eşitliğini önlemek için **tek sayı** tercih edilir (sınıflandırma problemlerinde)
- **Cross-validation** (çapraz doğrulama) ile optimize edilir
- Yaygın başlangıç noktası olarak √n (n: örnek sayısı) kullanılabilir

#### 3.2 Ağırlıklı Oylama

Standart KNN'de tüm komşular eşit ağırlığa sahiptir. Ağırlıklı KNN'de ise her komşu, test noktasına olan uzaklığına göre farklı ağırlıklar alır:

$$w_i = \frac{1}{d(x,x_i)^2}$$

Böylece yakın komşulara daha fazla önem verilir, bu da özellikle sınır durumlarında daha doğru tahminler yapılmasını sağlar.

### 4. Veri Ön İşleme

KNN algoritması, veri ölçeklendirme ve boyut indirgeme gibi ön işleme adımlarına oldukça duyarlıdır.

#### 4.1 Feature Scaling (Özellik Ölçeklendirme)

KNN uzaklık metriklerine dayandığından, farklı ölçeklerdeki özelliklerin dengelenmesi önemlidir:

$$x_{norm} = \frac{x - \mu}{\sigma}$$


#### Z-Skor Normalizasyonu (Standart Skor)
Bu formül, Z-skor normalizasyonu veya standartlaştırma olarak bilinen bir özellik ölçeklendirme tekniğini temsil eder.

$$x_{norm} = \frac{x - \mu}{\sigma}$$

Formüldeki değişkenler:

$x$ = Orijinal veri noktası
$\mu$ = Veri setinin ortalaması (mean)
$\sigma$ = Veri setinin standart sapması
$x_{norm}$ = Normalize edilmiş (standartlaştırılmış) veri noktası
İşlem Adımları
Her veri noktasından veri setinin ortalamasını çıkar ($x - \mu$)
Sonucu standart sapmaya böl ($(x - \mu) / \sigma$)
Sonuçlar ve Faydaları
Normalleştirme sonucunda veri ortalaması 0, standart sapması 1 olur
Veriler standart normal dağılıma (z-dağılımı) dönüştürülür
Farklı ölçeklerdeki özellikler karşılaştırılabilir hale gelir
Elde edilen değerler, orijinal değerin ortalamadan kaç standart sapma uzakta olduğunu gösterir

**Özellik Ölçeklendirme Yöntemleri**:
- **Standart Scaler**: Verileri ortalama=0, standart sapma=1 olacak şekilde dönüştürür
- **Min-Max Scaler**: Verileri [0,1] aralığına ölçeklendirir
- **Robust Scaler**: Aykırı değerlere karşı dayanıklı ölçeklendirme sağlar

#### 4.2 Boyut İndirgeme

Yüksek boyutlu verilerde "boyutsallık laneti" (curse of dimensionality) problemi ortaya çıkar ve KNN'in etkinliği azalır. Bu nedenle boyut indirgeme teknikleri kullanılabilir:

- **PCA** (Principal Component Analysis): Doğrusal boyut indirgeme
- **t-SNE** (t-distributed Stochastic Neighbor Embedding): Doğrusal olmayan boyut indirgeme
- **UMAP** (Uniform Manifold Approximation and Projection): Modern ve hızlı boyut indirgeme tekniği

### 5. Optimizasyon Teknikleri

KNN'in tahmin aşamasındaki yüksek hesaplama maliyetini azaltmak için çeşitli optimizasyon teknikleri geliştirilmiştir:

#### 5.1 Ball Tree

Veri noktalarını küresel sınırlarla gruplandırarak arama işlemini hızlandırır:

- Uzayı küresel bölgelere böler
- Ortalama arama zamanını O(n) -> O(log n) seviyesine indirir
- Özellikle yüksek boyutlu verilerde etkilidir

#### 5.2 KD Tree (k-dimensional tree)

Veri noktalarını dikdörtgen bölgelere ayırarak hiyerarşik bir ağaç yapısı oluşturur:

- Her seviyede uzayı bir özellik boyunca böler
- Dengeli ağaç yapısı sayesinde sorgu zamanını iyileştirir
- Düşük-orta boyutlu uzaylarda (d < 20) etkilidir

#### 5.3 Approximate Nearest Neighbors

Büyük veri setlerinde tam doğru en yakın komşuları bulmak yerine, yaklaşık sonuçlar veren hızlı algoritmalar kullanılabilir:

- **LSH** (Locality Sensitive Hashing): Benzer noktaları aynı "kovaya" atmayı amaçlayan hash fonksiyonları kullanır
- Kesin doğruluktan biraz ödün vererek çok daha hızlı sonuçlar elde edilir
- Milyonlarca veya milyarlarca veri noktası içeren problemlerde tercih edilir

### 6. Uygulama Örnekleri

KNN algoritması çeşitli problemlerde esnek bir şekilde kullanılabilir:

#### 6.1 Sınıflandırma

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Model oluşturma
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)  # p=2 Öklid uzaklığı

# Eğitim
knn.fit(X_train, y_train)

# Tahmin
y_pred = knn.predict(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")
```

#### 6.2 Regresyon

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Model oluşturma
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # uzaklığa bağlı ağırlıklandırma

# Eğitim
knn.fit(X_train, y_train)

# Tahmin
y_pred = knn.predict(X_test)

# Performans değerlendirme
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
```

#### 6.3 Anomali Tespiti

```python
from sklearn.neighbors import LocalOutlierFactor

# Model oluşturma
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Aykırı değer tespiti
outlier_scores = lof.fit_predict(X)

# -1: aykırı değer, 1: normal veri
outliers = X[outlier_scores == -1]
```

### 7. Avantajlar ve Dezavantajlar

#### Avantajlar
- Basit ve sezgisel bir algoritma
- Eğitim aşaması çok hızlıdır (verileri sadece depolar)
- Doğrusal olmayan karar sınırlarını modelleyebilir
- Parametrik olmayan bir model (veri dağılımı hakkında varsayım yapmaz)
- Hem sınıflandırma hem de regresyon problemlerinde kullanılabilir

#### Dezavantajlar
- Büyük veri setlerinde tahmin aşaması yavaştır
- Yüksek boyutlu verilerde performansı düşer (boyutsallık laneti)
- Gürültülü verilerden etkilenir
- Eğitim verisinin tamamını bellekte tutması gerekir
- Eksik verilerle doğrudan çalışamaz

### 8. KNN ve Doğrusal Regresyon Arasındaki Farklar

| Özellik | KNN | Doğrusal Regresyon |
|---------|-----|-------------------|
| Model tipi | Parametrik olmayan | Parametrik |
| Eğitim süreci | Örnek depolama (O(1)) | Parametre optimizasyonu (O(nd²)) |
| Tahmin süreci | Komşu arama (O(nd)) | Basit çarpım (O(d)) |
| Bellek gereksinimi | Yüksek (tüm örnekler) | Düşük (sadece parametreler) |
| Yorumlanabilirlik | Düşük | Yüksek |
| Doğrusal olmayan ilişkiler | Doğal olarak modelleyebilir | Ek dönüşümler gerektirir |
| Büyük veri setleri | Zorluk yaşar | Daha verimli çalışır |
| Baz alınan matematiksel kavram | Uzaklık metrikleri | En küçük kareler optimizasyonu |

### 9. Sorulabilecek Sorular

1. **Soru**: KNN algoritmasında K değerinin seçimi neden önemlidir ve nasıl belirlenir?
   **Cevap**: K değeri, KNN algoritmasının tahmin sonuçlarını doğrudan etkiler. Çok küçük K değerleri (örneğin K=1) modelin gürültüye karşı hassas olmasına ve aşırı uyum (overfitting) göstermesine neden olur. Çok büyük K değerleri ise modelin yetersiz uyum (underfitting) göstermesine ve karar sınırlarının çok düzgünleşmesine yol açar. K değeri genellikle tek sayı olarak seçilir (sınıflandırma problemlerinde oy eşitliğini önlemek için) ve çapraz doğrulama (cross-validation) yöntemiyle optimum değer belirlenir. Genellikle √n (n: örnek sayısı) değerinden başlayarak deneme yapılır.

2. **Soru**: KNN algoritmasında "curse of dimensionality" (boyutsallık laneti) problemi nedir ve bu problemi nasıl ele alabilirsiniz?
   **Cevap**: Boyutsallık laneti, özellik sayısı arttıkça veri noktaları arasındaki ortalama uzaklığın da artması ve KNN algoritmasının etkinliğinin azalması problemidir. Yüksek boyutlu uzayda, veri noktaları gittikçe seyrekleşir ve mesafe metrikleri anlamlılığını kaybeder. Bu sorunla başa çıkmak için: 1) Özellik seçimi (feature selection): İlgisiz veya az bilgi içeren özellikleri elemek, 2) Boyut indirgeme teknikleri (PCA, t-SNE, UMAP) kullanmak, 3) Özellik ağırlıklandırma: Önemli özelliklere daha fazla ağırlık vermek, 4) Farklı uzaklık metrikleri kullanmak, 5) Yerel uzay yapısını dikkate alan yaklaşımlar (manifold learning) uygulamak.

3. **Soru**: KNN ve Doğrusal Regresyon algoritmalarının temel çalışma prensipleri ve uygulama alanları açısından nasıl karşılaştırırsınız?
   **Cevap**: KNN, örnek tabanlı ve parametrik olmayan bir algoritmadır; yeni bir örneği sınıflandırmak/tahmin etmek için eğitim verilerindeki en yakın K komşusuna bakar. Doğrusal regresyon ise parametrik bir modeldir; bağımsız değişkenler ve bağımlı değişken arasında doğrusal bir ilişki kurar ve parametreleri optimize eder. KNN, doğrusal olmayan karmaşık ilişkileri modelleyebilirken, doğrusal regresyon daha basit ve yorumlanabilir modeller sunar. KNN, küçük-orta ölçekli veri setleri için uygundur ve anomali tespiti gibi görevlerde etkilidir. Doğrusal regresyon ise değişkenler arasında doğrusal ilişki olduğunda, özellik önem analizi gerektiren durumlarda ve büyük veri setlerinde daha verimlidir.

4. **Soru**: KNN algoritmasında uzaklık metriklerinin seçimi model performansını nasıl etkiler? Hangi durumlarda hangi uzaklık metriğini tercih edersiniz?
   **Cevap**: Uzaklık metriği seçimi, KNN'in komşuluk ilişkilerini nasıl tanımladığını belirler ve bu da model tahminlerini doğrudan etkiler. Öklid uzaklığı (p=2), genel amaçlı kullanımda ve sürekli değişkenlerde tercih edilir. Manhattan uzaklığı (p=1), özelliklerin eksenlere paralel olduğu durumlarda ve şehir bloğu tipindeki problemlerde daha uygundur. Minkowski uzaklığı, genel bir formül olup p parametresi ayarlanarak farklı problem tiplerine uyarlanabilir. Kategorik değişkenler için Hamming uzaklığı tercih edilir. Ayrıca, yüksek boyutlu verilerde kosinüs benzerliği gibi metrikler daha etkili olabilir. Veri yapısına uygun uzaklık metriği seçimi, cross-validation ile deneysel olarak belirlenmeli ve optimize edilmelidir.

5. **Soru**: KNN algoritmasında ağırlıklı ve ağırlıksız oylama arasındaki fark nedir? Hangi durumlarda ağırlıklı oylama daha avantajlı olur?
   **Cevap**: Ağırlıksız oylamada, K en yakın komşunun her biri eşit oy hakkına sahiptir ve çoğunluk kararı tahmin olarak kullanılır. Ağırlıklı oylamada ise her komşu, test noktasına olan uzaklığına göre farklı ağırlıklar alır - genellikle uzaklığın karesiyle ters orantılı olarak: $w_i = 1/d(x,x_i)^2$. Ağırlıklı oylama, özellikle 1) Veri noktalarının yoğunluğunun değişken olduğu durumlarda, 2) Sınıflar arasındaki sınırların belirgin olmadığı durumlarda, 3) Gürültülü verilerde, 4) Uzaktaki komşuların etkisini azaltmak istediğimizde daha avantajlıdır. Ayrıca K değeri büyük seçildiğinde, ağırlıklı oylama yakın komşuların etkisini korumaya yardımcı olur ve modelin hassasiyetini artırır.