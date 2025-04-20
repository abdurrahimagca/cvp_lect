# Yapay Zeka ve Makine Öğrenmesi Notları

## K-En Yakın Komşu (K-Nearest Neighbors)

### 1. Algoritma Temelleri

KNN, sınıflandırma ve regresyon problemlerinde kullanılan bir **\"lazy learning\" (tembel öğrenme)** algoritmasıdır. Bu terim, algoritmanın geleneksel anlamda bir \"eğitim\" aşaması olmamasından gelir. Model, eğitim verilerini ezberler (hafızaya alır) ve tahmin yapması istendiğinde bu depolanmış veriler üzerinden hesaplama yapar. Yani, hesaplama yükü tahmin aşamasına ertelenir.

#### 1.1 Çalışma Prensibi

Bir yeni veri noktası (tahmin yapılacak örnek) geldiğinde KNN şu adımları izler:

1.  **Uzaklık Hesaplama**: Yeni örneğin, eğitim setindeki **tüm** örneklere olan uzaklığı belirlenen bir metrik (örn. Öklid) kullanılarak hesaplanır.
2.  **Komşu Bulma**: Hesaplanan uzaklıklara göre yeni örneğe en yakın **K** adet eğitim örneği (komşu) seçilir.
3.  **Tahmin Yapma**:
    *   **Sınıflandırma için**: En yakın K komşunun ait olduğu sınıflara bakılır. En çok tekrar eden sınıf (çoğunluk oylaması), yeni örneğin sınıfı olarak tahmin edilir. Beraberlik durumlarını önlemek için K genellikle tek sayı seçilir.
    *   **Regresyon için**: En yakın K komşunun hedef değerlerinin (sayısal değerler) ortalaması (veya ağırlıklı ortalaması) alınarak yeni örneğin değeri tahmin edilir.

#### 1.2 Uzaklık Metrikleri

KNN algoritmasında, veri noktaları arasındaki \"yakınlık\" veya \"benzerlik\" çeşitli metriklerle ölçülür. Seçilen metrik, algoritmanın performansını etkileyebilir.

##### Öklid Uzaklığı (Euclidean Distance)
En yaygın kullanılan metriktir. İki nokta arasındaki en kısa, **düz çizgi mesafesini** temsil eder. İki nokta $x = (x_1, x_2, ..., x_n)$ ve $y = (y_1, y_2, ..., y_n)$ için n boyutlu uzayda şu şekilde hesaplanır:

$$d(x,y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}$$

##### Manhattan Uzaklığı (Manhattan Distance)
İki nokta arasındaki mesafeyi, koordinat eksenlerine paralel hareketlerle (bir şehirdeki binalar arasındaki yollar gibi) hesaplar. Bu nedenle \"şehir bloğu\" veya \"taksi mesafesi\" olarak da bilinir.

$$d(x,y) = \\sum_{i=1}^n |x_i - y_i|$$

*Örnek*: 2D uzayda (2, 3) ve (5, 7) noktaları arası Manhattan uzaklığı: $|5-2| + |7-3| = 3 + 4 = 7$.

##### Minkowski Uzaklığı (Minkowski Distance)
Öklid ve Manhattan uzaklıklarını genelleştiren bir metriktir. **p** parametresine bağlıdır:

$$d(x,y) = (\\sum_{i=1}^n |x_i - y_i|^p)^{\\frac{1}{p}}$$

*   **p=1** ise **Manhattan** uzaklığına eşittir.
*   **p=2** ise **Öklid** uzaklığına eşittir.
*   **p=∞** ise **Chebyshev** uzaklığına (maksimum koordinat farkı) yaklaşır.

*Diğer Metrikler*: Kosinüs benzerliği (özellikle metin verilerinde), Hamming uzaklığı (kategorik verilerde), Mahalanobis uzaklığı (verilerin korelasyonunu dikkate alır) gibi farklı metrikler de kullanılabilir.

### 2. Algoritma Karmaşıklığı

KNN'in hesaplama maliyeti, diğer birçok algoritmadan farklı olarak eğitim ve tahmin aşamalarında dengesizdir.

#### 2.1 Eğitim Aşaması
*   **Zaman Karmaşıklığı**: **O(1)** - Çünkü bu aşamada sadece veri setini hafızaya alır, herhangi bir model parametresi hesaplamaz veya optimizasyon yapmaz.
*   **Uzay Karmaşıklığı**: **O(nd)** - Tüm eğitim verilerini (n örnek, d özellik) saklamak zorundadır. Bu, büyük veri setleri için önemli bir bellek gereksinimi anlamına gelir.
*   \"Lazy learning\" olmasının sonucu: Hesaplama maliyeti tahmin aşamasına kaydırılır.

#### 2.2 Tahmin Aşaması
*   **Zaman Karmaşıklığı**: **O(nd)** - Yeni bir örneğin tahminini yapmak için, bu örneğin eğitim setindeki **tüm n örnekle** olan uzaklığını hesaplamak gerekir. Her uzaklık hesaplaması d özelliğin karşılaştırılmasını içerir. Bu, KNN'in büyük veri setlerinde (n büyük olduğunda) yavaş olmasının ana nedenidir. Optimizasyon teknikleri (KD-Tree, Ball Tree) bu süreyi ortalamada düşürebilir, ancak en kötü durum karmaşıklığı genellikle O(nd) kalır.
*   **Uzay Karmaşıklığı**: **O(kd)** - Tahmin sırasında en yakın k komşunun bilgilerini geçici olarak saklamak gerekebilir. Genellikle k, n'den çok daha küçük olduğu için bu önemli bir ek yük getirmez.

Bu karmaşıklık analizi, KNN'in **küçük ve orta boyutlu veri setleri** için pratik olduğunu, ancak **büyük veri setlerinde (milyonlarca örnek)** tahmin süresinin kabul edilemez derecede uzayabileceğini gösterir.

### 3. Hiperparametreler

KNN'in performansını ayarlamak için kullanılan temel hiperparametreler şunlardır:

#### 3.1 K Değeri Seçimi

**K** (komşu sayısı), algoritmanın en kritik hiperparametresidir ve modelin genelleme yeteneğini doğrudan etkiler:

*   **Çok küçük K** (örneğin K=1): Model, eğitim verisindeki gürültüye ve aykırı değerlere karşı çok **hassas** hale gelir. Karar sınırları çok düzensizleşir ve model **aşırı uyum (overfitting)** eğilimi gösterir. Yani, eğitim verisini çok iyi öğrenir ama yeni, görülmemiş verilerde kötü performans gösterir.
*   **Çok büyük K**: Model, farklı sınıflardan veya uzak bölgelerden komşuları da dikkate almaya başlar. Karar sınırları aşırı **düzgünleşir** ve model verinin yerel yapısını yakalayamaz. Bu durum **yetersiz uyum (underfitting)** olarak adlandırılır. Model, eğitim verisini bile yeterince iyi öğrenemez.
*   **Optimum K Seçimi**:
    *   Genellikle oy eşitliğini önlemek için sınıflandırma problemlerinde **tek sayı** tercih edilir.
    *   En iyi K değeri probleme ve veriye bağlıdır. Standart yöntem, farklı K değerlerini deneyerek **çapraz doğrulama (cross-validation)** ile en iyi performansı (örn. doğruluk, F1 skoru) veren K değerini bulmaktır.
    *   Yaygın bir başlangıç noktası olarak **√n** (n: eğitim örnek sayısı) değeri önerilebilir, ancak bu sadece bir başlangıç noktasıdır ve optimize edilmelidir.

#### 3.2 Ağırlıklı Oylama (Weighted Voting)

Standart KNN'de, K komşunun hepsi tahmin üzerinde eşit etkiye sahiptir. Ağırlıklı KNN'de ise her komşunun oyu veya katkısı, test noktasına olan **uzaklığına göre ağırlıklandırılır**. Yakın komşular daha fazla, uzak komşular daha az etkiye sahip olur.

Yaygın bir ağırlıklandırma şeması, uzaklığın tersinin karesini kullanmaktır:

$$w_i = \\frac{1}{d(x_{test}, x_i)^p}$$

Burada $d(x_{test}, x_i)$ test noktası ile i'inci komşu arasındaki uzaklık, $p$ ise genellikle 1 veya 2 olan bir üs değeridir (p=2 daha yaygındır).

*   **Avantajı**: Sınıf sınırlarına yakın noktalarda veya seyrek bölgelerde daha sağlam tahminler yapmaya yardımcı olabilir. Uzaktaki komşuların potansiyel olarak yanıltıcı etkisini azaltır.
*   **Kullanım**: `scikit-learn`'de `KNeighborsClassifier` veya `KNeighborsRegressor` içinde `weights='distance'` parametresi ile aktive edilir.

#### 3.3 Uzaklık Metriği Seçimi (`metric` parametresi)

Kullanılacak uzaklık metriği de bir hiperparametredir (`metric` veya `p` parametresi ile `scikit-learn`'de ayarlanır). Öklid (`'euclidean'` veya `p=2`) varsayılan ve en yaygın olanıdır, ancak probleme göre Manhattan (`'manhattan'` veya `p=1`) veya diğer metrikler daha iyi sonuç verebilir. Bu da çapraz doğrulama ile test edilebilir.

### 4. Veri Ön İşleme

KNN algoritması, girdi verisinin özelliklerine ve ölçeklerine oldukça **duyarlıdır**. Bu nedenle uygun veri ön işleme adımları genellikle zorunludur.

#### 4.1 Feature Scaling (Özellik Ölçeklendirme)

KNN, uzaklık hesaplamalarına dayandığı için, farklı birimlerde veya farklı aralıklarda olan özellikler (örn. yaş [0-100] ve maaş [1000-100000]) uzaklık hesaplamalarını **domine edebilir**. Büyük aralıktaki özellikler, küçük aralıktaki özelliklere göre uzaklık üzerinde çok daha fazla etkiye sahip olur, bu da modelin yanlı olmasına neden olabilir. Bunu önlemek için özellik ölçeklendirme **kritik öneme sahiptir**.

##### Z-Skor Normalizasyonu (Standardizasyon - Standardization)
Bu yöntem, her özelliğin ortalamasını 0 ve standart sapmasını 1 olacak şekilde dönüştürür. Aykırı değerlere karşı Min-Max'a göre daha dayanıklıdır.

$$x_{scaled} = \\frac{x - \\mu}{\\sigma}$$

Formüldeki değişkenler:
*   $x$: Orijinal özellik değeri
*   $\mu$: Özelliğin veri setindeki ortalaması (mean)
*   $\sigma$: Özelliğin veri setindeki standart sapması (standard deviation)
*   $x_{scaled}$: Ölçeklendirilmiş (standartlaştırılmış) özellik değeri

*Sonuçlar ve Faydaları*:
*   Ölçeklendirme sonrası her özelliğin ortalaması 0, standart sapması 1 olur.
*   Veriler standart normal dağılıma (z-dağılımı) benzer bir yapıya getirilir (tam olarak normal dağılıma dönüştürmez).
*   Farklı ölçeklerdeki özellikler karşılaştırılabilir ve uzaklık hesaplamalarında eşit derecede etkili hale gelir.
*   Elde edilen değerler, orijinal değerin ortalamadan kaç standart sapma uzakta olduğunu gösterir.

**Diğer Yaygın Ölçeklendirme Yöntemleri**:
*   **Min-Max Scaler**: Verileri belirli bir aralığa, genellikle [0, 1] veya [-1, 1] aralığına sıkıştırır. $x_{scaled} = \\frac{x - min(x)}{max(x) - min(x)}$. Aykırı değerlere karşı hassastır.
*   **Robust Scaler**: Medyanı çıkarıp çeyrekler açıklığına (IQR - Interquartile Range) bölerek ölçeklendirme yapar. Aykırı değerlerin etkisini azaltmak için tasarlanmıştır.

**Hangi Yöntem?**: Genellikle Standard Scaler iyi bir başlangıç noktasıdır. Veri dağılımı ve aykırı değerlerin varlığına göre diğerleri denenebilir.

#### 4.2 Boyut İndirgeme (Dimensionality Reduction)

Özellik sayısı (boyut - d) çok yüksek olduğunda KNN'in performansı düşer. Bu duruma **\"boyutsallık laneti\" (curse of dimensionality)** denir. Sebepleri:

*   **Seyreklik**: Yüksek boyutlu uzayda veri noktaları arasındaki ortalama uzaklık artar, noktalar birbirinden çok uzaklaşır ve \"en yakın\" komşu kavramı anlamını yitirmeye başlar.
*   **Hesaplama Maliyeti**: Tahmin süresi (O(nd)) boyut sayısı (d) ile doğru orantılı olarak artar.
*   **Gürültü Etkisi**: İlgisiz veya gürültülü özelliklerin sayısı arttıkça, uzaklık hesaplamaları yanıltıcı olabilir.

Bu sorunları hafifletmek için boyut indirgeme teknikleri kullanılabilir:

*   **PCA (Principal Component Analysis - Temel Bileşen Analizi)**: Verideki varyansı en çok açıklayan yeni, daha az sayıda ve birbiriyle ilişkisiz (ortogonal) özellikler (temel bileşenler) bulur. Doğrusal bir tekniktir.
*   **t-SNE (t-distributed Stochastic Neighbor Embedding)**: Yüksek boyutlu verinin düşük boyutlu (genellikle 2D veya 3D) bir uzayda görselleştirilmesi için kullanılan, doğrusal olmayan bir tekniktir. Komşuluk ilişkilerini korumaya çalışır. Genellikle veri keşfi için kullanılır, doğrudan KNN girdisi olarak kullanımı daha az yaygındır.
*   **UMAP (Uniform Manifold Approximation and Projection)**: t-SNE'ye benzeyen, ancak genellikle daha hızlı çalışan ve hem görselleştirme hem de genel boyut indirgeme için kullanılabilen modern, doğrusal olmayan bir tekniktir.

Boyut indirgeme, hem hesaplama süresini kısaltabilir hem de ilgisiz özelliklerin etkisini azaltarak KNN'in doğruluğunu artırabilir.

### 5. Optimizasyon Teknikleri

KNN'in O(nd) olan naif tahmin süresini iyileştirmek için çeşitli veri yapıları ve algoritmalar geliştirilmiştir. `scikit-learn` gibi kütüphaneler, veri setinin özelliklerine göre bu tekniklerden uygun olanı (`algorithm` parametresi: `'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`) otomatik olarak seçmeye çalışır.

#### 5.1 Ball Tree

Veri noktalarını iç içe geçmiş **hiper-küreler (balls)** kullanarak hiyerarşik bir ağaç yapısında organize eder. Her düğüm, belirli bir merkez etrafındaki bir grup noktayı ve bu grubu çevreleyen kürenin yarıçapını temsil eder. Arama sırasında, sorgu noktasına uzak olan küreler (ve içindeki tüm noktalar) tamamen elenerek arama uzayı daraltılır.

*   Uzayı küresel bölgelere böler.
*   Ortalama arama zamanını O(n) seviyesinden **O(log n)** seviyesine indirebilir (veri dağılımına bağlı olarak).
*   Özellikle **yüksek boyutlu** verilerde ve çeşitli uzaklık metrikleriyle (Öklid dışı) KD Tree'ye göre daha etkili olabilir.

#### 5.2 KD Tree (k-dimensional tree)

Veri noktalarını, her seviyede farklı bir **boyut (eksen)** boyunca medyan değere göre ikiye bölerek hiyerarşik bir **dikdörtgen (veya hiper-dikdörtgen)** bölgeler ağı oluşturur. Arama, sorgu noktasını içeren bölgeye inerek ve komşu bölgeleri gerektiğinde kontrol ederek yapılır.

*   Her seviyede uzayı bir özellik (koordinat ekseni) boyunca böler.
*   Dengeli bir ağaç yapısı oluşturarak sorgu zamanını ortalamada **O(log n)** seviyesine indirebilir.
*   Genellikle **düşük ve orta boyutlu (d < 20 civarı)** uzaylarda Ball Tree'den daha hızlı olabilir, ancak yüksek boyutlarda etkinliği azalır. Sadece belirli metriklerle (örn. Minkowski metrikleri) verimli çalışır.

#### 5.3 Approximate Nearest Neighbors (ANN - Yaklaşık En Yakın Komşular)

Çok büyük veri setlerinde (milyonlarca/milyarlarca örnek) veya çok yüksek boyutlu uzaylarda, tam olarak en yakın komşuları bulmak yerine, çok daha hızlı bir şekilde **yüksek olasılıkla** en yakın komşuları bulan yaklaşık algoritmalar kullanılır. Bu algoritmalar, doğruluktan küçük bir ödün vererek arama süresini **önemli ölçüde** azaltır.

*   **LSH (Locality Sensitive Hashing - Yerelliğe Duyarlı Karma)**: Benzer veri noktalarının aynı veya yakın \"karma (hash) kovalarına\" düşme olasılığını artıran özel hash fonksiyonları kullanır. Sorgu noktasının düştüğü kova(lar)daki noktalar potansiyel komşu olarak değerlendirilir.
    *   **Temel Prensip**: Geometrik olarak birbirine yakın noktaların, özel tasarlanmış hash fonksiyonları altında aynı hash değerini üretme olasılığı, uzak noktalara göre daha yüksektir.
    *   **Hash Fonksiyonları Ailesi**: Kullanılan uzaklık metriğine göre farklı LSH aileleri vardır:
        *   **MinHash**: Jaccard benzerliği (kümeler arası örtüşme) için. Metin belgeleri, setler.
        *   **SimHash**: Kosinüs benzerliği (açısal benzerlik) için. Metin belgeleri, vektörler.
        *   **Random Projection LSH (E2LSH)**: Öklid uzaklığı için. Vektör uzayları.
    *   **Çalışma Prensibi**:
        1.  Birden fazla (genellikle bağımsız) hash fonksiyonu kullanılarak veri noktaları için hash değerleri (veya imzalar) oluşturulur.
        2.  Bu hash değerleri kullanılarak noktalar bir veya daha fazla hash tablosuna yerleştirilir.
        3.  Sorgu noktası geldiğinde, onun hash değer(ler)i hesaplanır ve aynı tablo konumuna (kovaya) düşen noktalar potansiyel komşu olarak alınır.
        4.  Bu potansiyel komşular arasında (daha küçük bir sette) gerçek uzaklık hesaplanarak en yakın K komşu bulunur.
    *   **Performans İyileştirme Teknikleri**: Multi-probe LSH (yakın kovaları da kontrol etme), LSH Forest (uyarlanabilir yapı), LSH Ensemble (en iyi şemayı seçme).
    *   **Uygulama Alanları**: Büyük ölçekli benzer görüntü/müzik/video arama, kopya tespiti, öneri sistemleri, genom analizi.
    *   **Kompleksite**: Brute-force O(nd) yerine, uygun parametrelerle sorgu zamanını yaklaşık **O(d * log n)** veya daha iyi seviyelere indirebilir (tablo sayısı ve boyutuna bağlı olarak alt-doğrusal olabilir).

*   **HNSW (Hierarchical Navigable Small World - Hiyerarşik Gezilebilir Küçük Dünya)**: Modern, graf tabanlı ve genellikle LSH'den daha hızlı ve daha doğru sonuçlar veren popüler bir ANN algoritmasıdır.
    *   **Küçük Dünya Ağları**: Veri noktalarını, hem yerel (yakın komşular) hem de global (uzak noktalar arası kestirme yollar) bağlantılar içeren bir graf yapısında organize eder. Bu, aramaların graf üzerinde verimli bir şekilde gezinmesini sağlar.
    *   **Hiyerarşik Yapı**: Graf, farklı yoğunluklarda veya ölçeklerde katmanlar halinde oluşturulur. Arama en üst (en seyrek) katmandan başlar ve giderek daha yoğun alt katmanlara inerek hedef noktaya yaklaşır.
    *   **Greedy Arama**: Her katmanda, mevcut noktadan hedef noktaya en yakın komşuya doğru ilerleyen bir açgözlü arama stratejisi kullanılır.
    *   **Performans**: Milyarlarca veri noktası ve yüksek boyutlu vektörler üzerinde bile **milisaniyeler** mertebesinde sorgu süreleri elde edebilir.
    *   **Ölçeklenebilirlik**: Yüksek boyutlu vektör verilerinde (örn. derin öğrenme çıktıları, kelime gömmeleri) bile etkinliğini korur.
    *   **Uygulamalar**: Vektör veritabanları (Faiss, Milvus, Weaviate), gerçek zamanlı benzerlik arama, öneri sistemleri, anomali tespiti.

ANN algoritmaları, kesin en yakın komşuyu garantilemez ancak pratikte çok yüksek doğruluk oranları (%95+) ile çok büyük hızlanmalar sağlarlar.

### 6. Uygulama Örnekleri (`scikit-learn` ile)

Aşağıda `scikit-learn` kütüphanesi kullanılarak KNN'in temel uygulamaları gösterilmiştir. `X_train`, `y_train`, `X_test`, `y_test` ve `X` değişkenlerinin uygun şekilde hazırlanmış veri setleri olduğu varsayılmıştır (örn. `train_test_split` ile bölünmüş ve ölçeklendirilmiş).

#### 6.1 Sınıflandırma (Classification)

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris # Örnek veri seti

# 1. Veri Yükleme ve Hazırlama
iris = load_iris()
X, y = iris.data, iris.target

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Özellik Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Test setine sadece transform uygulanır

# 3. Model Oluşturma ve Eğitim
# n_neighbors: K değeri
# weights: 'uniform' (eşit ağırlık) veya 'distance' (uzaklığa göre ağırlık)
# p: Minkowski metriği için üs (1: Manhattan, 2: Öklid)
# metric: Kullanılacak uzaklık metriği ('euclidean', 'manhattan', 'minkowski' vb.)
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean') # p=2 de aynı anlama gelir

# Eğitim (Aslında sadece veriyi saklama)
knn_clf.fit(X_train_scaled, y_train)

# 4. Tahmin Yapma
y_pred = knn_clf.predict(X_test_scaled)

# 5. Performans Değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu (K=5, Öklid, Uniform): {accuracy:.4f}")
print("\\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Farklı parametrelerle deneyelim: K=7, Manhattan, Ağırlıklı
knn_clf_weighted = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan') # p=1
knn_clf_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = knn_clf_weighted.predict(X_test_scaled)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
print(f"\\nModel Doğruluğu (K=7, Manhattan, Distance): {accuracy_weighted:.4f}")

```

#### 6.2 Regresyon (Regression)

```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing # Örnek veri seti

# 1. Veri Yükleme ve Hazırlama
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Özellik Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Oluşturma ve Eğitim
# n_neighbors: K değeri
# weights: 'uniform' (komşuların ortalaması) veya 'distance' (uzaklığa göre ağırlıklı ortalama)
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean')

# Eğitim
knn_reg.fit(X_train_scaled, y_train)

# 4. Tahmin Yapma
y_pred = knn_reg.predict(X_test_scaled)

# 5. Performans Değerlendirme
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performansı (K=5, Öklid, Distance):")
print(f"  RMSE (Kök Ortalama Kare Hata): {rmse:.4f}")
print(f"  R-kare (R2 Score): {r2:.4f}")

# Farklı parametrelerle deneyelim: K=10, Uniform
knn_reg_uniform = KNeighborsRegressor(n_neighbors=10, weights='uniform', metric='euclidean')
knn_reg_uniform.fit(X_train_scaled, y_train)
y_pred_uniform = knn_reg_uniform.predict(X_test_scaled)
rmse_uniform = np.sqrt(mean_squared_error(y_test, y_pred_uniform))
r2_uniform = r2_score(y_test, y_pred_uniform)
print(f"\\nModel Performansı (K=10, Öklid, Uniform):")
print(f"  RMSE: {rmse_uniform:.4f}")
print(f"  R2 Score: {r2_uniform:.4f}")

```

#### 6.3 Anomali Tespiti (Outlier Detection)

KNN fikri, anomali (aykırı değer) tespiti için de kullanılabilir. Bir noktanın komşularına olan ortalama uzaklığı, o noktanın ne kadar "izole" olduğunu gösterebilir. `LocalOutlierFactor` bu prensibe dayanır.

```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Örnek veri oluşturalım (çoğunluk normal, azınlık aykırı)
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

# Model oluşturma
# n_neighbors: Komşuluk tanımı için K değeri
# contamination: Veri setindeki beklenen aykırı değer oranı ('auto' veya 0 ile 0.5 arası bir float)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Aykırı değer tespiti ve etiketleme (-1: aykırı, 1: normal)
# fit_predict hem modeli eğitir hem de etiketleri döndürür
y_pred_lof = lof.fit_predict(X)

# Aykırı olarak etiketlenen noktaları bulma
outliers_indices = np.where(y_pred_lof == -1)[0]
outliers = X[outliers_indices]
inliers = X[y_pred_lof == 1]

print(f"Tespit edilen aykırı değer sayısı: {len(outliers)}")

# Görselleştirme (isteğe bağlı)
plt.figure(figsize=(8, 6))
plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', s=50, label='Normal (Inlier)')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', s=50, label='Aykırı (Outlier)')
plt.title('Local Outlier Factor (LOF) ile Anomali Tespiti')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.legend()
plt.show()

# Not: LOF'un döndürdüğü negatif_outlier_factor_ skoru da kullanılabilir.
# Bu skor ne kadar küçükse (daha negatifse), noktanın aykırı olma olasılığı o kadar yüksektir.
# scores = lof.negative_outlier_factor_
```

### 7. Avantajlar ve Dezavantajlar

#### Avantajlar
*   **Basit ve Sezgisel**: Anlaşılması ve uygulanması kolay bir algoritmadır.
*   **Hızlı Eğitim**: Geleneksel anlamda bir eğitim süreci yoktur, sadece veriyi depolar (O(1)).
*   **Doğrusal Olmayan Sınırlar**: Karmaşık ve doğrusal olmayan karar sınırlarını öğrenebilir.
*   **Parametrik Olmayan**: Verinin altta yatan dağılımı hakkında varsayım yapmaz. Bu, çeşitli veri yapılarına uyum sağlamasını sağlar.
*   **Çok Yönlü**: Hem sınıflandırma hem de regresyon problemleri için kullanılabilir. Anomali tespiti gibi görevlere de uyarlanabilir.
*   **Az Hiperparametre**: Ayarlanması gereken az sayıda temel hiperparametre (K, metrik, ağırlık) vardır.

#### Dezavantajlar
*   **Yavaş Tahmin**: Büyük veri setlerinde (n büyük) her tahmin için tüm veri setini taraması gerektiğinden (O(nd)) yavaştır. Optimizasyonlar (KD/Ball Tree) bunu iyileştirse de temel sorun devam edebilir.
*   **Bellek Yoğun**: Tüm eğitim verisini hafızada tutması gerekir (O(nd)).
*   **Boyutsallık Laneti**: Yüksek boyutlu verilerde (d büyük) performansı önemli ölçüde düşer, uzaklık metrikleri anlamını yitirebilir.
*   **Ölçeklendirme Gerekliliği**: Özelliklerin ölçeklerine çok duyarlıdır, ölçeklendirme genellikle zorunludur.
*   **Gürültüye Duyarlılık**: Özellikle küçük K değerlerinde gürültülü verilere ve aykırı değerlere karşı hassastır.
*   **İlgisiz Özellikler**: Çok sayıda ilgisiz veya alakasız özellik içeren veri setlerinde performansı düşebilir, çünkü bu özellikler uzaklık hesaplamalarını bozar.
*   **Eksik Veriler**: Eksik (NaN) değerlerle doğrudan başa çıkamaz, ön işleme (doldurma veya çıkarma) gerektirir.
*   **Yorumlanabilirlik**: Özellikle yüksek boyutlarda veya karmaşık sınırlarda, modelin neden belirli bir tahminde bulunduğunu yorumlamak zordur (Doğrusal modeller veya karar ağaçları kadar yorumlanabilir değildir).

### 8. KNN ve Doğrusal Regresyon Arasındaki Farklar

| Özellik                     | KNN (K-Nearest Neighbors)                     | Doğrusal Regresyon (Linear Regression)          |
| :-------------------------- | :-------------------------------------------- | :---------------------------------------------- |
| **Model Tipi**              | Parametrik olmayan (Non-parametric)           | Parametrik (Parametric)                         |
| **Temel Fikir**             | Benzer girdiler benzer çıktılar üretir        | Girdiler ve çıktılar arasında doğrusal ilişki   |
| **Öğrenme Yaklaşımı**       | Örnek tabanlı (Instance-based), Tembel (Lazy) | Model tabanlı (Model-based), İstekli (Eager)    |
| **Eğitim Süreci**           | Veri depolama (Çok hızlı, O(1))               | Parametre (katsayılar) optimizasyonu (O(nd²))   |
| **Tahmin Süreci**           | Komşu arama ve oylama/ortalama (Yavaş, O(nd)) | Matematiksel formül uygulama (Çok hızlı, O(d)) |
| **Bellek Gereksinimi**      | Yüksek (Tüm eğitim verisi)                    | Düşük (Sadece model parametreleri)              |
| **Varsayımlar**             | Az (Yerellik varsayımı)                       | Çok (Doğrusallık, bağımsızlık, normallik vb.)   |
| **Doğrusal Olmayan İlişkiler**| Doğal olarak modelleyebilir                   | Ek özellik mühendisliği/dönüşümler gerektirir |
| **Yorumlanabilirlik**       | Düşük (Hangi komşuların etkili olduğu belirsiz)| Yüksek (Katsayılar özellik önemini gösterir)    |
| **Büyük Veri Setleri**      | Tahmin süresi ve bellek sorunları yaşar       | Genellikle daha verimli çalışır                 |
| **Özellik Ölçeklendirme**   | Genellikle **gerekli**                        | Genellikle **önerilir** (özellikle regülarizasyonla) |
| **Kullanım Alanı**          | Sınıflandırma, Regresyon, Anomali Tespiti     | Regresyon, Nedensellik analizi (dikkatle)       |
| **Matematiksel Temel**      | Uzaklık metrikleri                            | En Küçük Kareler (Least Squares) optimizasyonu  |

### 9. Sorulabilecek Sorular

1.  **Soru**: KNN algoritmasında K değerinin seçimi neden önemlidir ve nasıl belirlenir?
    **Cevap**:
    K değeri, modelin karmaşıklığını ve genelleme yeteneğini kontrol eden en önemli hiperparametredir.
    *   **Küçük K**: Modeli gürültüye hassas yapar ve **aşırı uyuma (overfitting)** yol açabilir. Karar sınırları çok girintili çıkıntılı olur.
    *   **Büyük K**: Modeli aşırı basitleştirir ve **yetersiz uyuma (underfitting)** yol açabilir. Karar sınırları çok düzgünleşir ve detayları kaçırır.
    *   **Belirlenmesi**: En iyi K değeri genellikle **çapraz doğrulama (cross-validation)** ile bulunur. Farklı K değerleri denenir ve doğrulama seti üzerinde en iyi performansı (örn. doğruluk, F1 skoru, RMSE) veren K seçilir. Sınıflandırmada oy eşitliğini önlemek için genellikle **tek sayı** tercih edilir. √n (n: örnek sayısı) iyi bir başlangıç noktası olabilir, ancak optimal olması garanti değildir.

2.  **Soru**: KNN algoritmasında "curse of dimensionality" (boyutsallık laneti) problemi nedir ve bu problemi nasıl ele alabilirsiniz?
    **Cevap**:
    Boyutsallık laneti, özellik (boyut) sayısı arttıkça ortaya çıkan çeşitli sorunları ifade eder:
    *   **Veri Seyrekliği**: Yüksek boyutlu uzayda noktalar arasındaki ortalama mesafe artar, noktalar birbirinden "uzaklaşır". Bu durum, "en yakın" komşu kavramını daha az anlamlı hale getirir.
    *   **Uzaklık Metriklerinin Anlamsızlaşması**: Yüksek boyutlarda farklı noktalara olan uzaklıklar birbirine yakınsama eğilimi gösterebilir, bu da komşuları ayırt etmeyi zorlaştırır.
    *   **Hesaplama Maliyeti**: Hem uzaklık hesaplama süresi (O(d)) hem de optimizasyon yapılarının (KD-Tree gibi) etkinliği boyut sayısı arttıkça azalır.
    *   **Gürültü Etkisi**: İlgisiz veya gürültülü özelliklerin sayısı arttıkça, uzaklık hesaplamaları üzerindeki olumsuz etkileri de artar.

    **Ele Alma Yöntemleri**:
    1.  **Özellik Seçimi (Feature Selection)**: En bilgilendirici özellikleri seçip, ilgisiz veya gereksiz olanları çıkarmak.
    2.  **Boyut İndirgeme (Dimensionality Reduction)**: PCA, UMAP gibi tekniklerle orijinal özellikleri daha düşük boyutlu yeni bir uzaya yansıtmak.
    3.  **Özellik Mühendisliği (Feature Engineering)**: Mevcut özelliklerden daha anlamlı yeni özellikler türetmek.
    4.  **Farklı Uzaklık Metrikleri**: Yüksek boyutlarda daha iyi çalıştığı bilinen metrikleri (örn. Kosinüs Benzerliği) denemek.
    5.  **Yerel Yöntemler**: Manifold learning gibi verinin yerel yapısını daha iyi yakalayan yöntemleri düşünmek.

3.  **Soru**: KNN ve Doğrusal Regresyon algoritmalarının temel çalışma prensipleri ve uygulama alanları açısından nasıl karşılaştırırsınız?
    **Cevap**:
    *   **KNN**:
        *   **Prensip**: Örnek tabanlı, tembel öğrenme. Yeni bir noktayı tahmin etmek için eğitim setindeki en yakın K komşusuna bakar (sınıflandırmada çoğunluk oyu, regresyonda ortalama). Veri dağılımı hakkında varsayım yapmaz (parametrik olmayan).
        *   **Uygulama**: Doğrusal olmayan ilişkilerin olduğu, yorumlanabilirliğin ikinci planda olduğu, küçük-orta ölçekli veri setlerinde sınıflandırma ve regresyon. Anomali tespiti.
    *   **Doğrusal Regresyon**:
        *   **Prensip**: Model tabanlı, istekli öğrenme. Bağımlı ve bağımsız değişkenler arasında doğrusal bir ilişki olduğunu varsayar ve bu ilişkiyi en iyi temsil eden doğruyu (veya hiper-düzlemi) bulmaya çalışır (parametrik).
        *   **Uygulama**: Değişkenler arasında doğrusal bir ilişki beklendiğinde, model yorumlanabilirliği önemli olduğunda, büyük veri setlerinde regresyon problemleri. Nedensellik analizi (dikkatli yorumlanmalı).

    **Temel Farklar**: KNN esnek ve varsayımsızdır ancak yavaş ve bellek yoğundur. Doğrusal Regresyon hızlı, yorumlanabilir ve bellek verimlidir ancak güçlü varsayımlara dayanır ve sadece doğrusal ilişkileri modelleyebilir (ekstra çaba olmadan).

4.  **Soru**: KNN algoritmasında uzaklık metriklerinin seçimi model performansını nasıl etkiler? Hangi durumlarda hangi uzaklık metriğini tercih edersiniz?
    **Cevap**:
    Uzaklık metriği, noktalar arasındaki "yakınlığı" nasıl ölçtüğümüzü tanımlar ve bu, hangi noktaların komşu olarak seçileceğini doğrudan etkiler. Dolayısıyla metrik seçimi model performansı için kritiktir.
    *   **Öklid Uzaklığı (p=2)**: En yaygın kullanılan, genel amaçlı metriktir. Noktalar arasındaki düz çizgi mesafesini ölçer. Sürekli özelliklerin olduğu çoğu problem için iyi bir başlangıç noktasıdır.
    *   **Manhattan Uzaklığı (p=1)**: Özelliklerin katkılarının ayrı ayrı toplandığı (şehir bloğu mesafesi) metriktir. Yüksek boyutlu verilerde veya özelliklerin birbirinden bağımsız olduğu düşünülen durumlarda bazen Öklid'den daha iyi çalışabilir. Aykırı değerlere Öklid'e göre biraz daha az duyarlı olabilir. Grid benzeri yapılarda anlamlıdır.
    *   **Minkowski Uzaklığı (genel p)**: p parametresi ayarlanarak farklı davranışlar elde edilebilir. p'nin optimize edilmesi gerekebilir.
    *   **Kosinüs Benzerliği/Uzaklığı**: Vektörlerin yönelimine odaklanır, büyüklüğüne değil. Özellikle metin verileri (TF-IDF vektörleri gibi) veya yüksek boyutlu seyrek veriler için tercih edilir.
    *   **Hamming Uzaklığı**: Kategorik (nominal) özellikler için kullanılır. İki vektör arasındaki farklı olan pozisyon sayısını ölçer.
    *   **Mahalanobis Uzaklığı**: Özellikler arasındaki korelasyonu dikkate alır ve verinin kovaryans matrisini kullanır. Özellikler ilişkili olduğunda ve farklı ölçeklerde olduğunda faydalı olabilir.

    **Tercih**: Genellikle Öklid ile başlanır. Verinin türüne (sürekli, kategorik, metin), boyut sayısına ve özellikler arasındaki ilişkiye göre diğer metrikler denenebilir. En iyi metrik genellikle çapraz doğrulama ile deneysel olarak bulunur.

5.  **Soru**: KNN algoritmasında ağırlıklı ve ağırlıksız oylama arasındaki fark nedir? Hangi durumlarda ağırlıklı oylama daha avantajlı olur?
    **Cevap**:
    *   **Ağırlıksız Oylama (`weights='uniform'`)**: K en yakın komşunun her birinin sınıflandırma veya regresyon tahminine katkısı eşittir. Sınıflandırmada her komşu bir oy verir, regresyonda değerlerinin basit ortalaması alınır.
    *   **Ağırlıklı Oylama (`weights='distance'`)**: Her komşunun katkısı, test noktasına olan uzaklığına göre belirlenir. Yakın komşular daha fazla ağırlığa (daha fazla etkiye), uzak komşular daha az ağırlığa sahip olur. Ağırlık genellikle uzaklığın tersi ($1/d$) veya tersinin karesi ($1/d^2$) ile orantılıdır.

    **Ağırlıklı Oylamanın Avantajlı Olduğu Durumlar**:
    1.  **Sınıf Sınırlarına Yakın Noktalar**: Test noktası farklı sınıflara ait komşuların olduğu bir sınıra yakınsa, ağırlıklı oylama daha yakın olan komşuların sınıfına daha fazla önem vererek daha doğru bir karar verilmesine yardımcı olabilir.
    2.  **Seyrek Veri Bölgeleri**: Bazı bölgelerde veri yoğunluğu düşükse, uzaktaki komşuların etkisi ağırlıklı oylama ile azaltılabilir.
    3.  **Gürültülü Veriler**: Potansiyel olarak gürültülü veya yanlış etiketlenmiş bir komşu uzaktaysa, ağırlıklı oylama onun etkisini sınırlar.
    4.  **Büyük K Değerleri**: K değeri büyük seçildiğinde, ağırlıklı oylama uzaktaki çok sayıda komşunun etkisini azaltarak modelin yerel yapıya daha duyarlı kalmasını sağlayabilir.

    Genel olarak, ağırlıklı oylama, komşuların hepsinin eşit derecede önemli olmadığı durumlarda daha sağlam ve potansiyel olarak daha doğru tahminler sunabilir. Ancak her zaman daha iyi sonuç vereceği garanti değildir ve çapraz doğrulama ile test edilmelidir.