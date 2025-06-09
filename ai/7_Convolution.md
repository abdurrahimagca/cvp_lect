# Evrişimsel Sinir Ağları (CNN) - Detaylı Ders Notları

Evrişimsel Sinir Ağları (Convolutional Neural Networks - CNN veya ConvNet), özellikle görüntü verileri gibi grid yapısındaki verileri işlemek üzere tasarlanmış özel bir derin öğrenme modelidir. Geleneksel sinir ağlarının aksine, CNN'ler mekansal hiyerarşileri (görüntülerdeki kenarlar, köşeler, dokular gibi) otomatik olarak ve verimli bir şekilde öğrenir.

## 1. CNN'in Temel Yapı Taşları

Bir CNN, genellikle üç ana tür katmanın birleşiminden oluşur: Evrişim (Convolution), Havuzlama (Pooling) ve Tam Bağlantılı (Fully Connected) katmanlar.

### a. Evrişim Katmanı (Convolutional Layer)

Bu katman, CNN'in temelini oluşturur ve özellik çıkarımından (feature extraction) sorumludur.

-   **Filtre/Kernel**: Küçük bir matristir (örn. 3x3, 5x5). Bu filtre, girdi görüntüsü üzerinde kaydırılarak, her konumda eleman bazında çarpım ve toplama işlemi (dot product) yapar. Filtrenin amacı, kenarlar, renk geçişleri, dokular gibi belirli özellikleri tespit etmektir. Bir evrişim katmanı, farklı özellikleri tespit etmek için birçok filtre öğrenir.
-   **Stride (Adım Kaydırma)**: Filtrenin girdi üzerinde gezerken her adımda ne kadar kayacağını belirler. Stride=1, filtrenin birer piksel kayması anlamına gelir. Daha büyük stride değerleri, çıktı boyutunu küçültür.
-   **Padding (Dolgu)**: Girdi matrisinin kenarlarına genellikle sıfırlardan oluşan ekstra pikseller ekleme işlemidir. Amacı:
    1.  Kenarlardaki piksellerin de merkezdeki pikseller kadar işleme dahil olmasını sağlamak.
    2.  Evrişim işlemi sonrası çıktı boyutunun küçülmesini kontrol etmek veya engellemek ("same" padding).
-   **Özellik Haritası (Feature Map)**: Bir filtrenin girdi üzerinde gezinmesi sonucu oluşan çıktı matrisidir. Her bir özellik haritası, girdinin belirli bir özelliğini (örneğin, dikey kenarlar) vurgular.

Bir evrişim katmanının çıktı boyutu şu formülle hesaplanır:
$$O = \lfloor \frac{I + 2P - K}{S} + 1 \rfloor$$
-   $O$: Çıktı boyutu (yükseklik veya genişlik)
-   $I$: Girdi boyutu
-   $P$: Padding miktarı
-   $K$: Kernel boyutu
-   $S$: Stride miktarı

Matematiksel olarak 2D evrişim (bir girdi kanalı için), $(f * g)(i, j) = \sum_{m}\sum_{n} f(m, n) g(i-m, j-n)$ olarak ifade edilir. Pratikte ise "çapraz korelasyon" (cross-correlation) operasyonu kullanılır ki bu da filtrenin döndürülmemesi anlamına gelir: $(f * g)(i, j) = \sum_{m}\sum_{n} f(i+m, j+n) g(m, n)$. Derin öğrenme kütüphaneleri "convolution" dediklerinde aslında bu ikinci işlemi yaparlar.

#### CNN'in Verimliliği: Parametre Paylaşımı ve Yerel Bağlantı

Geleneksel bir sinir ağının (Fully Connected) aksine, CNN'ler iki önemli prensip sayesinde çok daha verimlidir:
1.  **Yerel Bağlantı (Sparse Connectivity)**: Bir çıktı nöronu, girdinin tamamına değil, sadece filtrenin o an üzerinde olduğu küçük bir alana (receptive field) bağlıdır. Bu, görüntüdeki mekansal yakınlığın önemli olduğu varsayımına dayanır.
2.  **Parametre Paylaşımı (Parameter Sharing)**: Bir özellik haritası oluşturulurken kullanılan filtre (kernel) tektir. Yani aynı ağırlık seti, görüntünün tamamı üzerinde kaydırılarak kullanılır. Bu, modelin görüntünün bir yerinde öğrendiği bir özelliği (örneğin dikey bir kenar) başka bir yerinde de tanımasını sağlar (konumdan bağımsızlık - translation invariance). Bu sayede öğrenilmesi gereken parametre sayısı drastik şekilde azalır.

### b. Aktivasyon Katmanı

Genellikle her evrişim katmanından sonra bir aktivasyon fonksiyonu uygulanır. CNN'lerde neredeyse her zaman **ReLU (Rectified Linear Unit)** kullanılır.
$$f(x) = \max(0, x)$$
ReLU, negatif değerleri sıfırlayarak modele non-lineerlik katar ve eğitim sürecini hızlandırır.

### c. Havuzlama Katmanı (Pooling Layer)

Havuzlama katmanının temel amacı, özellik haritalarının boyutunu (mekansal boyutlarını) azaltmaktır. Bunun iki önemli faydası vardır:
1.  **Hesaplama Yükünü Azaltma**: Daha küçük özellik haritaları, daha az parametre ve daha hızlı hesaplama demektir.
2.  **Özelliklerin Konumuna Karşı Dayanıklılık (Translational Invariance)**: Özellik haritasındaki küçük kaymalara karşı modeli daha duyarsız hale getirir. Özelliğin tam olarak nerede olduğundan çok, o bölgede var olup olmadığına odaklanılmasını sağlar.

En yaygın kullanılan iki havuzlama türü:
-   **Max Pooling**: Bir pencere (örn. 2x2) içindeki en büyük değeri alır. Genellikle en belirgin özelliği korumada daha etkilidir.
-   **Average Pooling**: Penceredeki değerlerin ortalamasını alır.

### d. Tam Bağlantılı Katman (Fully Connected - FC Layer)

CNN mimarisinin sonlarına doğru, evrişim ve havuzlama katmanlarından çıkan yüksek seviyeli özellik haritaları **düzleştirilir (flattening)** ve bir veya daha fazla tam bağlantılı katmana girdi olarak verilir. Bu katmanlar, standart bir yapay sinir ağı gibi çalışır. Amaçları, öğrenilen bu özelliklere dayanarak nihai sınıflandırma veya regresyon görevini gerçekleştirmektir.

**Düzleştirme (Flattening) Örneği**: Eğer son havuzlama katmanının çıktısı `5x5` boyutunda `64` farklı özellik haritası ise, bu `(64, 5, 5)` boyutlu bir tensördür. Düzleştirme işlemi bunu `64 * 5 * 5 = 1600` elemanlı tek boyutlu bir vektöre dönüştürür. Bu vektör, ilk tam bağlantılı katmanın girdisi olur.

## 2. Tipik Bir CNN Mimarisi

Genel bir CNN mimarisi şu şekilde özetlenebilir:

`GİRDİ -> [[CONV -> ReLU] * N -> POOL?] * M -> [FC -> ReLU] * K -> FC (ÇIKIŞ)`

-   `[CONV -> ReLU] * N`: N kez tekrarlanan Evrişim ve ReLU katmanı bloğu.
-   `POOL?`: İsteğe bağlı Havuzlama katmanı.
-   `* M`: Bu blokların M kez tekrarlanması.
-   Düzleştirme (Flattening) işleminden sonra K adet Tam Bağlantılı katman.
-   Son olarak sınıf skorlarını üreten bir çıkış katmanı.

### Soru & Cevap

**S: Basit bir CNN mimarisinin parametre sayısı nasıl hesaplanır? Bu, sınavlar için klasik bir sorudur.**

C: Örnek bir mimari üzerinden gidelim:
- Girdi: 32x32x3 (RGB görüntü)
1.  **CONV1**: 10 tane 5x5 filtre, stride=1, padding=0.
2.  **POOL1**: 2x2 max pooling, stride=2.
3.  **FC1**: 50 nöron.
4.  **FC2 (Çıkış)**: 10 nöron (10 sınıf için).

Hesaplama:
1.  **CONV1 Parametreleri**:
    - Her bir filtrenin boyutu: 5x5x3 (Genişlik x Yükseklik x Girdi Kanalı Sayısı).
    - Her filtrenin bir de bias terimi vardır.
    - Parametre sayısı = (Filtre Boyutu + Bias) * Filtre Sayısı
    - Parametre sayısı = ((5 * 5 * 3) + 1) * 10 = (75 + 1) * 10 = **760 parametre**.
    - *Not: POOL katmanlarının öğrenilecek parametresi yoktur.*

2.  **FC1'e Girmeden Önceki Boyut**:
    - CONV1 sonrası çıktı boyutu: $(32-5)/1 + 1 = 28$. Yani `28x28x10`.
    - POOL1 sonrası çıktı boyutu: `28/2 = 14`. Yani `14x14x10`.
    - Düzleştirilmiş (Flattened) vektör boyutu: `14 * 14 * 10 = 1960`. Bu, FC1'in girdi sayısıdır.

3.  **FC1 Parametreleri**:
    - Girdi sayısı: 1960. Nöron sayısı: 50.
    - Ağırlık matrisi boyutu: 1960 x 50.
    - Bias sayısı: 50.
    - Parametre sayısı = (1960 * 50) + 50 = 98,000 + 50 = **98,050 parametre**.

4.  **FC2 (Çıkış) Parametreleri**:
    - Girdi sayısı: 50 (FC1'in çıktısı). Nöron sayısı: 10.
    - Ağırlık matrisi boyutu: 50 x 10.
    - Bias sayısı: 10.
    - Parametre sayısı = (50 * 10) + 10 = 500 + 10 = **510 parametre**.

**Toplam Parametre Sayısı** = 760 (CONV1) + 98,050 (FC1) + 510 (FC2) = **99,320**.

## 3. Önemli CNN Mimarileri

-   **LeNet-5 (1998)**: İlk başarılı CNN mimarilerinden biridir. Posta kodları gibi el yazısı rakamları tanımak için geliştirilmiştir.
-   **AlexNet (2012)**: ImageNet yarışmasını kazanarak derin öğrenme devrimini başlatan mimaridir. LeNet'e göre çok daha derin ve geniştir. ReLU ve Dropout gibi modern teknikleri popülerleştirmiştir.
-   **VGGNet (2014)**: Sadece 3x3'lük küçük evrişim filtreleri kullanarak çok derin ağlar (16-19 katman) oluşturmanın etkili olduğunu göstermiştir. Mimarisi basit ve homojendir.
-   **GoogLeNet / Inception (2014)**: "Inception modülü" adı verilen ve aynı katmanda farklı boyutlu filtreleri (1x1, 3x3, 5x5) paralel olarak kullanan bir yapı sunmuştur. Bu, ağın farklı ölçeklerdeki özellikleri aynı anda öğrenmesini sağlar. "1x1 evrişim" katmanlarını, boyut azaltma (dimensionality reduction) amacıyla filtre sayısını düşürmek için akıllıca kullanmıştır.
-   **ResNet (2015)**: "Artık bağlantılar" (residual connections) veya "kısayol bağlantıları" (shortcut connections) konseptini tanıtmıştır. Bu yapı, bir katman bloğunun, girdiyi ($x$) doğrudan çıktıya eklemesini sağlar: $H(x) = F(x) + x$. Ağ, $H(x)$'i (temeldeki eşleşmeyi) öğrenmek yerine, artık fonksiyon olan $F(x) = H(x) - x$'i öğrenmeye çalışır. $F(x)$'i sıfıra yaklaştırmak, kimlik dönüşümünü (identity mapping) öğrenmekten daha kolay olduğu için çok derin ağlarda bile gradyanların kaybolmadan (vanishing gradient) akmasını sağlayarak 150'den fazla katmana sahip ağların eğitilmesini mümkün kılmıştır.

## 4. İleri Düzey Teknikler

-   **Transfer Learning (Transfer Öğrenme)**: Milyonlarca görüntüyle (örn. ImageNet) önceden eğitilmiş bir CNN modelini alıp, kendi problemimize (genellikle daha küçük bir veri setiyle) uyarlamaktır.
    -   **Feature Extraction**: Önceden eğitilmiş modelin evrişimsel katmanları bir "özellik çıkarıcı" olarak kullanılır, sadece sonundaki tam bağlantılı katmanlar yeniden eğitilir.
    -   **Fine-Tuning**: Önceden eğitilmiş modelin tüm ağırlıkları (veya bir kısmının) "çözülür" ve yeni veri seti üzerinde çok düşük bir öğrenme oranıyla tekrar eğitilir. Modelin tamamı yeni göreve adapte olur.
-   **Data Augmentation (Veri Artırma)**: Mevcut eğitim verilerinden yapay olarak yeni örnekler oluşturma tekniğidir. Görüntüleri rastgele döndürme, kırpma, çevirme, renk tonunu değiştirme gibi işlemlerle yapılır. Bu, overfitting'i azaltır ve modelin genelleme yeteneğini artırır.
-   **Batch Normalization**: Her katmanın girdisini, o anki mini-batch'in ortalama ve standart sapmasını kullanarak normalleştirir. Eğitimi hızlandırır, stabilize eder ve bir miktar düzenlileştirme (regularization) etkisi sağlar.

### Soru & Cevap

**S: Transfer öğrenmede "Feature Extraction" ve "Fine-Tuning" arasındaki fark nedir? Ne zaman hangisini tercih etmeliyiz?**

C: Her ikisi de önceden eğitilmiş bir modeli kullanır, ancak farklı stratejilere sahiptir.
-   **Feature Extraction (Özellik Çıkarımı)**: Önceden eğitilmiş modelin evrişimsel katmanları (base) dondurulur, yani ağırlıkları eğitim sırasında güncellenmez. Sadece en sona eklediğimiz yeni sınıflandırıcı (classifier/head) katmanlar eğitilir.
    -   **Ne zaman kullanılır?**:
        1.  Kendi veri setiniz küçükse.
        2.  Kendi veri setiniz, orijinal modelin eğitildiği veri setine (örn. ImageNet) benziyorsa.
    -   **Avantajı**: Çok daha hızlıdır ve daha az veri gerektirir. Overfitting riski daha düşüktür.

-   **Fine-Tuning (İnce Ayar)**: Önceden eğitilmiş modelin tüm ağırlıkları (veya bir kısmının) "çözülür" ve yeni veri seti üzerinde çok düşük bir öğrenme oranıyla tekrar eğitilir. Modelin tamamı yeni göreve adapte olur.
    -   **Ne zaman kullanılır?**:
        1.  Kendi veri setiniz büyükse.
        2.  Yeterli hesaplama gücünüz varsa.
    -   **Avantajı**: Modelin yeni göreve daha iyi adapte olmasını sağlayarak potansiyel olarak daha yüksek bir performans sunar.

**S: PyTorch'ta Veri Artırma (Data Augmentation) nasıl uygulanır?**

C: `torchvision.transforms` modülü ile çok kolay bir şekilde uygulanır. Genellikle eğitim verisi için bir dönüşüm zinciri (`Compose`) oluşturulur. Test/Validasyon verisine genellikle sadece normalizasyon uygulanır.

```python
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Eğitim verisi için veri artırma ve normalizasyon dönüşümleri
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # Rastgele boyutta kırp ve 224x224'e yeniden boyutlandır
    transforms.RandomHorizontalFlip(),      # %50 ihtimalle yatayda çevir
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Parlaklık ve kontrastı rastgele değiştir
    transforms.ToTensor(),                  # Görüntüyü PyTorch tensörüne çevir
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet ortalama ve std. sapması ile normalleştir
                         std=[0.229, 0.224, 0.225])
])

# Validasyon/Test verisi için sadece temel dönüşümler
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset ve DataLoader'ları oluştururken bu dönüşümleri kullanırız
train_dataset = ImageFolder(root='path/to/train_data', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```
Bu sayede her `epoch`'ta, model eğitim verisinin biraz değiştirilmiş versiyonlarını görerek daha iyi genelleme yapmayı öğrenir.
