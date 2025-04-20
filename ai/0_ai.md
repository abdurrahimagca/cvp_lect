# Bilgisayarlı Görme ve Makine Öğrenmesi: Giriş ve Uygulama

Bu doküman, `notes_md/0_Intro.md` dosyasındaki teorik bilgileri ve `Uygulama/0_hello.ipynb` dosyasındaki pratik uygulamayı birleştirerek daha derinlemesine bir anlayış sunmayı amaçlamaktadır.

## 1. Teorik Temeller (Özet - `0_Intro.md`)

* **Bilgisayarlı Görme (BG):** Makinelerin görsel dünyayı "görmesini" ve yorumlamasını sağlayan alan. İbnü'l Heysem'in erken dönem görme teorilerinden, modern RGB temsillerine ve dijital görüntülerin zorluklarına (aydınlatma, pozlama, sınıf içi çeşitlilik vb.) kadar uzanır.
* **Makine Öğrenmesi (MÖ):** Bilgisayarların verilerden öğrenmesini sağlayan yöntemler bütünü.
  * **Denetimli Öğrenme:** Etiketli verilerle model eğitimi (örn. Sınıflandırma, Regresyon).
  * **Denetimsiz Öğrenme:** Etiketsiz verilerde desen keşfi (örn. Kümeleme).
  * **Pekiştirmeli Öğrenme:** Ödül/ceza mekanizması ile öğrenme.
* **Temel Problemler:**
  * **Sınıflandırma:** Veriyi kategorilere ayırma (örn. Spam tespiti, görüntü tanıma).
  * **Regresyon:** Sürekli bir değeri tahmin etme (örn. Fiyat tahmini).
  * **Kümeleme:** Benzer veri noktalarını gruplama (örn. Müşteri segmentasyonu).
* **Model Değerlendirme:** Accuracy, Precision/Recall, F1, ROC/AUC, MSE/RMSE/MAE gibi metrikler ve Cross-Validation teknikleri. Overfitting/Underfitting sorunları ve Bias-Variance dengesi.
* **Veri Hazırlama:** Temizleme, Feature Engineering ve Augmentation.
* **Araçlar:** Python (NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch).

## 2. Pratik Uygulama: MNIST ile KNN Sınıflandırması (`0_hello.ipynb`)

`0_hello.ipynb` not defteri, Makine Öğrenmesinin temel problemlerinden biri olan **Sınıflandırma** üzerine pratik bir örnek sunar. Bu örnekte, **MNIST** veri seti kullanılarak el yazısı rakamları tanımak için **K-En Yakın Komşu (K-Nearest Neighbors - KNN)** algoritması uygulanmıştır.

### Adımlar

1. **Kütüphanelerin İçe Aktarılması:** `idx2numpy`, `numpy`, `collections.Counter`, `matplotlib.pyplot` gibi gerekli kütüphaneler yüklenir.
2. **Veri Setinin Yüklenmesi:** MNIST veri seti (`train-images-idx3-ubyte` ve `train-labels-idx1-ubyte`) `idx2numpy` ile yüklenir. `X` değişkeni görüntüleri (piksel değerleri), `y` değişkeni ise etiketleri (0-9 arası rakamlar) içerir.
3. **Veri Görselleştirme:** `matplotlib` kullanılarak veri setinden örnek görüntüler ve etiketleri görselleştirilir. Bu, veri setini anlamak için önemli bir adımdır.
4. **Uzaklık Fonksiyonları:**
    * `l1_distance` (Manhattan Mesafesi): İki görüntü arasındaki piksel farklarının mutlak değerlerinin toplamı.
    * `l2_distance` (Öklid Mesafesi): İki görüntü arasındaki piksel farklarının karelerinin toplamının karekökü.
    Bu fonksiyonlar, görüntülerin birbirine ne kadar "benzer" veya "uzak" olduğunu ölçmek için kullanılır.
5. **KNN Mantığı (Adım Adım):**
    * Bir sorgu görüntüsü (`query_img`) seçilir.
    * Bu sorgu görüntüsünün eğitim setindeki (`X`) diğer tüm görüntülere olan uzaklıkları hesaplanır (`uzakliklar`).
    * Uzaklıklar küçükten büyüğe sıralanır (`np.argsort`).
    * En küçük uzaklığa sahip ilk `K` adet komşu bulunur (`en_yakin_komsular`). (Not: İlk eleman (indeks 0) genellikle sorgu görüntüsünün kendisidir, bu yüzden `[1:K+1]` kullanılır).
    * Bu `K` komşunun sınıfları (`y` etiketleri) bulunur (`komsu_siniflari`).
    * Komşu sınıfları arasında en sık görünen sınıf, sorgu görüntüsünün tahmini sınıfı olarak belirlenir (`Counter(...).most_common()[0][0]`).
6. **KNN Sınıfının Oluşturulması:** Yukarıdaki mantık, `KNN` adında bir sınıf içinde yeniden kullanılabilir hale getirilir.
    * `__init__`: Uzaklık fonksiyonu (`distance_fun`) ve komşu sayısı (`K`) ile başlatılır.
    * `fit`: Eğitim verisini (`X`, `y`) alır ve saklar. KNN'de "eğitim" aslında sadece veriyi saklamaktır.
    * `predict`: Yeni bir sorgu görüntüsü için yukarıda açıklanan KNN adımlarını uygular ve tahmin edilen sınıfı döndürür.
7. **Modelin Kullanımı:** `KNN` sınıfından bir nesne oluşturulur (`knn = KNN(l2_distance, 7)`), `fit` ile eğitilir ve bir örnek görüntü için `predict` ile tahmin yapılır.

## 3. Teorik ve Pratik Bağlantılar

* **Denetimli Öğrenme:** KNN, etiketli veri (MNIST görüntüleri ve rakam etiketleri) kullandığı için bir denetimli öğrenme algoritmasıdır.
* **Sınıflandırma Problemi:** Amaç, bir görüntüyü önceden tanımlanmış 10 sınıftan (0-9 rakamları) birine atamaktır.
* **Özellik (Feature):** Bu örnekte, her bir piksel değeri bir özellik olarak düşünülebilir. Görüntünün tamamı (28x28 piksel = 784 özellik) girdi olarak kullanılır.
* **Uzaklık Metrikleri:** `l1_distance` ve `l2_distance`, özellik uzayında noktalar (görüntüler) arasındaki mesafeyi ölçer. Farklı metrikler farklı sonuçlar verebilir.
* **"K" Değeri (Hiperparametre):** `K` sayısı, modelin performansını etkileyen, dışarıdan ayarlanan bir parametredir (hiperparametre). Farklı `K` değerleri denenerek en iyi sonuç veren seçilebilir.
* **Model Değerlendirme (Eksiklik):** Notebook'ta modelin genel performansı (örn. test seti üzerindeki accuracy) ölçülmemiştir. Gerçek bir uygulamada bu adım kritiktir. `0_Intro.md`'de bahsedilen metrikler burada kullanılabilir.

## 4. Sorular ve Düşünme Egzersizleri

1. `0_Intro.md`'de bahsedilen Bilgisayarlı Görme zorluklarından (örn. aydınlatma, sınıf içi çeşitlilik) hangileri MNIST veri setinde KNN algoritmasını zorlayabilir? Neden?
2. `0_hello.ipynb`'deki KNN uygulamasında neden `np.argsort` ile bulunan indekslerin `[1:K+1]` aralığı kullanılıyor? `[0:K]` kullanılsa ne olurdu?
3. L1 ve L2 uzaklık metrikleri arasındaki temel fark nedir? Hangi durumlarda biri diğerine tercih edilebilir? Notebook'ta `K=7` için hem L1 hem de L2 ile tahmin yapıp sonuçları karşılaştırın.
4. `K` değerini değiştirmenin (örn. K=1, K=20 yapmanın) tahmin sonuçları üzerindeki etkisi ne olur? Çok küçük veya çok büyük K değerlerinin potansiyel dezavantajları nelerdir? (`0_Intro.md`'deki Overfitting/Underfitting ile ilişkilendirin).
5. Notebook'taki KNN sınıfına, bir test seti üzerinde doğruluk (accuracy) hesaplayan bir `evaluate` metodu ekleyin. (`0_Intro.md`'deki Model Değerlendirme bölümünden yararlanın).
6. MNIST veri setindeki bir görüntüyü hafifçe döndürürseniz veya birkaç pikselini değiştirirseniz, KNN tahmininin değişme olasılığı nedir? Bu durum, algoritmanın hangi özelliğini gösterir?
7. KNN algoritmasının avantajları ve dezavantajları nelerdir? Özellikle çok büyük veri setlerinde veya yüksek boyutlu özellik uzaylarında ne gibi sorunlar yaşanabilir?
8. `0_Intro.md`'de bahsedilen diğer sınıflandırma algoritmalarından (örn. Lojistik Regresyon, Destek Vektör Makineleri) hangileri MNIST problemi için kullanılabilir? KNN ile karşılaştırıldığında potansiyel avantajları/dezavantajları ne olabilir?
