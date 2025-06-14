# Transpoze Evrişim (Transposed Convolution) - Detaylı Anlatım

Bu doküman, transpoze evrişimin (genellikle hatalı bir şekilde dekonvolüsyon olarak da adlandırılır) ne olduğunu, nasıl çalıştığını, uygulama alanlarını ve PyTorch ile nasıl hayata geçirileceğini detaylı bir şekilde açıklamaktadır.

## 1. Giriş: Transpoze Evrişim Nedir?

**Transpoze Evrişim**, standart bir evrişimin (convolution) yaptığı boyut küçültme (downsampling) işleminin tam tersi olarak, bir özellik haritasını (feature map) büyütmek (upsampling) için kullanılan bir katman türüdür. Amacı, bir evrişim işlemi sonucunda kaybedilen uzamsal (spatial) çözünürlüğü geri kazanmaktır.

Normal bir evrişim katmanı, girdisindeki pikselleri komşularıyla birleştirerek daha küçük bir çıktı üretir. Transpoze evrişim ise tam tersine, girdisindeki her bir pikseli daha geniş bir alana "dağıtarak" veya "saçarak" daha büyük bir çıktı oluşturur. Bu işlem, özellikle Kodlayıcı-Kod Çözücü (Encoder-Decoder) mimarilerinde, anlamsal segmentasyon (semantic segmentation) ve görüntü üretme gibi görevlerde kritik bir rol oynar.

Matematiksel olarak, standart bir evrişim işlemi bir matris çarpımı ($y = Wx$) olarak ifade edilebilirse, transpoze evrişim bu matrisin transpozu ($W^T$) ile yapılan bir çarpım olarak düşünülebilir: $x = W^T y$. Bu nedenle "transpoze" evrişim olarak adlandırılır.

## 2. Çalışma Prensibi

Normal evrişim ile transpoze evrişim arasındaki temel fark, uzamsal boyutları nasıl etkiledikleridir.

-   **Normal Evrişim**: `n x n` boyutundaki bir girdiyi `f x f` boyutundaki bir filtre ile işleyerek `(n-f+1) x (n-f+1)` boyutunda bir çıktı üretir (padding ve stride hariç). Boyut küçülür.
-   **Transpoze Evrişim**: `n x n` boyutundaki bir girdiden `(n+f-1) x (n+f-1)` boyutunda bir çıktı üretir. Boyut büyür.

### İşlem Adımları (Kavramsal)

Transpoze evrişimin nasıl çalıştığını anlamanın en kolay yolu, onu bir "öğrenilebilir yukarı örnekleme" (learnable upsampling) olarak düşünmektir. İşlem genel olarak şu adımları içerir:

1.  **Genişletme (Stride ile Boşluk Doldurma)**: Girdi özellik haritasının pikselleri arasına, stride (adım) sayısına bağlı olarak boşluklar (genellikle sıfırlar) eklenir. Örneğin, `stride=2` ise piksellerin arasına birer satır/sütun sıfır eklenir. Bu, özellik haritasını etkili bir şekilde genişletir.
2.  **Padding Ekleme**: Genişletilmiş haritanın etrafına padding eklenir. Transpoze evrişimde padding, standart evrişimin aksine çıktı boyutunu küçültmek yerine, kenar etkilerini kontrol altına almak için kullanılır.
3.  **Standart Evrişim Uygulama**: Bu genişletilmiş ve doldurulmuş harita üzerinde standart bir evrişim işlemi gerçekleştirilir. Filtre (kernel) bu geniş harita üzerinde gezinerek pikselleri toplar ve nihai büyük çıktıyı oluşturur.

Bu süreç, ağın veriyi nasıl büyüteceğini (upsample) öğrenmesini sağlar. Eklenen sıfırlar yerine ne geleceği ve piksellerin nasıl birleştirileceği, geri yayılım sırasında öğrenilen filtre ağırlıkları tarafından belirlenir.

## 3. Uygulama Alanları

Transpoze evrişim, bir sinir ağının uzamsal bilgiyi yeniden yapılandırması veya artırması gereken her yerde kullanılır.

### 1. Kodlayıcı-Kod Çözücü Mimarileri (Autoencoders)

Bu mimariler, bir veriyi daha düşük boyutlu bir gizli uzaya (latent space) sıkıştıran bir **kodlayıcı (encoder)** ve bu gizli uzay temsilinden orijinal veriyi yeniden oluşturan bir **kod çözücüden (decoder)** oluşur.

-   **Kodlayıcı**: Standart `Conv2d` ve `MaxPool2d` katmanları kullanarak girdinin (örneğin bir resmin) uzamsal boyutlarını azaltır ve özellik haritalarının derinliğini artırır.
-   **Kod Çözücü**: Tam tersini yapar. Transpoze evrişim katmanları (`ConvTranspose2d`) kullanarak özellik haritasının uzamsal boyutlarını artırır ve orijinal girdi boyutlarına geri döndürür.

Aşağıda `Uygulama/10_autoencoders.ipynb` dosyasından alınan bir evrişimli oto-kodlayıcı örneği bulunmaktadır:

```python
import torch.nn as nn

class AutoEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Encoder ---
        # Girdi: [N, 1, 28, 28]
        self.encoder = nn.Sequential(
            # Çıktı: [N, 16, 14, 14]
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Çıktı: [N, 32, 7, 7]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Çıktı: [N, 64, 1, 1] (Latent Space)
            nn.Conv2d(32, 64, kernel_size=7)
        )

        # --- Decoder ---
        # Girdi: [N, 64, 1, 1]
        self.decoder = nn.Sequential(
            # Çıktı: [N, 32, 7, 7]
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            # Çıktı: [N, 16, 14, 14]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Çıktı: [N, 1, 28, 28]
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Görüntü pikselleri 0-1 arasında olduğu için Sigmoid
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

### 2. Anlamsal Segmentasyon (Semantic Segmentation)

Bu görevde amaç, bir görüntüdeki her bir pikseli bir sınıfa (örneğin, "araba", "yol", "insan") atamaktır. U-Net gibi popüler mimariler, bir kodlayıcı ile görüntünün özelliklerini çıkarır (boyut küçülür) ve ardından bir kod çözücü ile bu özellik haritalarını orijinal görüntü boyutuna geri büyüterek piksel bazında bir sınıflandırma haritası oluşturur. Bu büyütme aşamasında transpoze evrişim katmanları kullanılır.

### 3. Görüntü Üretme ve Süper Çözünürlük (Image Generation & Super Resolution)

Üretken Çekişmeli Ağlar (GAN'lar) gibi üretici modeller, rastgele bir gürültü vektöründen gerçekçi görüntüler oluşturur. Bu süreç, küçük boyutlu bir vektörden büyük boyutlu bir görüntüye geçişi gerektirir ve bu genellikle transpoze evrişim katmanları ile yapılır. Benzer şekilde, süper çözünürlük modelleri, düşük çözünürlüklü bir görüntüyü alıp detayları yeniden oluşturarak yüksek çözünürlüklü bir versiyonunu üretmek için transpoze evrişim kullanır.

## 4. PyTorch ile Uygulama ve Hiperparametreler

PyTorch'ta transpoze evrişim, `nn.ConvTranspose2d` modülü ile kolayca uygulanabilir.

```python
import torch.nn as nn

conv_transpose = nn.ConvTranspose2d(
    in_channels=64,   # Girdi kanal sayısı (derinlik)
    out_channels=32,  # Çıktı kanal sayısı
    kernel_size=3,    # Filtre boyutu
    stride=2,         # Büyütme faktörü
    padding=1,        # Kenarlara eklenecek dolgu
    output_padding=1  # Çıktı boyutunu ayarlamak için ek dolgu
)
```

### Önemli Hiperparametreler

-   **`stride`**: Bu parametre, yukarı örnekleme (upsampling) faktörünü belirler. `stride=2` genellikle boyutları ikiye katlamak için kullanılır. Standart evrişimde pikselleri atlamak için kullanılırken, burada girdi pikselleri arasına ne kadar boşluk ekleneceğini kontrol eder.
-   **`kernel_size`**: Öğrenilebilir filtrenin boyutudur. Standart evrişimde olduğu gibi, daha büyük filtreler daha fazla parametre anlamına gelir.
-   **`padding`**: Standart evrişimdeki `padding`'den farklı çalışır. Transpoze evrişimde, bu parametre filtrenin ne kadar "dışarıdan" başlayacağını belirler, bu da çıktı boyutunu etkiler. Çıktı boyutu formülü:
    $$ \text{output\_size} = (\text{input\_size} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding} $$
-   **`output_padding`**: Bu parametre, belirli `stride` ve `kernel_size` kombinasyonları için oluşabilecek belirsizlikleri çözmek için kullanılır. Bazen birden fazla çıktı boyutu aynı parametrelerle elde edilebilir. `output_padding`, çıktının kenarlarına fazladan dolgu ekleyerek istediğiniz boyutu elde etmenizi sağlar. Genellikle `stride > 1` olduğunda kullanılır.

## 5. "Checkerboard" Artefaktları ve Çözümleri

Transpoze evrişimin en bilinen sorunlarından biri, "checkerboard" (dama tahtası) adı verilen artefaktlara neden olabilmesidir. Bu durum, filtrenin örtüşme oranının düzensiz olmasından kaynaklanır. `stride`'ın `kernel_size`'a tam bölünemediği durumlarda, bazı pikseller diğerlerinden daha fazla "boyanır", bu da çıktıda dama tahtasına benzer, düzensiz bir desen oluşturur.

### Çözüm Yöntemleri

Bu artefaktları önlemek için, transpoze evrişim katmanına alternatif olarak kullanılan daha modern yaklaşımlar vardır:

1.  **Önce Büyüt, Sonra Evrişim (Upsample + Convolution)**: Bu en yaygın ve etkili çözümdür. Önce `nn.Upsample` veya `nn.functional.interpolate` gibi daha basit ve "öğrenilemeyen" bir yöntemle (örneğin, 'nearest' veya 'bilinear' interpolasyon) özellik haritası istenen boyuta getirilir. Ardından, bu büyütülmüş harita üzerinde standart bir `nn.Conv2d` katmanı kullanılır. Bu, checkerboard etkisini ortadan kaldırır ve genellikle daha iyi sonuçlar verir.

    ```python
    # Transpoze evrişim yerine...
    # self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    
    # Bu kullanılır:
    self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(64, 32, kernel_size=3, padding=1)
    )
    ```

2.  **Pixel Shuffle (`nn.PixelShuffle`)**: Bu katman, kanallardaki (channels) bilgiyi uzamsal boyutlara (yükseklik ve genişlik) yeniden düzenleyerek büyütme işlemi yapar. Örneğin, `[N, C * r^2, H, W]` boyutundaki bir tensörü `[N, C, H * r, W * r]` boyutuna dönüştürür. Genellikle daha temiz ve detaylı çıktılar ürettiği için süper çözünürlük görevlerinde popülerdir.

## 6. Soru & Cevap

**S: "Dekonvolüsyon" ve "Transpoze Evrişim" aynı şey midir?**

C: Hayır. Bu iki terim sık sık birbirinin yerine kullanılsa da, teknik olarak farklı anlamlara gelirler. **Dekonvolüsyon**, bir evrişim işleminin matematiksel olarak tam tersini alma işlemidir. Pratikte derin öğrenme kütüphanelerinde (`ConvTranspose2d` gibi) uygulanan işlem ise bir **transpoze evrişimdir**. Transpoze evrişim, yalnızca boyutları tersine çevirir, değerleri değil. Bu nedenle "transpoze evrişim" daha doğru bir isimlendirmedir.

**S: Neden `output_padding`'e ihtiyaç duyarız?**

C: Çıktı boyutu formülünü düşünelim. `stride > 1` olduğunda, birden fazla çıktı boyutunu mümkün kılan parametre kombinasyonları olabilir. Örneğin, `stride=2` ile 13x13'lük bir çıktıdan 25x25 veya 26x26'lık bir çıktı elde etmek mümkün olabilir. `output_padding`, bu belirsizliği çözmek ve tam olarak istediğiniz çıktı boyutunu elde etmek için kullanılır. Encoder-decoder yapılarında, decoder'daki katmanın çıktısının, encoder'daki karşılık gelen katmanınkiyle tam olarak aynı boyutta olması gerektiğinde bu parametre hayati önem taşır.

**S: Transpoze evrişim katmanının parametre sayısı nasıl hesaplanır?**

C: Standart bir evrişim katmanıyla tamamen aynı şekilde hesaplanır. Bias dahil edilirse:
$$ \text{Parametre Sayısı} = (\text{girdi\_kanalı} \times \text{filtre\_yüksekliği} \times \text{filtre\_genişliği} + 1) \times \text{çıktı\_kanalı} $$
Örneğin, `nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)` için: $(64 \times 3 \times 3 + 1) \times 32 = 18464$ parametre bulunur. Ağı, `stride` veya `padding` değerleri parametre sayısını etkilemez.
