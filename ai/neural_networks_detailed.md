# Yapay Sinir Ağları (YSA) - Detaylı Ders Notları

Bu notlar, yapay sinir ağlarının temel kavramlarından başlayarak, modern derin öğrenme mimarilerinde kullanılan ileri düzey tekniklere kadar geniş bir yelpazeyi kapsamaktadır.

## 1. Giriş: Yapay Sinir Ağları Nedir?

Yapay Sinir Ağları (YSA), insan beyninin çalışma şeklinden esinlenerek geliştirilmiş bir makine öğrenmesi modelidir. Birbirine bağlı "nöron"lardan oluşan katmanlı bir yapıya sahiptir. Bu ağlar, karmaşık desenleri tanımak, verileri sınıflandırmak veya gelecekteki olayları tahmin etmek gibi görevler için eğitilebilir.

## 2. Temel Yapı Taşı: Yapay Nöron (Perceptron)

Her bir yapay sinir ağı, en temel birim olan yapay nörondan oluşur. Biyolojik nörondan esinlenen bu model, girdileri alır, onları belirli ağırlıklarla çarpar, bir bias değeri ekler ve sonucu bir aktivasyon fonksiyonundan geçirerek bir çıktı üretir.

Matematiksel olarak bir nöronun çıktısı şu şekilde ifade edilir:
$$y = f(\sum_{i=1}^n w_ix_i + b)$$

Burada:
-   $x_i$: Giriş değerleri (bir önceki katmanın çıktıları veya ham veri).
-   $w_i$: Ağırlıklar (her bir girişin önemini belirten öğrenilebilir parametreler).
-   $b$: Bias değeri (nöronun ne kadar kolay aktifleşeceğini belirleyen öğrenilebilir bir parametre).
-   $f$: Aktivasyon fonksiyonu (nöronun çıktısını normalleştiren ve non-lineerlik katan fonksiyon).

### Soru & Cevap

**S: Perceptron'un biyolojik nörondan temel farkı nedir?**

C: Biyolojik nöronlar karmaşık kimyasal ve elektriksel sinyallerle (aksiyon potansiyelleri) çalışırken, Perceptron bu süreci basitleştirilmiş bir matematiksel modele indirger: ağırlıklı toplam ve aktivasyon eşiği. Biyolojik nörondaki sinapslar ağırlıklara ($w_i$), hücre gövdesi toplamaya ve aktivasyon eşiğine, akson ise çıktıya karşılık gelir.

**S: Bias ($b$) teriminin matematiksel amacı nedir?**

C: Bias, aktivasyon fonksiyonu grafiğini yatay eksende sola veya sağa kaydırır. Eğer bias olmasaydı ($b=0$), bir nöronun lineer kısmı olan $z = \sum w_ix_i$ ifadesi, $x_i$ girdileri sıfır olduğunda her zaman sıfır olurdu. Bias eklemek, $z = \sum w_ix_i + b$, nöronun tüm girdiler sıfır olsa bile bir çıktı üretmesini sağlar. Geometrik olarak bu, karar sınırının orijinden geçmek zorunda kalmamasını sağlar, bu da modelin esnekliğini ve öğrenme kapasitesini artırır.

## 3. Aktivasyon Fonksiyonları

Aktivasyon fonksiyonları, bir sinir ağının non-lineer (doğrusal olmayan) problemleri öğrenebilmesini sağlar. Eğer aktivasyon fonksiyonu olmasaydı, kaç katman eklersek ekleyelim ağımız sadece lineer bir dönüşüm yapabilirdi.

### Yaygın Kullanılan Aktivasyon Fonksiyonları

1.  **Sigmoid**:
    -   Formül: $\sigma(x) = \frac{1}{1 + e^{-x}}$
    -   Türevi: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
    -   Çıktı Aralığı: (0, 1)
    -   Kullanım: Genellikle ikili sınıflandırma problemlerinin çıkış katmanında olasılık değeri elde etmek için kullanılır.
    -   Dezavantajları: "Vanishing gradient" (türevlerin sıfıra yaklaşması) problemine yol açabilir, bu da öğrenmeyi yavaşlatır. Çıktısı sıfır merkezli değildir.

2.  **Tanh (Hiperbolik Tanjant)**:
    -   Formül: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$
    -   Türevi: $\tanh'(x) = 1 - \tanh^2(x)$
    -   Çıktı Aralığı: (-1, 1)
    -   Özellikleri: Sigmoid'e benzer ancak sıfır merkezli olması bir avantajdır. Yine de vanishing gradient problemi yaşayabilir.

3.  **ReLU (Rectified Linear Unit)**:
    -   Formül: $f(x) = \max(0, x)$
    -   Türevi: $f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$
    -   Özellikleri: Hesaplaması çok hızlıdır ve pratikte çok iyi sonuçlar verir. Derin öğrenmede gizli katmanlar için en popüler seçimdir.
    -   Dezavantajları: "Dying ReLU" problemine yol açabilir (negatif girdiler için türevin sıfır olması ve o nöronun bir daha güncellenmemesi).

4.  **Leaky ReLU**:
    -   Formül: $f(x) = \max(0.01x, x)$
    -   Özellikleri: Dying ReLU problemini çözmek için negatif girdilere küçük bir eğim verir.

5.  **Softmax**:
    -   Kullanım: Çok sınıflı sınıflandırma problemlerinin çıkış katmanında kullanılır. Çıktıları, her sınıf için bir olasılık dağılımı olarak yorumlanabilecek şekilde normalleştirir.
    -   Formül: $S(y_i) = \frac{e^{y_i}}{\sum_{j=1}^k e^{y_j}}$ (Burada $y$ vektörü, çıkış katmanındaki lineer dönüşümün sonucudur)

### Soru & Cevap

**S: Neden aktivasyon fonksiyonlarına ihtiyacımız var? Eğer kullanmasaydık ne olurdu?**

C: Aktivasyon fonksiyonları ağa **non-lineerlik (doğrusal olmama)** özelliği kazandırır. Eğer aktivasyon fonksiyonu kullanmasaydık (veya lineer bir aktivasyon fonksiyonu $f(x)=x$ kullansaydık), her katman sadece bir lineer dönüşüm yapardı. İki lineer dönüşümün birleşimi yine lineer bir dönüşümdür. Örneğin, iki katmanlı bir ağ şu hale gelirdi:
$$ a^{[2]} = W^{[2]}a^{[1]} + b^{[2]} = W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]} = (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]}) $$
Bu ifade, $W_{yeni} = W^{[2]}W^{[1]}$ ve $b_{yeni} = W^{[2]}b^{[1]} + b^{[2]}$ olmak üzere, $a^{[2]} = W_{yeni}x + b_{yeni}$ formundadır. Yani, kaç katman eklersek ekleyelim, ağımız tek katmanlı bir lineer modelden (Lineer Regresyon gibi) daha güçlü olmazdı. Karmaşık ve doğrusal olmayan ilişkileri (örneğin bir XOR problemi) öğrenemezdi.

**S: "Vanishing Gradient" problemi matematiksel olarak nedir ve ReLU bunu nasıl çözer?**

C: Vanishing gradient, geri yayılım sırasında gradyanların (türevlerin) zincir kuralı ile çarpılarak katman katman geriye gittikçe küçülerek sıfıra yaklaşmasıdır. Özellikle Sigmoid ($\sigma'(x) \in (0, 0.25]$) ve Tanh ($\tanh'(x) \in (0, 1]$) fonksiyonlarının türevleri 1'den küçük veya eşittir. Derin ağlarda bu küçük sayıların sürekli birbiriyle çarpılması, başlangıç katmanlarına yakın gradyanların neredeyse sıfır olmasına neden olur ($0.25^N$ çok hızlı küçülür). Bu da o katmanlardaki ağırlıkların güncellenmemesine, yani "öğrenmemesine" yol açar.
ReLU'nun türevi ise pozitif girdiler için sabittir ve 1'dir ($f'(x) = 1$ if $x > 0$). Bu sayede, aktivasyon pozitif olduğu sürece gradyan geriye doğru yayılırken küçülmez ve kaybolmaz. Bu, derin ağların çok daha verimli eğitilmesini sağlar.

**S: Python'da (Numpy ile) ReLU ve türevini nasıl yazarız?**

C:
```python
import numpy as np

def relu(z):
    # z'deki her eleman için, 0'dan büyükse kendisini, değilse 0'ı döndürür.
    return np.maximum(0, z)

def relu_derivative(z):
    # z > 0 olan yerler için 1, diğer yerler için 0 döndürür.
    # Bu, pratikteki uygulamadır. Matematiksel olarak z=0'da türev tanımsızdır
    # ancak pratikte bu durum nadiren sorun yaratır ve 0 olarak ele alınır.
    return np.where(z > 0, 1, 0)

# Kullanım
z = np.array([-2, -0.5, 0, 1, 3])
a = relu(z)
# a -> array([0., 0., 0., 1., 3.])
        
da_dz = relu_derivative(z)
# da_dz -> array([0, 0, 0, 1, 1])
```

## 4. Ağ Mimarisi ve İleri Yayılım (Forward Propagation)

Bir sinir ağı, katmanlar halinde düzenlenmiş nöronlardan oluşur: bir giriş katmanı, bir veya daha fazla gizli katman ve bir çıkış katmanı.

**İleri Yayılım**, girdinin ağ üzerinden katman katman ilerleyerek bir çıktı üretmesi sürecidir. Her katmanda şu işlemler yapılır:

1.  Bir önceki katmanın çıktısı ($a^{[l-1]}$) alınır.
2.  Ağırlık matrisi ($W^{[l]}$) ile çarpılır ve bias vektörü ($b^{[l]}$) eklenir. Bu adıma **lineer dönüşüm** denir:
    $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
3.  Sonuç, o katmanın aktivasyon fonksiyonundan ($g^{[l]}$) geçirilir:
    $$a^{[l]} = g^{[l]}(z^{[l]})$$
Bu işlem, çıkış katmanına ulaşana kadar her katman için tekrarlanır.

### Boyut Analizi (Önemli!)
-   Eğer katman $l-1$'de $n^{[l-1]}$ nöron ve katman $l$'de $n^{[l]}$ nöron varsa:
    -   $a^{[l-1]}$'in boyutu $(n^{[l-1]}, m)$ olur (m: örnek sayısı).
    -   $W^{[l]}$'in boyutu $(n^{[l]}, n^{[l-1]})$ olur.
    -   $b^{[l]}$'in boyutu $(n^{[l]}, 1)$ olur (Python'da broadcasting ile toplanır).
    -   $z^{[l]}$ ve $a^{[l]}$'nin boyutu $(n^{[l]}, m)$ olur.

## 5. Öğrenme Süreci: Geri Yayılım ve Optimizasyon

Ağın "öğrenmesi", tahmin edilen çıktı ile gerçek çıktı arasındaki hatayı minimize edecek şekilde ağırlıkların ($W$) ve bias'ların ($b$) ayarlanması demektir.

### Hata (Maliyet) Fonksiyonu - Loss Function

Modelin ne kadar "kötü" performans gösterdiğini ölçen bir fonksiyondur.
-   **Regresyon için**: Mean Squared Error (MSE)
-   **Sınıflandırma için**: Cross-Entropy Loss (Binary veya Categorical)

### Geri Yayılım (Backpropagation)

Geri yayılım, hatayı minimize etmek için ağırlıkları ve bias'ları nasıl güncelleyeceğimizi belirleyen algoritmadır. Özünde, maliyet fonksiyonunun her bir parametreye (ağırlık ve bias) göre kısmi türevini (gradyanını) hesaplamak için **zincir kuralını** kullanır.

Algoritma, çıkış katmanından başlar ve geriye doğru çalışır. Her $l$ katmanı için aşağıdaki gradyanları hesaplar:

1.  **$dZ^{[l]}$**: Maliyetin, $l$. katmandaki lineer çıktı $Z^{[l]}$'ye göre gradyanı.
    $$ dZ^{[l]} = dA^{[l]} * g^{[l]'}(Z^{[l]}) $$
    Burada `*` eleman bazında çarpımdır (Hadamard product) ve $g^{[l]'}$ aktivasyon fonksiyonunun türevidir. $dA^{[l]}$ ise bir sonraki katmandan gelen gradyandır ($dA^{[l]} = W^{[l+1]T} dZ^{[l+1]}$).

2.  **$dW^{[l]}$**: Maliyetin, $l$. katmandaki ağırlık matrisi $W^{[l]}$'ye göre gradyanı.
    $$ dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} $$
    (m: batch'teki örnek sayısı, ortalama almak için bölünür).

3.  **$db^{[l]}$**: Maliyetin, $l$. katmandaki bias vektörü $b^{[l]}$'ye göre gradyanı.
    $$ db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)} $$
    (Python'da `np.sum(dZ, axis=1, keepdims=True)` ile kolayca hesaplanır).

4.  **$dA^{[l-1]}$**: Maliyetin, bir önceki katmanın aktivasyonu $A^{[l-1]}$'e göre gradyanı. Bu, bir sonraki döngüde kullanılmak üzere geriye aktarılır.
    $$ dA^{[l-1]} = W^{[l]T} dZ^{[l]} $$

Bu hesaplamalar, giriş katmanına kadar geriye doğru tekrarlanır.

### Gradient Descent (Gradyan İnişi)

Gradyanlar hesaplandıktan sonra, parametreleri gradyanın tersi yönünde küçük bir adımla güncelleriz. Bu işleme **Gradient Descent** denir.
$$\theta_{yeni} = \theta_{eski} - \eta \nabla_\theta J(\theta)$$
-   $\theta$: Güncellenecek parametre (bir ağırlık veya bias).
-   $\eta$: **Öğrenme oranı (learning rate)**, her adımda ne kadar ilerleyeceğimizi belirleyen bir hiperparametredir.
-   $\nabla_\theta J(\theta)$: Maliyet fonksiyonunun $\theta$'ya göre gradyanı.

## 6. Eğitim Stratejileri

Gradyanların nasıl ve ne sıklıkla hesaplandığına göre farklı eğitim stratejileri vardır:

1.  **Batch Gradient Descent**: Her güncelleme için tüm veri setini kullanır. Kararlı ama yavaştır ve büyük veri setlerinde çok fazla bellek gerektirir.
2.  **Stochastic Gradient Descent (SGD)**: Her güncelleme için rastgele seçilmiş tek bir veri örneğini kullanır. Çok hızlıdır ama güncellemeler gürültülü olabilir.
3.  **Mini-batch Gradient Descent**: Pratikte en yaygın kullanılan yöntemdir. Veri setini küçük "batch"lere (örneğin 32, 64, 128'lik gruplar) ayırır ve her güncelleme için bir batch kullanır. Bu, SGD'nin hız avantajı ile Batch GD'nin kararlılığını birleştirir.

## 7. Gelişmiş Optimizasyon Algoritmaları

Standart Gradient Descent'in yavaş yakınsama veya kötü yerel minimumlara takılma gibi sorunlarını çözmek için daha gelişmiş optimizasyon algoritmaları geliştirilmiştir.

1.  **Momentum**: Önceki adımlardaki güncelleme yönünü bir "momentum" terimi olarak hesaba katar. Bu, algoritmanın küçük salınımları yumuşatmasına ve daha tutarlı bir yönde ilerlemesine yardımcı olur.
    $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
    $$\theta = \theta - v_t$$
2.  **Adam (Adaptive Moment Estimation)**: Momentum (gradyanların birinci momenti) ve RMSprop (gradyanların karesinin ikinci momenti) fikirlerini birleştirir. Her parametre için ayrı ve adaptif bir öğrenme oranı kullanır. Genellikle çoğu problem için varsayılan olarak iyi çalışan, hızlı ve güvenilir bir optimizasyon algoritmasıdır.

## 8. Aşırı Öğrenmeyi (Overfitting) Önleme: Düzenlileştirme (Regularization)

**Overfitting**, bir modelin eğitim verisini "ezberlemesi" ancak yeni, daha önce görmediği verilerde kötü performans göstermesidir. Bunu önlemek için düzenlileştirme teknikleri kullanılır.

1.  **L1 ve L2 Regularization**: Maliyet fonksiyonuna, ağırlıkların büyüklüğünü cezalandıran bir terim ekler. Bu, modelin çok büyük ağırlık değerleri öğrenmesini engelleyerek daha "basit" bir model olmaya zorlar.
    -   L1 Regularization: $J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n |\theta_i|$ (Bazı ağırlıkları sıfır yapabilir, özellik seçimi için kullanışlıdır).
    -   L2 Regularization: $J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n \theta_i^2$ (Ağırlıkları küçültür ama sıfır yapmaz).
2.  **Dropout**: Eğitim sırasında, her adımda rastgele seçilen bazı nöronları (çıktılarıyla birlikte) geçici olarak ağdan "atar". Bu, nöronların belirli diğer nöronlara aşırı bağımlı hale gelmesini engeller ve daha dayanıklı (robust) özellikler öğrenmelerini teşvik eder.
3.  **Early Stopping**: Modelin test (veya validasyon) verisi üzerindeki performansı artmayı durdurduğunda (hatta kötüleşmeye başladığında) eğitimi erken durdurma tekniğidir.

## 9. Parametreler ve Hiperparametreler

-   **Parametreler**: Modelin veri üzerinden kendi kendine öğrendiği değerlerdir. Bunlar ağırlıklar ($W$) ve bias'lardır ($b$).
-   **Hiperparametreler**: Modelin öğrenme sürecini kontrol eden, bizim tarafımızdan ayarlanan değerlerdir. Bunlar:
    -   Öğrenme oranı (learning rate)
    -   Epoch (iterasyon) sayısı
    -   Batch boyutu
    -   Gizli katman sayısı
    -   Her katmandaki nöron sayısı
    -   Aktivasyon fonksiyonu seçimi
    -   Optimizasyon algoritması
    -   Düzenlileştirme parametresi ($\lambda$)

Doğru hiperparametreleri bulmak, model performansını önemli ölçüde etkiler ve genellikle deneme yanılma (grid search, random search vb.) gerektirir.

### Soru & Cevap

**S: Hiperparametre optimizasyonu için hangi yöntemler kullanılır?**

C: En yaygın yöntemler şunlardır:
1.  **Grid Search**: Belirlediğiniz hiperparametreler için olası tüm kombinasyonları dener. Örneğin, öğrenme oranı için `[0.1, 0.01, 0.001]` ve batch boyutu için `[32, 64]` belirlerseniz, toplamda 3x2=6 farklı model eğitilir ve en iyi performansı veren kombinasyon seçilir. Sistematiktir ancak parametre sayısı arttıkça hesaplama maliyeti katlanarak artar.
2.  **Random Search**: Belirlediğiniz aralıklardan rastgele hiperparametre kombinasyonları seçerek dener. Genellikle Grid Search'ten daha verimlidir çünkü bazı hiperparametrelerin diğerlerinden daha önemli olduğu durumlarda, daha geniş bir aralığı daha az deneme ile tarayabilir.
3.  **Bayesian Optimization**: Bir sonraki denenecek hiperparametre setini, önceki sonuçlara dayanarak akıllıca seçen bir yöntemdir. Daha az deneme ile daha iyi sonuçlar bulma potansiyeline sahiptir ancak kurulumu daha karmaşıktır.

**S: Bir PyTorch modelinde parametreler ve hiperparametreler nasıl görünür?**

C:
```python
import torch
import torch.nn as nn

# Hiperparametreler
learning_rate = 0.01
n_epochs = 100
batch_size = 64
input_size = 784  # MNIST için 28x28 piksel
hidden_size = 500
num_classes = 10

# Model Tanımı
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Bu katmanlardaki ağırlıklar ve biaslar, öğrenilecek 'parametrelerdir'.
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Parametreler modelin içinde saklanır ve optimizer tarafından güncellenir.
# model.parameters() bu parametreleri optimizer'a verir.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
Bu örnekte `learning_rate`, `n_epochs`, `batch_size` gibi değişkenler bizim tarafımızdan ayarlanan hiperparametrelerdir. `model.fc1` ve `model.fc2` katmanlarının içindeki ağırlık (`weight`) ve sapma (`bias`) tensörleri ise eğitim sırasında `optimizer` tarafından güncellenen parametrelerdir.
