# Yapay Sinir Ağları

## Temel Kavramlar

### Yapay Nöron (Perceptron)
Biyolojik nörondan esinlenilmiş yapay nöron modeli:

$y = f(\sum_{i=1}^n w_ix_i + b)$

Burada:
- $x_i$: Giriş değerleri
- $w_i$: Ağırlıklar
- $b$: Bias değeri
- $f$: Aktivasyon fonksiyonu

### Aktivasyon Fonksiyonları

1. **Sigmoid**:
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

2. **Tanh**:
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

3. **ReLU** (Rectified Linear Unit):
   $$f(x) = \max(0, x)$$

### İleri Yayılım (Forward Propagation)

Giriş verisinin ağ katmanları boyunca ilerleyerek çıkış katmanına ulaşması işlemidir. Her katmanda ($l$) şu adımlar gerçekleşir:

1.  **Lineer Kombinasyon:** Önceki katmanın aktivasyonları ($a^{[l-1]}$) ile mevcut katmanın ağırlıkları ($W^{[l]}$) çarpılır ve bias ($b^{[l]}$) eklenir. Bu, katmanın ağırlıklı toplamını ($z^{[l]}$) verir.
    $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
    - $a^{[0]}$ giriş verisidir ($x$).
    - $W^{[l]}$: $l$. katmanın ağırlık matrisi.
    - $b^{[l]}$: $l$. katmanın bias vektörü.

2.  **Aktivasyon:** Lineer kombinasyon sonucu ($z^{[l]}$), katmanın aktivasyon fonksiyonundan ($g^{[l]}$) geçirilir. Bu, katmanın çıkışını veya aktivasyonunu ($a^{[l]}$) oluşturur.
    $$a^{[l]} = g^{[l]}(z^{[l]})$$
    - $g^{[l]}$: $l$. katmanın aktivasyon fonksiyonu (ReLU, Sigmoid, Tanh vb.).

Bu işlem, ağın son katmanına kadar tekrarlanır ve nihai tahmin ($a^{[L]}$ veya $\hat{y}$) elde edilir.

### Geri Yayılım (Backpropagation)

Ağın tahminleri ile gerçek değerler arasındaki hatayı (genellikle bir kayıp fonksiyonu $J$ ile ölçülür) kullanarak, bu hatanın ağın ağırlıklarına ($W$) ve biaslarına ($b$) göre gradyanlarını hesaplama işlemidir. Amaç, bu gradyanları kullanarak parametreleri güncellemek ve hatayı azaltmaktır.

Zincir kuralı kullanılarak, çıkış katmanından başlayıp geriye doğru gidilerek her katmanın parametrelerine göre kayıp fonksiyonunun kısmi türevleri hesaplanır:

Hata fonksiyonu $J$ (veya $E$) için, $l$. katmandaki $j$. nöronun ağırlığı $w_{jk}$'ya göre gradyan:
$$\frac{\partial J}{\partial w_{jk}^{[l]}} = \frac{\partial J}{\partial a^{[L]}} \frac{\partial a^{[L]}}{\partial z^{[L]}} \dots \frac{\partial a^{[l]}}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial w_{jk}^{[l]}}$$

Pratikte bu hesaplama, $\delta^{[l]} = \frac{\partial J}{\partial z^{[l]}}$ terimleri üzerinden daha verimli yapılır:
- Çıkış katmanı için: $\delta^{[L]} = \nabla_{a^{[L]}} J \odot g^{[L]'}(z^{[L]})$
- Önceki katmanlar için: $\delta^{[l]} = (W^{[l+1]T} \delta^{[l+1]}) \odot g^{[l]'}(z^{[l]})$

Gradyanlar:
$$\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} a^{[l-1]T}$$
$$\frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}$$

Bu gradyanlar daha sonra optimizasyon algoritmaları (örn. Gradient Descent) tarafından parametreleri güncellemek için kullanılır:
$$W^{[l]} = W^{[l]} - \eta \frac{\partial J}{\partial W^{[l]}}$$
$$b^{[l]} = b^{[l]} - \eta \frac{\partial J}{\partial b^{[l]}}$$
(Burada $\eta$ öğrenme oranıdır.)

## Eğitim Stratejileri

Gradyan inişi (Gradient Descent) algoritmasının farklı varyasyonları, eğitim verisinin nasıl kullanılacağını belirler:

### 1. Batch Gradient Descent
- **İşlem:** Her parametre güncellemesi için tüm eğitim veri seti kullanılır. Gradyanlar, tüm örnekler üzerinden hesaplanan ortalama kayıp üzerinden bulunur.
- **Avantajları:**
    - Gradyanlar daha doğru ve kararlıdır, bu da daha düzgün bir yakınsama sağlar.
    - Vektörize işlemlerle verimli hesaplanabilir.
- **Dezavantajları:**
    - Çok büyük veri setlerinde her adım çok maliyetli ve yavaştır.
    - Tüm veriyi bellekte tutmak gerekebilir.
    - Keskin yerel minimumlara takılabilir.

### 2. Stochastic Gradient Descent (SGD)
- **İşlem:** Her parametre güncellemesi için rastgele seçilen *tek bir* eğitim örneği kullanılır.
- **Avantajları:**
    - Her adım çok hızlıdır.
    - Düşük bellek kullanımı.
    - Gradyanlardaki gürültü, algoritmanın sığ yerel minimumlardan kaçmasına yardımcı olabilir.
- **Dezavantajları:**
    - Yakınsama çok gürültülü ve kararsız olabilir (parametreler salınım yapabilir).
    - Vektörizasyonun avantajlarından tam olarak yararlanılamaz.
    - Öğrenme oranının dikkatli ayarlanması gerekir.

### 3. Mini-batch Gradient Descent
- **İşlem:** Eğitim veri seti küçük alt kümelere (mini-batch'ler, örn. 32, 64, 128 örnek) bölünür. Her parametre güncellemesi için bir mini-batch kullanılır.
- **Avantajları:**
    - Batch GD'nin kararlılığı ile SGD'nin hızını birleştirir.
    - Vektörize işlemlerden yararlanarak verimli hesaplama sağlar.
    - Bellek kullanımı Batch GD'den daha düşüktür.
    - SGD'den daha kararlı yakınsama sağlar.
- **Dezavantajları:**
    - Yeni bir hiperparametre (batch_size) ekler.
- **Not:** Pratikte derin öğrenme modellerinin eğitiminde en yaygın kullanılan yöntemdir.

## Optimizasyon Algoritmaları

Gradient Descent'in temel güncelleme kuralını iyileştiren, daha hızlı ve daha güvenilir yakınsama sağlayan algoritmalardır.

### 1. Momentum
- **Fikir:** Önceki adımlardaki güncelleme yönünü (momentum) mevcut gradyana ekleyerek güncellemeyi hızlandırır ve salınımları azaltır. Fizikteki momentum kavramına benzer; yokuş aşağı yuvarlanan bir top gibi hız kazanır.
- **Güncelleme Kuralı:**
    $$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$$
    $$\theta = \theta - v_t$$
    - $v_t$: $t$ anındaki momentum (veya hız) vektörü.
    - $\gamma$: Momentum katsayısı (genellikle 0.9 civarı). Önceki momentumun ne kadarının korunacağını belirler.
    - $\eta$: Öğrenme oranı.
- **Etkisi:** Gradyanların sürekli aynı yönde olduğu durumlarda güncellemeyi hızlandırır, yön değiştirdiği durumlarda ise salınımları sönümler.

### 2. Adam (Adaptive Moment Estimation)
- **Fikir:** Her parametre için ayrı ayrı uyarlanabilir öğrenme oranları kullanır. Gradyanların hem birinci momentini (ortalama, momentum gibi) hem de ikinci momentini (karelerinin ortalaması, varyans gibi) takip eder.
- **Güncelleme Kuralı (Basitleştirilmiş):**
    1.  **Momentum (1. Moment):** Gradyanların üssel hareketli ortalaması hesaplanır.
        $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta J(\theta_t)$$
    2.  **RMSprop (2. Moment):** Gradyanların karelerinin üssel hareketli ortalaması hesaplanır.
        $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta J(\theta_t))^2$$
    3.  **Bias Düzeltmesi:** Başlangıçta momentlerin sıfıra yakın olmasını düzeltmek için:
        $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
        $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    4.  **Parametre Güncelleme:**
        $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
    - $\beta_1$, $\beta_2$: Üssel azalma oranları (genellikle 0.9 ve 0.999).
    - $\epsilon$: Sıfıra bölmeyi önlemek için küçük bir sabit (örn. $10^{-8}$).
- **Etkisi:** Momentum ve RMSprop'un avantajlarını birleştirir. Genellikle farklı problemler için iyi çalışır ve daha az hiperparametre ayarı gerektirir. Pratikte sıklıkla varsayılan optimizasyon algoritması olarak kullanılır.

## Düzenlileştirme (Regularization)

### 1. L1 Regularization
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n |\theta_i|$$

### 2. L2 Regularization
$$J_{yeni}(\theta) = J(\theta) + \lambda \sum_{i=1}^n \theta_i^2$$

### 3. Dropout
- Eğitim sırasında rastgele nöronları devre dışı bırakma
- Overfitting'i önlemeye yardımcı olur

## Model Mimarisi Seçimi

### 1. Katman Sayısı
- Problem karmaşıklığına göre belirlenir
- Çok derin ağlar gradient vanishing/exploding problemi yaşayabilir

### 2. Nöron Sayısı
- Genelde piramit yapısı tercih edilir
- Giriş boyutu > Ara katmanlar > Çıkış boyutu

### 3. Aktivasyon Fonksiyonu Seçimi
- Gizli katmanlar: ReLU (genelde)
- Çıkış katmanı: Problem tipine göre
  - Sınıflandırma: Sigmoid/Softmax
  - Regresyon: Linear
