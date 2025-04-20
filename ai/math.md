# Bilgisayarlı Görme ve Makine Öğrenmesi için Matematiksel Temeller

Bu doküman, `notes_md/` dizinindeki ayrıntılı teorik ders notlarına ek olarak,
`Uygulama/` dizinindeki pratik örneklerle kavramları pekiştirir.

## İçindekiler
- Vektör ve Matris İşlemleri
- Mesafe Metriği
- Doğrusal Regresyon
- Lojistik Regresyon
- Kaynaklar

## Vektör ve Matris İşlemleri

### Nokta Çarpımı (Dot Product)
İki vektörün karşılık gelen elemanlarının çarpımlarının toplamıdır:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

**Kodda Kullanımı:**
- `np.dot` fonksiyonu, `Uygulama/2_lin_reg.ipynb` içindeki `LinearRegression` sınıfının `predict` metodunda ve çok değişkenli model tahminlerinde kullanılmaktadır.

## Mesafe Metriği

### L1 (Manhattan) ve L2 (Öklid) Mesafesi
- L1: \(L1(a,b)=\sum_i|a_i-b_i|\)
- L2: \(L2(a,b)=\sqrt{\sum_i(a_i-b_i)^2}\)

```python
# Uygulama/0_hello.ipynb'den L1 ve L2 hesaplama örneği
import numpy as np

def l1_distance(a,b):
    return np.sum(np.abs(a-b))

def l2_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
```
**Kodda Kullanımı:**
- `l1_distance` ve `l2_distance` fonksiyonları, `Uygulama/0_hello.ipynb` hücrelerinde tanımlanarak KNN algoritmalarında uzaklık metriği olarak kullanılmaktadır.

## Doğrusal Regresyon

Tek değişkenli ve çok değişkenli doğrusal regresyon hakkında ayrıntılı bilgiler `notes_md/2_lin_reg.md` içinde yer alır.

```python
# Uygulama/2_lin_reg.ipynb'den basit lineer regresyon örneği
import numpy as np
from sklearn.linear_model import LinearRegression

# Veri hazırlama
datas = np.array([[1],[2],[3],[4]])
targets = np.array([2,4,6,8])

model = LinearRegression()
model.fit(datas, targets)
print(f"Katsayı: {model.coef_[0]}, Sabit: {model.intercept_}")
```
**Kodda Kullanımı:**
- Maliyet fonksiyonu ve gradyan inişi, `Uygulama/2_lin_reg.ipynb` içindeki `LinearRegression_1_feature.fit` ve `LinearRegression.fit` metodlarında `hata_fonksiyonu`, `d_theta_0`, `d_theta_1` hesaplamaları ve parametre güncellemeleri ile gerçekleştirilir.

## Lojistik Regresyon

İkili sınıflandırma için kullanılan lojistik regresyonun teorisi `notes_md/3_log_reg.md`'de bulunur.

```python
# Uygulama/3_log_reg.ipynb'den sigmoid fonksiyonu örneği
def sigmoid(z):
    return 1/(1+np.exp(-z))
```
**Kodda Kullanımı:**
- `sigmoid` fonksiyonu, `Uygulama/3_log_reg.ipynb` hücresinde tanımlanır.
- PyTorch ile `LogReg` sınıfı, `Uygulama/4_hello_torch.ipynb` içinde `torch.sigmoid` kullanılarak uygulanmaktadır.

## Kaynaklar
- Teorik Ders Notları: `notes_md/`
- Uygulama Örnekleri: `Uygulama/`
