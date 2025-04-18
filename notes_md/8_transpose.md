# Transposed Convolution (Deconvolution)

## Temel Kavramlar

### Transposed Convolution Nedir?
- Normal konvolüsyonun tersi işlem
- Upsampling (yukarı örnekleme) için kullanılır
- Encoder-Decoder mimarilerinde yaygın

### Matematiksel Tanım
$$y = W^T x$$
Burada:
- $x$: Giriş
- $W^T$: Transpoze edilmiş konvolüsyon matrisi
- $y$: Çıkış

## Çalışma Prensibi

### Normal Konvolüsyon vs Transposed
```
Normal:      n x n * f x f → (n-f+1) x (n-f+1)
Transposed:  n x n → (n+f-1) x (n+f-1)
```

### İşlem Adımları
1. Padding ekleme
2. Stride ayarlama
3. Konvolüsyon matrisi transpozu
4. Çıktı hesaplama

## Uygulama Alanları

### 1. Otoenkoderler
```
Input → Encoder → Latent → Decoder → Output
      ↓          ↓        ↓         ↓
    Conv       Conv    TransConv  TransConv
```

### 2. Semantic Segmentation
- U-Net mimarisi
- Feature extraction
- Upsampling

### 3. Super Resolution
- Düşük çözünürlükten yüksek çözünürlüğe
- Detail generation
- Quality enhancement

## Hiperparametreler

### 1. Output Padding
$$output\_size = (input\_size - 1) * stride + kernel\_size - 2 * padding$$

### 2. Stride
- Upsampling faktörü
- Çıktı boyutu kontrolü
- Yaygın değerler: 2, 4

### 3. Kernel Size
- Öğrenilecek parametre sayısı
- Receptive field
- Yaygın değerler: 3x3, 4x4

## Pytorch Implementasyonu

```python
import torch.nn as nn

class TransposedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1
        )
    
    def forward(self, x):
        return self.conv_transpose(x)
```

## Checkerboard Artifaktları

### Problem
- Düzensiz örtüşme
- Piksel değeri dalgalanmaları
- Görsel artifaktlar

### Çözümler
1. **Resize + Convolution**
```python
nn.Upsample(scale_factor=2, mode='bilinear')
nn.Conv2d(64, 3, kernel_size=3, padding=1)
```

2. **Pixel Shuffle**
```python
nn.PixelShuffle(upscale_factor=2)
```

## Best Practices

### 1. Initialization
- Xavier/Glorot initialization
- Kernel size'a göre scaling
- Bias initialization

### 2. Normalization
- Batch Normalization
- Instance Normalization
- Layer Normalization

### 3. Activation
- ReLU
- LeakyReLU
- Sigmoid (son katman)

## Advanced Topics

### 1. Group Convolution
$$y_i = \sum_{j \in \mathcal{G}_i} W_{ij} * x_j$$

### 2. Dilated Convolution
- Receptive field artırma
- Spatial resolution korunumu
- Semantic segmentation

### 3. Adaptif Kernel Size
- Dynamic filters
- Content-aware convolution
- Spatial attention


