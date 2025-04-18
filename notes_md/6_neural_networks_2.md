# İleri Düzey Yapay Sinir Ağları

## Modern Optimizasyon Teknikleri

### 1. Adam Optimizer


$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$


### 2. Learning Rate Scheduling


#### Cosine Annealing


$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t\pi}{T}))$$

#### Step Decay


$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

### 3. Gradient Clipping


$$g_t = \min(\frac{\theta}{||g_t||_2}, 1)g_t$$

## İleri Düzey Regularizasyon


### 1. Weight Decay


$$L_{new} = L + \lambda \sum_w w^2$$

### 2. Label Smoothing

$$y_{smooth} = (1-\alpha)y_{one-hot} + \alpha/K$$


### 3. Mixup Training

$$\tilde{x} = \lambda x_i + (1-\lambda)x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda)y_j$$

## Modern Mimariler

### 1. ResNet Blokları

```python
class ResBlock(nn.Module):

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.conv2(out)
        out += identity
        return F.relu(out)
```

### 2. Attention Mekanizması

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$


### 3. Transformer Blokları

- Multi-head attention
- Position-wise feed-forward

- Layer normalization
- Residual connections


## Modern Training Teknikleri

### 1. Progressive Training

- Düşük çözünürlükten başlama
- Kademeli olarak arttırma
- Hızlı yakınsama


### 2. Knowledge Distillation


$$L_{KD} = \alpha L_{CE}(y, \sigma(z_s/T)) + (1-\alpha)L_{CE}(y_t, \sigma(z_s/T))$$

### 3. Few-Shot Learning


- Meta-learning
- Prototype networks
- Matching networks


## Advanced Loss Functions

### 1. Focal Loss

$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

### 2. Contrastive Loss


$$L = \frac{1}{2N}\sum_{i=1}^N [y_i d_i^2 + (1-y_i)\max(margin-d_i, 0)^2]$$

### 3. Triplet Loss


$$L = \max(d(a,p) - d(a,n) + margin, 0)$$

## Model Analysis

### 1. Gradient Flow Analysis

```python

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.grad.abs().mean()}")
```


### 2. Feature Visualization

- Activation maximization
- Gradient ascent
- Layer visualization


### 3. Attribution Methods

- Integrated Gradients
- GradCAM
- SHAP values


## Architecture Search

### 1. Neural Architecture Search (NAS)

- Reinforcement learning based

- Gradient based
- Evolution based

### 2. AutoML


- Hiperparametre optimizasyonu
- Model seçimi
- Pipeline optimizasyonu

- Subnetwork sampling
- Progressive training
- Architecture adaptation

## Deployment Strategies

### 1. Model Quantization

- INT8 quantization
- Weight pruning
- Knowledge distillation

### 2. Model Serving

- TorchServe
- ONNX Runtime
- TensorRT

### 3. Edge Deployment

- TFLite
- CoreML
- Model optimization
