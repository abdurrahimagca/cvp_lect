import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import idx2numpy

# =============================================================================
# TODO 1: Veri yükleme ve ön işleme
# Challenge: Normalizasyonu değiştirmeyi dene (örn: 0-1 yerine -1 ile 1 arası)
# =============================================================================

MNIST_DIR = "../../mnist/"

X_mnist = idx2numpy.convert_from_file(MNIST_DIR + "train-images-idx3-ubyte")

X_mnist = X_mnist.reshape(60000, -1) / 255.0  # TODO: Farklı normalizasyon teknikleri dene
# X_mnist = (X_mnist.reshape(60000, -1) / 255.0) * 2.0 - 1.0

y_mnist = idx2numpy.convert_from_file(MNIST_DIR + "train-labels-idx1-ubyte")

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)  

x = torch.from_numpy(X_train.astype(np.float32))
y = torch.from_numpy(y_train.astype(np.int64))

class MnistDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0] 

# =============================================================================
# TODO 2: Eğitim fonksiyonunu geliştir
# Challenge: Early stopping, learning rate scheduler ekle
# =============================================================================

def train_model(model, train_data_loader, test_data_loader, epochs=20, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)  # TODO: Adam optimizer'ı dene
    train_loss = []
    validation_loss = []
    
    # TODO: Accuracy tracking ekle
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(epochs):  # TODO: Epoch sayısını ayarlanabilir yap
        batch_train_loss = []
        
        # TODO: Accuracy hesaplama için doğru tahmin sayacı ekle
        
        for X, y in train_data_loader:
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_train_loss.append(loss.item())
        
        train_loss.append(np.array(batch_train_loss).mean())

        with torch.no_grad():
            batch_validation_loss = []
            for X, y in test_data_loader:
                y_hat = model(X)
                loss = criterion(y_hat, y)
                batch_validation_loss.append(loss.item())

            validation_loss.append(np.array(batch_validation_loss).mean())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} done!")
            
    return train_loss, validation_loss

# =============================================================================
# CHALLENGE 1: Model mimarilerini geliştir
# - Dropout ekle
# - Batch normalization dene
# - Daha derin modeller oluştur
# =============================================================================

class MNistClassifier_1(nn.Module):
    """Basit linear model - baseline"""
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 10)
        
    def forward(self, x):
        x = self.input_layer(x)
        return x

class MNistClassifier_2(nn.Module):
    """Tek gizli katmanlı model - Sigmoid aktivasyon"""
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 16)
        self.activation = nn.Sigmoid()
        self.output_layer = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class MNistClassifier_3(nn.Module):
    """Tek gizli katmanlı model - ReLU aktivasyon"""
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 16)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# =============================================================================
# TODO 3: Yeni model sınıfları oluştur
# Challenge: Aşağıdaki modelleri implement et
# =============================================================================

# TODO: MNistClassifier_4 - Dropout'lu model
class MNistClassifier_4(nn.Module):
    """TODO: Dropout ekleyerek overfitting'i azalt"""
    def __init__(self):
        super().__init__()
        # TODO: Dropout layer'ları ekle (örn: nn.Dropout(0.2))
        pass
        
    def forward(self, x):
        # TODO: Forward pass'i implement et
        pass

# TODO: MNistClassifier_5 - Daha derin model
class MNistClassifier_5(nn.Module):
    """TODO: 3+ katmanlı derin model oluştur"""
    def __init__(self):
        super().__init__()
        # TODO: Birden fazla gizli katman ekle
        # Örn: 784 -> 128 -> 64 -> 32 -> 10
        pass
        
    def forward(self, x):
        # TODO: Derin ağ için forward pass
        pass

# TODO: MNistClassifier_6 - Batch Normalization'lı model
class MNistClassifier_6(nn.Module):
    """TODO: Batch normalization ekle"""
    def __init__(self):
        super().__init__()
        # TODO: nn.BatchNorm1d katmanları ekle
        pass
        
    def forward(self, x):
        # TODO: Batch norm ile forward pass
        pass

# =============================================================================
# CHALLENGE 2: CNN modeli oluştur
# MNIST için convolutional neural network dene
# =============================================================================

# TODO: CNN Sınıfı
class MNistCNN(nn.Module):
    """TODO: Convolutional Neural Network implement et"""
    def __init__(self):
        super().__init__()
        # TODO: Conv2d, MaxPool2d, flatten işlemleri
        # İpucu: MNIST 28x28 boyutunda, reshape etmen gerekebilir
        pass
        
    def forward(self, x):
        # TODO: CNN forward pass
        # İpucu: x.view() ile reshape et
        pass

# =============================================================================
# Veri yükleyicileri
# =============================================================================

dataset_train = MnistDataset(X_train, y_train)
data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)

dataset_test = MnistDataset(X_test, y_test)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)

# =============================================================================
# TODO 4: Görselleştirme fonksiyonlarını geliştir
# =============================================================================

def plot_loss(train_loss, validation_loss):
    """Loss grafiğini çiz"""
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, c="b", label="Train")
    plt.plot(validation_loss, c="r", label="Validation")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

# TODO: Accuracy plotting fonksiyonu
def plot_accuracy(train_acc, validation_acc):
    """TODO: Accuracy grafiğini çizen fonksiyon yaz"""
    pass

# TODO: Confusion matrix fonksiyonu
def plot_confusion_matrix(model, data_loader):
    """TODO: Confusion matrix çizen fonksiyon"""
    pass

# TODO: Yanlış sınıflandırılan örnekleri göster
def show_misclassified_examples(model, data_loader, num_examples=10):
    """TODO: Yanlış tahmin edilen digit'ları göster"""
    pass

# =============================================================================
# CHALLENGE 3: Model karşılaştırma sistemi
# =============================================================================

def compare_models(models, model_names, train_loader, test_loader):
    """TODO: Birden fazla modeli karşılaştır"""
    # TODO: Her modeli eğit ve sonuçları kaydet
    # TODO: Karşılaştırmalı grafik oluştur
    pass

# =============================================================================
# Ana eğitim kodu - Mevcut modeller
# =============================================================================

if __name__ == "__main__":
    print("=== Model 1: Linear Classifier ===")
    model_1 = MNistClassifier_1()
    train_loss_1, validation_loss_1 = train_model(model_1, data_loader_train, data_loader_test)
    plot_loss(train_loss_1, validation_loss_1)
    
    print("\n=== Model 2: Sigmoid Hidden Layer ===")
    model_2 = MNistClassifier_2()
    train_loss_2, validation_loss_2 = train_model(model_2, data_loader_train, data_loader_test)
    plot_loss(train_loss_2, validation_loss_2)
    
    print("\n=== Model 3: ReLU Hidden Layer ===")
    model_3 = MNistClassifier_3()
    train_loss_3, validation_loss_3 = train_model(model_3, data_loader_train, data_loader_test)
    plot_loss(train_loss_3, validation_loss_3)
    
    # =============================================================================
    # TODO 5: Yeni modellerinizi test edin
    # =============================================================================
    
    # TODO: Model 4'ü eğit ve test et
    # print("\n=== Model 4: Dropout Model ===")
    # model_4 = MNistClassifier_4()
    # train_loss_4, validation_loss_4 = train_model(model_4, data_loader_train, data_loader_test)
    # plot_loss(train_loss_4, validation_loss_4)
    
    # TODO: Diğer modellerinizi de benzer şekilde test edin
    
    # =============================================================================
    # CHALLENGE 4: Hyperparameter tuning
    # =============================================================================
    
    # TODO: Farklı learning rate'leri dene
    # TODO: Farklı batch size'ları test et
    # TODO: Farklı optimizer'ları karşılaştır (SGD vs Adam vs RMSprop)
    # TODO: Farklı aktivasyon fonksiyonlarını dene (ReLU, LeakyReLU, ELU)
    
    print("\n=== Tüm Eğitimler Tamamlandı! ===")
    print("Şimdi TODO'ları ve Challenge'ları tamamlamaya başlayabilirsin!")
    
    print("\n📝 Yapılacaklar Listesi:")
    print("1. Dropout'lu model (MNistClassifier_4) implement et")
    print("2. Daha derin model (MNistClassifier_5) oluştur")
    print("3. Batch Normalization ekle (MNistClassifier_6)")
    print("4. CNN modeli (MNistCNN) implement et")
    print("5. Accuracy tracking ekle")
    print("6. Model karşılaştırma sistemi yaz")
    print("7. Confusion matrix görselleştirmesi ekle")
    print("8. Hyperparameter tuning yap")
    
    print("\n🎯 Challenge'lar:")
    print("- En yüksek accuracy'yi elde etmeye çalış!")
    print("- Overfitting'i minimize et")
    print("- En hızlı converge olan modeli bul")
    print("- Farklı normalizasyon tekniklerini dene")