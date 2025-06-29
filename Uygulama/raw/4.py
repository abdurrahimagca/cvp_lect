
import torch
import torch.nn as nn
from torch.optim import SGD

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import idx2numpy

# =====================================================
# DOĞRUSAL REGRESYON UYGULAMASI
# =====================================================

print("=== DOĞRUSAL REGRESYON ===")

# Doğrusal Regresyon için Veri Seti Hazırlama
num_samples = 500
num_features = 2
X_np, y_np = datasets.make_regression(num_samples, num_features, noise=5, random_state=42)
y_np = np.expand_dims(y_np, axis=1)

print("X shape:", X_np.shape)
print("İlk 5 X:")
print(X_np[:5])
print("-" * 40)
print("y shape:", y_np.shape)
print("ilk 5 y:")
print(y_np[:5])

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# NumPy Dizilerinden PyTorch Tensörlerine Dönüştürme
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

# Doğrusal Regresyon Modeli Oluşturma
lineer_model = nn.Linear(num_features, 1)
loss_func = nn.MSELoss()
optimizer = SGD(lineer_model.parameters(), lr=0.001)

print("\nBaşlangıç parametreleri:")
for name, param in lineer_model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Model Eğitimi
print("\nModel eğitimi başlıyor...")
history = {"epoch": [], "loss": []}

for epoch in range(2000):
    y_hat = lineer_model(X_train)
    loss = loss_func(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())

print("Model eğitimi tamamlandı!")

# Eğitim sonrası parametreler
print("\nEğitim sonrası parametreler:")
for name, param in lineer_model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Eğitim Sürecinin Görselleştirilmesi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Lin Reg Model Loss")
plt.plot(history["epoch"], history["loss"], linewidth=3)
plt.xlabel("Epoch")
plt.ylabel("MSE")

# Test Veri Seti Üzerinde Model Performansı
y_predicted = lineer_model(X_test).detach().numpy()
plt.subplot(1, 2, 2)
plt.title("Lin Reg Testleri")
plt.plot(np.linspace(0, 10, 100), y_test.numpy(), label="real", c="r")
plt.plot(np.linspace(0, 10, 100), y_predicted, label="prediction", c="b")
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# LOJİSTİK REGRESYON UYGULAMASI
# =====================================================

print("\n=== LOJİSTİK REGRESYON ===")

# MNIST Veri Setini Okuma ve Hazırlama
MNIST_DIR = "mnist/"
try:
    train_arr = idx2numpy.convert_from_file(MNIST_DIR + "train-images-idx3-ubyte")
    train_labels = idx2numpy.convert_from_file(MNIST_DIR + "train-labels-idx1-ubyte")

    X_train = train_arr.reshape(60000, -1)
    X_train = X_train / 255.0
    y_train = np.copy(train_labels)

    # Sadece 3 ve 7 rakamlarını seçme
    X_3 = X_train[y_train == 3]
    y_3 = np.zeros(X_3.shape[0])

    X_7 = X_train[y_train == 7]
    y_7 = np.ones(X_7.shape[0])

    X_3_7 = np.append(X_3, X_7, axis=0)
    y_3_7 = np.append(y_3, y_7)

    print(f"3 rakamı örnekleri: {X_3.shape[0]}")
    print(f"7 rakamı örnekleri: {X_7.shape[0]}")
    print(f"Toplam örnekler: {X_3_7.shape[0]}")

    # Bazı örnekleri görselleştirme
    ds_check_indexes = [0, 1000, 5000, 5200, 6200, 11000, 12300, 12301, 12395]

    plt.figure(figsize=(9, 9))
    for i, index in enumerate(ds_check_indexes):
        plt.subplot(3, 3, i+1)
        plt.title(str(y_3_7[index]))
        plt.imshow(X_3_7[index].reshape(28, 28), cmap="gray")
    plt.tight_layout()
    plt.show()

    # Veri setini hazırlama
    y_3_7 = np.expand_dims(y_3_7, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_3_7, y_3_7, test_size=0.2, random_state=42)

    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Test shapes: {X_test.shape}, {y_test.shape}")

    # NumPy Dizilerini PyTorch Tensörlerine Dönüştürme
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Lojistik Regresyon Modeli Tanımlama
    class LogReg(nn.Module):
        def __init__(self, n_input_features):
            super().__init__()
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            y_lin = self.linear(x)
            y_hat = torch.sigmoid(y_lin)
            return y_hat

    print(f"Özellik sayısı: {X_train.shape[1]}")

    # Lojistik Regresyon Modelinin Eğitimi
    num_features = X_train.shape[1]
    logistic_model = LogReg(num_features)
    loss_func = nn.BCELoss()
    optimizer = SGD(logistic_model.parameters(), lr=0.005)

    print("\nLojistik regresyon eğitimi başlıyor...")
    history = {"epoch": [], "loss": []}
    for epoch in range(1000):
        y_hat = logistic_model(X_train)
        loss = loss_func(y_hat, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 20 == 0:
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())

    print("Lojistik regresyon eğitimi tamamlandı!")

    # Eğitim Sürecinin Görselleştirilmesi
    plt.figure(figsize=(5, 4))
    plt.title("Logistic Reg Model Loss")
    plt.plot(history["epoch"], history["loss"], linewidth=3)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.show()

    # Model parametrelerini görüntüleme (sadece şekil bilgisi)
    for name, param in logistic_model.named_parameters():
        if param.requires_grad:
            print(f"{name} şekli: {param.shape}")

    # Model Performansının Değerlendirilmesi
    with torch.no_grad():
        y_predicted = logistic_model(X_test).numpy()

    print(f"Tahmin şekli: {y_predicted.shape}")

    # Verileri düzleştirme ve sınıf etiketlerine dönüştürme
    y_predicted = np.squeeze(y_predicted)
    y_test = np.squeeze(y_test.numpy())

    y_predicted_classes = np.where(y_predicted > 0.5, 1, 0)

    # Doğruluk hesaplama
    acc = accuracy_score(y_test, y_predicted_classes)
    print(f"Test doğruluğu: {acc:.4f}")

except FileNotFoundError:
    print("MNIST veri seti bulunamadı. mnist/ klasöründe train-images-idx3-ubyte ve train-labels-idx1-ubyte dosyalarının olduğundan emin olun.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")

print("\n=== İşlemler tamamlandı! ===")
