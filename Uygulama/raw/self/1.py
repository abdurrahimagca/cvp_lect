import torch
import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

import idx2numpy
import time

# =============================================================================
# TODO 1: Veri yükleme ve ön işleme - TamamlandÄ±!
# Challenge: Normalizasyonu değiştirmeyi dene (örn: 0-1 yerine -1 ile 1 arası)
# =============================================================================

MNIST_DIR = "../../mnist/"

def load_data_with_different_normalizations():
    """
    MNIST verisini farklı normalizasyon teknikleriyle yükler
    """
    print("📊 MNIST verisi yükleniyor...")
    X_mnist = idx2numpy.convert_from_file(MNIST_DIR + "train-images-idx3-ubyte")
    y_mnist = idx2numpy.convert_from_file(MNIST_DIR + "train-labels-idx1-ubyte")
    
    print(f"   • Orijinal veri boyutu: {X_mnist.shape}")
    print(f"   • Label sayısı: {len(np.unique(y_mnist))}")
    
    # Farklı normalizasyon teknikleri
    normalization_methods = {
        'standard': lambda x: x.reshape(60000, -1) / 255.0,  # 0-1 arası
        'centered': lambda x: (x.reshape(60000, -1) / 255.0) * 2.0 - 1.0,  # -1 ile 1 arası
        'z_score': lambda x: (x.reshape(60000, -1) - x.mean()) / x.std(),  # Z-score normalizasyonu
    }
    
    return X_mnist, y_mnist, normalization_methods

X_mnist, y_mnist, norm_methods = load_data_with_different_normalizations()

# Standart normalizasyon kullanıyoruz (isteğe göre değiştirilebilir)
X_mnist_normalized = norm_methods['standard'](X_mnist)

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist_normalized, y_mnist, test_size=0.2, random_state=42, stratify=y_mnist
)

print(f"✅ Veri bölümü tamamlandı:")
print(f"   • Eğitim seti: {X_train.shape[0]} örnek")
print(f"   • Test seti: {X_test.shape[0]} örnek")

class MnistDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0] 

# =============================================================================
# TODO 2: Eğitim fonksiyonunu geliştir - TamamlandÄ±!
# Challenge: Early stopping, learning rate scheduler ekle
# =============================================================================

def train_model(model, train_data_loader, test_data_loader, epochs=20, lr=0.01, 
                optimizer_type='SGD', early_stopping_patience=5, verbose=True):
    """
    Gelişmiş model eğitim fonksiyonu
    - Accuracy tracking
    - Early stopping
    - Multiple optimizer desteği
    - Learning rate scheduler
    """
    print(f"🚀 Model eğitimi başlıyor:")
    print(f"   • Optimizer: {optimizer_type}")
    print(f"   • Learning Rate: {lr}")
    print(f"   • Max Epochs: {epochs}")
    print(f"   • Early Stopping Patience: {early_stopping_patience}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer seçimi
    if optimizer_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Tracking listeleri
    train_loss = []
    validation_loss = []
    train_accuracies = []
    validation_accuracies = []
    
    # Early stopping için değişkenler
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # ============ Eğitim Fazı ============
        model.train()
        batch_train_loss = []
        correct_train = 0
        total_train = 0
        
        for X, y in train_data_loader:
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            batch_train_loss.append(loss.item())
            
            # Accuracy hesaplama
            _, predicted = torch.max(y_hat.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
        
        train_loss.append(np.array(batch_train_loss).mean())
        train_acc = 100 * correct_train / total_train
        train_accuracies.append(train_acc)
        
        # ============ Validation Fazı ============
        model.eval()
        with torch.no_grad():
            batch_validation_loss = []
            correct_val = 0
            total_val = 0
            
            for X, y in test_data_loader:
                y_hat = model(X)
                loss = criterion(y_hat, y)
                batch_validation_loss.append(loss.item())
                
                # Accuracy hesaplama
                _, predicted = torch.max(y_hat.data, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
            
            val_loss = np.array(batch_validation_loss).mean()
            val_acc = 100 * correct_val / total_val
            
            validation_loss.append(val_loss)
            validation_accuracies.append(val_acc)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Early stopping kontrolü
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress raporu
        if verbose and (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"   📈 Epoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
            print(f"      Train: Loss={train_loss[-1]:.4f}, Acc={train_acc:.2f}%")
            print(f"      Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"      LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"   ⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # En iyi modeli geri yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   ✅ En iyi model (val_loss: {best_val_loss:.4f}) geri yüklendi")
    
    total_time = time.time() - start_time
    print(f"   🏁 Eğitim tamamlandı! Süre: {total_time:.1f}s")
    
    return train_loss, validation_loss, train_accuracies, validation_accuracies

# =============================================================================
# CHALLENGE 1: Model mimarilerini geliştir - TamamlandÄ±!
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
# TODO 3: Yeni model sınıfları oluştur - TamamlandÄ±!
# Challenge: Aşağıdaki modelleri implement et
# =============================================================================

class MNistClassifier_4(nn.Module):
    """Dropout'lu model - Overfitting'i azaltır"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        print(f"   🎯 Model 4: Dropout rate = {dropout_rate}")
        
        self.input_layer = nn.Linear(784, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout1(x)  # %30 nöron rastgele kapatılır
        x = self.activation(self.hidden1(x))
        x = self.dropout2(x)  # Overfitting'i önler
        x = self.output_layer(x)
        return x

class MNistClassifier_5(nn.Module):
    """Daha derin model - 4 katmanlı ağ"""
    def __init__(self):
        super().__init__()
        print("   🏗️  Model 5: Derin ağ (784→256→128→64→10)")
        
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.layers(x)

class MNistClassifier_6(nn.Module):
    """Batch Normalization'lı model - Eğitimi hızlandırır"""
    def __init__(self):
        super().__init__()
        print("   ⚡ Model 6: Batch Normalization ile")
        
        self.input_layer = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
        self.hidden1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 10)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.bn1(self.input_layer(x)))
        x = self.activation(self.bn2(self.hidden1(x)))
        x = self.output_layer(x)
        return x

# =============================================================================
# CHALLENGE 2: CNN modeli oluştur - TamamlandÄ±!
# MNIST için convolutional neural network dene
# =============================================================================

class MNistCNN(nn.Module):
    """Convolutional Neural Network - En güçlü model"""
    def __init__(self):
        super().__init__()
        print("   🖼️  CNN Model: Conv2D + MaxPool + Fully Connected")
        
        # Convolutional katmanlar
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)  # Boyutu yarıya indirir
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 7x7x64 -> 128
        self.fc2 = nn.Linear(128, 10)  # 128 -> 10 (sınıf sayısı)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Input: (batch_size, 784) -> (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        
        # İlk conv bloğu: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        
        # İkinci conv bloğu: Conv -> ReLU -> Pool  
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        
        # Flatten: (batch, 64, 7, 7) -> (batch, 64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected katmanlar
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# =============================================================================
# Veri yükleyicileri
# =============================================================================

dataset_train = MnistDataset(X_train, y_train)
data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)

dataset_test = MnistDataset(X_test, y_test)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=256, shuffle=False)

print(f"📦 Data Loader'lar hazır:")
print(f"   • Batch size: 256")
print(f"   • Train batches: {len(data_loader_train)}")
print(f"   • Test batches: {len(data_loader_test)}")

# =============================================================================
# TODO 4: Görselleştirme fonksiyonlarını geliştir - TamamlandÄ±!
# =============================================================================

def plot_loss(train_loss, validation_loss, title="Model Performance"):
    """Loss grafiğini çiz - Geliştirilmiş versiyon"""
    plt.figure(figsize=(12, 5))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'b-', label="Train Loss", linewidth=2)
    plt.plot(validation_loss, 'r-', label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss farkı
    plt.subplot(1, 2, 2)
    loss_diff = np.array(train_loss) - np.array(validation_loss)
    plt.plot(loss_diff, 'g-', linewidth=2)
    plt.xlabel("Epoch") 
    plt.ylabel("Train Loss - Val Loss")
    plt.title("Overfitting Indicator")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_accuracy(train_acc, validation_acc, title="Model Accuracy"):
    """Accuracy grafiğini çizen fonksiyon - TamamlandÄ±!"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_acc, 'b-', label="Train Accuracy", linewidth=2, marker='o', markersize=4)
    plt.plot(validation_acc, 'r-', label="Validation Accuracy", linewidth=2, marker='s', markersize=4)
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title} - Accuracy Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # En yüksek accuracy'leri göster
    max_train = max(train_acc)
    max_val = max(validation_acc)
    plt.axhline(y=max_train, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=max_val, color='red', linestyle='--', alpha=0.5)
    
    plt.text(0.02, 0.98, f'Max Train: {max_train:.2f}%\nMax Val: {max_val:.2f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, data_loader, class_names=None):
    """Confusion matrix çizen fonksiyon - TamamlandÄ±!"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("🔍 Confusion matrix hesaplanÄ±yor...")
    
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    # Confusion matrix hesapla
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or range(10),
                yticklabels=class_names or range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Accuracy hesapla ve yazdır
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
    print(f"📊 Overall Accuracy: {accuracy:.2f}%")
    
    # Sınıf başına accuracy
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1) * 100
    print("📈 Sınıf başına accuracy:")
    for i, acc in enumerate(class_accuracies):
        print(f"   Sınıf {i}: {acc:.2f}%")
    
    return cm

def show_misclassified_examples(model, data_loader, num_examples=10):
    """Yanlış tahmin edilen digit'ları göster - TamamlandÄ±!"""
    model.eval()
    misclassified = []
    
    print(f"🔍 Yanlış sınıflandırılan {num_examples} örnek aranıyor...")
    
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            
            # Yanlış tahminleri bul
            wrong_indices = (predicted != y).nonzero(as_tuple=False).flatten()
            
            for idx in wrong_indices:
                if len(misclassified) >= num_examples:
                    break
                    
                misclassified.append({
                    'image': X[idx].cpu().numpy().reshape(28, 28),
                    'true_label': y[idx].item(),
                    'predicted_label': predicted[idx].item(),
                    'confidence': torch.softmax(outputs[idx], 0).max().item()
                })
            
            if len(misclassified) >= num_examples:
                break
    
    # Yanlış örnekleri görselleştir
    if misclassified:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)
        
        for i, example in enumerate(misclassified[:10]):
            row = i // 5
            col = i % 5
            
            axes[row, col].imshow(example['image'], cmap='gray')
            axes[row, col].set_title(
                f"Gerçek: {example['true_label']}\n"
                f"Tahmin: {example['predicted_label']}\n"
                f"Güven: {example['confidence']:.2f}",
                fontsize=10
            )
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ {len(misclassified)} yanlış örnek gösterildi")
    else:
        print("🎉 Hiç yanlış sınıflandırma bulunamadÄ±!")

# =============================================================================
# CHALLENGE 3: Model karşılaştırma sistemi - TamamlandÄ±!
# =============================================================================

def compare_models(models_dict, train_loader, test_loader, epochs=15):
    """Birden fazla modeli karşılaştır - TamamlandÄ±!"""
    print("🏆 Model Karşılaştırma Başlıyor!")
    print("=" * 50)
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n🚀 {model_name} eğitiliyor...")
        
        # Modeli eğit
        train_loss, val_loss, train_acc, val_acc = train_model(
            model, train_loader, test_loader, 
            epochs=epochs, optimizer_type='Adam', lr=0.001,
            verbose=False
        )
        
        # Sonuçları kaydet
        results[model_name] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'final_train_acc': train_acc[-1],
            'final_val_acc': val_acc[-1],
            'best_val_acc': max(val_acc),
            'model': model
        }
        
        print(f"   ✅ {model_name} tamamlandÄ±!")
        print(f"      Final Validation Accuracy: {val_acc[-1]:.2f}%")
        print(f"      Best Validation Accuracy: {max(val_acc):.2f}%")
    
    # Karşılaştırmalı grafikler
    plot_model_comparison(results)
    
    # Sonuç tablosu
    print_comparison_table(results)
    
    return results

def plot_model_comparison(results):
    """Model karşılaştırma grafikleri"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Karşılaştırması', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Training Loss
    axes[0, 0].set_title('Training Loss')
    for i, (name, result) in enumerate(results.items()):
        axes[0, 0].plot(result['train_loss'], label=name, 
                       color=colors[i % len(colors)], linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss') 
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0, 1].set_title('Validation Loss')
    for i, (name, result) in enumerate(results.items()):
        axes[0, 1].plot(result['val_loss'], label=name,
                       color=colors[i % len(colors)], linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Accuracy  
    axes[1, 0].set_title('Training Accuracy')
    for i, (name, result) in enumerate(results.items()):
        axes[1, 0].plot(result['train_acc'], label=name,
                       color=colors[i % len(colors)], linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[1, 1].set_title('Validation Accuracy')
    for i, (name, result) in enumerate(results.items()):
        axes[1, 1].plot(result['val_acc'], label=name,
                       color=colors[i % len(colors)], linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_comparison_table(results):
    """Sonuç tablosunu yazdır"""
    print("\n" + "="*80)
    print("📊 MODEL KARŞILAŞTIRMA SONUÇLARI")
    print("="*80)
    print(f"{'Model':<20} {'Final Train Acc':<15} {'Final Val Acc':<15} {'Best Val Acc':<15}")
    print("-"*80)
    
    # Sonuçları sırala (en iyi validation accuracy'ye göre)
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['best_val_acc'], 
                          reverse=True)
    
    for i, (name, result) in enumerate(sorted_results):
        symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{symbol} {name:<18} {result['final_train_acc']:>12.2f}% "
              f"{result['final_val_acc']:>12.2f}% {result['best_val_acc']:>12.2f}%")
    
    print("="*80)
    winner = sorted_results[0]
    print(f"🏆 KAZANAN: {winner[0]} (Best Val Acc: {winner[1]['best_val_acc']:.2f}%)")

# =============================================================================
# Ana eğitim kodu - Tüm modeller
# =============================================================================

if __name__ == "__main__":
    print("🎯 MNIST Sınıflandırıcı - Tüm TODO'lar TamamlandÄ±!")
    print("=" * 60)
    
    # Temel modeller
    models_to_train = {
        'Linear (Baseline)': MNistClassifier_1(),
        'Sigmoid Hidden': MNistClassifier_2(), 
        'ReLU Hidden': MNistClassifier_3(),
        'Dropout Model': MNistClassifier_4(dropout_rate=0.3),
        'Deep Model': MNistClassifier_5(),
        'BatchNorm Model': MNistClassifier_6(),
        'CNN Model': MNistCNN()
    }
    
    # Hızlı test (az epoch ile)
    print("\n🚀 Hızlı model karşılaştırması başlıyor...")
    results = compare_models(models_to_train, data_loader_train, data_loader_test, epochs=10)
    
    # En iyi modeli seç ve detaylı analiz
    best_model_name = max(results.keys(), key=lambda k: results[k]['best_val_acc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 En iyi model: {best_model_name}")
    print("🔍 Detaylı analiz yapılıyor...")
    
    # Confusion matrix
    plot_confusion_matrix(best_model, data_loader_test)
    
    # Yanlış örnekler
    show_misclassified_examples(best_model, data_loader_test, num_examples=10)
    
    # =============================================================================
    # CHALLENGE 4: Hyperparameter tuning - TamamlandÄ±!
    # =============================================================================
    
    print("\n⚙️  HYPERPARAMETER TUNING")
    print("=" * 40)
    
    # Farklı learning rate'leri test et
    learning_rates = [0.001, 0.01, 0.1]
    lr_results = {}
    
    for lr in learning_rates:
        print(f"\n🔧 Learning Rate: {lr} test ediliyor...")
        model = MNistClassifier_4()  # Dropout model kullan
        
        _, _, _, val_acc = train_model(
            model, data_loader_train, data_loader_test,
            epochs=8, lr=lr, optimizer_type='Adam', verbose=False
        )
        
        lr_results[f'LR_{lr}'] = max(val_acc)
        print(f"   Best Validation Accuracy: {max(val_acc):.2f}%")
    
    # Farklı optimizer'ları test et
    optimizers = ['SGD', 'Adam', 'RMSprop']
    opt_results = {}
    
    for opt in optimizers:
        print(f"\n🔧 Optimizer: {opt} test ediliyor...")
        model = MNistClassifier_4()
        
        _, _, _, val_acc = train_model(
            model, data_loader_train, data_loader_test,
            epochs=8, lr=0.001, optimizer_type=opt, verbose=False
        )
        
        opt_results[opt] = max(val_acc)
        print(f"   Best Validation Accuracy: {max(val_acc):.2f}%")
    
    # Hyperparameter sonuçları
    print("\n📈 HYPERPARAMETER SONUÇLARI:")
    print("-" * 40)
    print("Learning Rates:")
    for lr, acc in lr_results.items():
        print(f"  {lr}: {acc:.2f}%")
    
    print("\nOptimizers:")
    for opt, acc in opt_results.items():
        print(f"  {opt}: {acc:.2f}%")
    
    print("\n🎉 TÜM TODO'LAR VE CHALLENGE'LAR TAMAMLANDI!")
    print("=" * 60)
    
    print("\n✅ Tamamlanan Özellikler:")
    print("• Farklı normalizasyon teknikleri")
    print("• Gelişmiş eğitim fonksiyonu (early stopping, scheduler)")
    print("• Dropout modeli (overfitting önleme)")
    print("• Derin neural network")
    print("• Batch normalization")
    print("• Convolutional Neural Network (CNN)")
    print("• Accuracy tracking ve görselleştirme") 
    print("• Confusion matrix analizi")
    print("• Yanlış sınıflandırılan örnekleri gösterme")
    print("• Model karşılaştırma sistemi")
    print("• Hyperparameter tuning")
    print("• Detaylı logging ve açıklamalar")
    
    print(f"\n🏆 Final Recommendation:")
    print(f"En yüksek performans: {best_model_name}")
    print(f"Best Learning Rate: {max(lr_results, key=lr_results.get)} ({max(lr_results.values()):.2f}%)")
    print(f"Best Optimizer: {max(opt_results, key=opt_results.get)} ({max(opt_results.values()):.2f}%)")