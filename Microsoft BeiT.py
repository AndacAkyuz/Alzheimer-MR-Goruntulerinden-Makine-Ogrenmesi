# Bağlantıyı kurma ve gerekli paketleri yükleme
from google.colab import drive
drive.mount('/content/drive')

zip_adres = "/content/drive/My Drive/archive.zip"
!cp "{zip_adres}" .

!unzip -q archive.zip
#!rm archive.zip

!pip install torch torchvision torchaudio
!pip install transformers
!pip install tensorflow_addons

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from transformers import BeitForImageClassification, BeitFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from PIL import Image

# Mixed precision kullanımı
from torch.cuda.amp import autocast, GradScaler

# Veri seti yolları
data_dir = '/content/Combined Dataset/train'
categories = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# ImageDataGenerator yerine PyTorch DataLoader ile veri yükleme ve artırma
class CustomDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dosya yollarını ve etiketleri elde etme
all_filepaths = []
all_labels = []
for i, category in enumerate(categories):
    category_dir = os.path.join(data_dir, category)
    for filename in os.listdir(category_dir):
        filepath = os.path.join(category_dir, filename)
        all_filepaths.append(filepath)
        all_labels.append(i)  # Integer formatında kategoriler

all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []

# Eğitim geçmişini kaydetme
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold, (train_index, val_index) in enumerate(kf.split(all_filepaths)):
    print(f"Fold {fold + 1}")

    train_filepaths = all_filepaths[train_index]
    train_labels = all_labels[train_index]
    val_filepaths = all_filepaths[val_index]
    val_labels = all_labels[val_index]

    train_dataset = CustomDataset(train_filepaths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_filepaths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Beit Modeli
    model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k", num_labels=4, ignore_mismatched_sizes=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scaler = GradScaler()

    # Eğitim fonksiyonu
    def train_epoch(model, dataloader, optimizer, scaler):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images).logits
                loss = torch.nn.functional.cross_entropy(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # Doğrulama fonksiyonu
    def validate_epoch(model, dataloader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images).logits
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy, all_labels, all_preds

    # Eğitim döngüsü
    num_epochs = 15
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_acc, val_labels, val_preds = validate_epoch(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Performans metriklerini kaydetme
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, average='weighted')
    recall = recall_score(val_labels, val_preds, average='weighted')
    f1 = f1_score(val_labels, val_preds, average='weighted')
    mcc = matthews_corrcoef(val_labels, val_preds)
    conf_matrix = confusion_matrix(val_labels, val_preds)
    specificity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    auc = roc_auc_score(torch.nn.functional.one_hot(torch.tensor(val_labels), num_classes=4).numpy(), torch.nn.functional.one_hot(torch.tensor(val_preds), num_classes=4).numpy(), multi_class='ovr')

    metrics_data = {
        'Fold': fold + 1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity.mean(),  # Averaging specificities
        'F1-Score': f1,
        'MCC': mcc,
        'AUC': auc
    }

    all_metrics.append(metrics_data)

    # Confusion Matrix ve ROC Eğrisi grafiği
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Fold {fold + 1} - Beit Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    for i in range(4):
        fpr, tpr, _ = roc_curve(torch.nn.functional.one_hot(torch.tensor(val_labels), num_classes=4).numpy()[:, i], torch.nn.functional.one_hot(torch.tensor(val_preds), num_classes=4).numpy()[:, i])
        plt.plot(fpr, tpr, label=f'ROC curve class {categories[i]} (area = %0.2f)' % roc_auc_score(torch.nn.functional.one_hot(torch.tensor(val_labels), num_classes=4).numpy()[:, i], torch.nn.functional.one_hot(torch.tensor(val_preds), num_classes=4).numpy()[:, i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold + 1} - Beit Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Modeli kaydetme
    torch.save(model.state_dict(), f'beit_model_fold_{fold + 1}.pth')
    print(f'Model fold {fold + 1} kaydedildi.')

    # Eğitim geçmişini kaydetme
    with open(f'history_fold_{fold + 1}.pkl', 'wb') as f:
        pickle.dump({'loss': train_losses, 'val_loss': val_losses}, f)

    # Loss vs Epoch grafiği
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Tüm katmanların performans metriklerini hesaplama
metrics_df = pd.DataFrame(all_metrics)
average_metrics = metrics_df.mean(numeric_only=True)
print("5-Fold Cross-Validation Average Metrics:\n", average_metrics)

# Ortalama performans metriklerini tablo formatında raporlama
plt.figure(figsize=(10, 6))
plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')
plt.axis('off')
plt.title('5-Fold Cross-Validation Metrics')
plt.show()

# Ortalama performans metriklerini kaydetme
average_metrics.to_csv('average_metrics.csv', index=False)
metrics_df.to_csv('all_folds_metrics.csv', index=False)
print("Ortalama performans metrikleri kaydedildi.")
print("Tüm katmanların metrikleri kaydedildi.")

# Eğitim geçmişini yükleme ve ortalama loss hesaplama
losses = []
val_losses = []

for i in range(5):
    with open(f'history_fold_{i + 1}.pkl', 'rb') as f:
        history = pickle.load(f)
    losses.append(history['loss'])
    val_losses.append(history['val_loss'])

average_loss = np.mean(losses, axis=0)
average_val_loss = np.mean(val_losses, axis=0)

plt.plot(average_loss, label='Average Training Loss')
plt.plot(average_val_loss, label='Average Validation Loss')
plt.title('Average Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
