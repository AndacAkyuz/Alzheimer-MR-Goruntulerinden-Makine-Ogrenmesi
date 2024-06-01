# Gerekli paketleri yükleme
from google.colab import drive
drive.mount('/content/drive')

zip_adres = "/content/drive/My Drive/archive.zip"
!cp "{zip_adres}" .

!unzip -q archive.zip

!pip install timm
!pip install torch torchvision
!pip install numpy pandas matplotlib seaborn scikit-learn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from timm import create_model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from PIL import Image

# Veri seti yolları
data_dir = '/content/Combined Dataset/train'
categories = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Özel veri seti sınıfı
class CustomDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transform işlemleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []

# PyTorch model sınıfı
class SwinModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(SwinModel, self).__init__()
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout katmanı eklendi

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)  # Dropout uygulandı
        return x

# Eğitim ve doğrulama fonksiyonları
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, accuracy, all_preds, all_labels

# Erken durdurma sınıfı
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Eğitim süreci
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(categories)

for fold, (train_index, val_index) in enumerate(kf.split(all_filepaths)):
    print(f"Fold {fold + 1}")

    train_filepaths = np.array(all_filepaths)[train_index]
    train_labels = np.array(all_labels)[train_index]
    val_filepaths = np.array(all_filepaths)[val_index]
    val_labels = np.array(all_labels)[val_index]

    train_dataset = CustomDataset(train_filepaths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_filepaths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SwinModel(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # weight decay eklendi
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 15  # Erken durdurma kullanılacağı için daha fazla epoch ayarlandı
    patience = 10  # Erken durdurma sabrı artırıldı
    early_stopping = EarlyStopping(patience=patience, delta=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_preds, val_labels = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)  # Öğrenme oranı azaltma

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Erken durdurma epoch {epoch + 1} 'de tetiklendi!")
            break

    # Performans metriklerini kaydetme
    precision = precision_score(val_labels, val_preds, average='weighted')
    recall = recall_score(val_labels, val_preds, average='weighted')
    f1 = f1_score(val_labels, val_preds, average='weighted')
    mcc = matthews_corrcoef(val_labels, val_preds)
    conf_matrix = confusion_matrix(val_labels, val_preds)
    specificity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    y_true_one_hot = np.eye(num_classes)[val_labels]
    auc = roc_auc_score(y_true_one_hot, np.eye(num_classes)[val_preds], multi_class='ovr')

    metrics_data = {
        'Fold': fold + 1,
        'Accuracy': val_accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity.mean(),  # Averaging specificities
        'F1-Score': f1,
        'MCC': mcc,
        'AUC': auc
    }

    all_metrics.append(metrics_data)

    # Loss vs Epoch grafiği
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} - Loss vs Epoch')
    plt.legend()
    plt.show()

    # Confusion Matrix ve ROC Eğrisi grafiği
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Fold {fold + 1} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], np.eye(num_classes)[val_preds][:, i])
        plt.plot(fpr, tpr, label=f'ROC curve class {categories[i]} (area = %0.2f)' % roc_auc_score(y_true_one_hot[:, i], np.eye(num_classes)[val_preds][:, i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold + 1} - Swin Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Modeli kaydetme
    torch.save(model.state_dict(), f'swin_model_fold_{fold + 1}.pth')
    print(f'Model fold {fold + 1} kaydedildi.')

# Tüm katmanların ortalama performans metrikleri
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
