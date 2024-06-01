# Bağlantıyı kurma ve gerekli paketleri yükleme
from google.colab import drive
drive.mount('/content/drive')

zip_adres = "/content/drive/My Drive/archive.zip"
!cp "{zip_adres}" .

!unzip -q archive.zip
#!rm archive.zip

!pip install vit-keras
!pip install tensorflow_addons


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from vit_keras import vit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

# Mixed precision kullanımı
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Veri seti yolları
data_dir = '/content/Combined Dataset/train'
categories = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# ImageDataGenerator ile veri yükleme ve artırma
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.2
)

# Dosya yollarını ve etiketleri elde etme
all_filepaths = []
all_labels = []
for i, category in enumerate(categories):
    category_dir = os.path.join(data_dir, category)
    for filename in os.listdir(category_dir):
        filepath = os.path.join(category_dir, filename)
        all_filepaths.append(filepath)
        all_labels.append(category)  # String formatında kategoriler

all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []


# Eğitim geçmişini kaydetme
import pickle

for fold, (train_index, val_index) in enumerate(kf.split(all_filepaths)):
    print(f"Fold {fold + 1}")

    train_filepaths = all_filepaths[train_index]
    train_labels = all_labels[train_index]
    val_filepaths = all_filepaths[val_index]
    val_labels = all_labels[val_index]

    train_gen = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_filepaths, 'class': train_labels}),
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=True
    )

    validation_gen = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_filepaths, 'class': val_labels}),
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )

    # ViT-B16 Modeli
    vit_model = vit.vit_b16(
        image_size=224,
        activation='softmax',
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=4
    )

    # Modeli derleme
    vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # ProgbarLogger callback'i ekleme
    callbacks = [
        tf.keras.callbacks.ProgbarLogger(count_mode='steps')
    ]

    # Modeli eğitme
    history_vit = vit_model.fit(
        train_gen,
        validation_data=validation_gen,
        epochs=15,
        verbose=1,
        callbacks=callbacks
    )

    # Performans metriklerini kaydetme
    y_pred_vit = vit_model.predict(validation_gen)
    y_pred_classes_vit = np.argmax(y_pred_vit, axis=1)
    y_true = validation_gen.labels

    accuracy_vit = accuracy_score(y_true, y_pred_classes_vit)
    precision_vit = precision_score(y_true, y_pred_classes_vit, average='weighted')
    recall_vit = recall_score(y_true, y_pred_classes_vit, average='weighted')
    f1_vit = f1_score(y_true, y_pred_classes_vit, average='weighted')
    mcc_vit = matthews_corrcoef(y_true, y_pred_classes_vit)
    conf_matrix_vit = confusion_matrix(y_true, y_pred_classes_vit)
    specificity_vit = np.diag(conf_matrix_vit) / np.sum(conf_matrix_vit, axis=1)
    y_true_one_hot_vit = tf.keras.utils.to_categorical(y_true, num_classes=4)
    auc_vit = roc_auc_score(y_true_one_hot_vit, y_pred_vit, multi_class='ovr')

    metrics_data_vit = {
        'Fold': fold + 1,
        'Accuracy': accuracy_vit,
        'Precision': precision_vit,
        'Recall (Sensitivity)': recall_vit,
        'Specificity': specificity_vit.mean(),  # Averaging specificities
        'F1-Score': f1_vit,
        'MCC': mcc_vit,
        'AUC': auc_vit
    }

    all_metrics.append(metrics_data_vit)

    # Confusion Matrix ve ROC Eğrisi grafiği
    sns.heatmap(conf_matrix_vit, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Fold {fold + 1} - ViT Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    for i in range(4):
        fpr, tpr, _ = roc_curve(y_true_one_hot_vit[:, i], y_pred_vit[:, i])
        plt.plot(fpr, tpr, label=f'ROC curve class {categories[i]} (area = %0.2f)' % roc_auc_score(y_true_one_hot_vit[:, i], y_pred_vit[:, i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold + 1} - ViT Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Modeli kaydetme
    vit_model.save(f'vit_model_fold_{fold + 1}.h5')
    print(f'Model fold {fold + 1} kaydedildi.')

    # Eğitim geçmişini kaydetme
    with open(f'history_fold_{fold + 1}.pkl', 'wb') as f:
        pickle.dump(history_vit.history, f)

    # Loss vs Epoch grafiği
    plt.plot(history_vit.history['loss'], label='Training Loss')
    plt.plot(history_vit.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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
