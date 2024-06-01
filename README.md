# MR Görüntülerinden Alzheimer Tespiti Makine Öğrenmesi

Bu proje, MR görüntülerinden Alzheimer hastalığını tespit etmek için kullanılan Dönüştürücü (Transformer) modellerinin uygulanmasını içermektedir. Çalışma, Google Colab üzerinde gerçekleştirilmiş ve çeşitli performans metrikleri ile değerlendirilmiştir.

##Kullanılan Veri Seti 

https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy


## İçindekiler

- [Özet](#özet)
- [Anahtar Kelimeler](#anahtar-kelimeler)
- [Giriş](#giriş)
- [Yöntem](#yöntem)
  - [Veri Hazırlığı](#veri-hazırlığı)
  - [Model Mimarisi](#model-mimarisi)
  - [Model Eğitim Parametreleri](#model-eğitim-parametreleri)
- [Sonuçlar](#sonuçlar)
  - [Google ViT-B16 Sonuçları](#google-vit-b16-sonuçları)
  - [Microsoft BeiT Sonuçları](#microsoft-beit-sonuçları)
  - [LeViT Sonuçları](#levit-sonuçları)
  - [DEViT Sonuçları](#devit-sonuçları)
  - [Swin Sonuçları](#swin-sonuçları)
- [Sonuç](#sonuç)
- [Kaynaklar](#kaynaklar)

## Özet

Bu raporda MR görüntülerinden Alzheimer hastalığını tespit etmek için kullanılan Dönüştürücü (Transformer) modellerinin uygulanması detaylandırılmaktadır. Bu çalışma Google Colab üzerinde gerçekleştirilmiş olup karışık hassasiyetli eğitim kullanılarak verimlilik artırılmış ve 5 katmanlı çapraz doğrulama ile performans değerlendirilmesi yapılmıştır. Performans metrikleri arasında doğruluk, kesinlik, geri çağırma, F1-skora, özgüllük, MCC ve AUC yer almaktadır. Veri artırma teknikleri modelin genelleme yeteneğini artırmak için kullanılmıştır.

## Anahtar Kelimeler

Alzheimer, MR Görüntüleri, Dönüştürücü Modeller, ViT, BeiT, DeiT, Swin, LeViT, Performans Metrikleri

## Giriş

Alzheimer hastalığının artan yaygınlığı, gelişmiş tanı araçlarını gerektirmektedir. Bu çalışmada çeşitli görüntü sınıflandırma görevlerinde üstün performans göstermiş olan Dönüştürücü (Transformer) modelleri kullanılmıştır. Amacımız MR görüntülerini dört kategoriye ayırmaktır: Hafif Bozukluk, Orta Bozukluk, Bozukluk Yok ve Çok Hafif Bozukluk.

## Yöntem

### Veri Hazırlığı

Veri seti dört sınıfa ayrılmış MR görüntülerinden oluşmaktadır. Veri artırma işlemleri modelin dayanıklılığını artırmak için “ImageDataGenerator” sınıfı kullanılarak gerçekleştirilmiştir. Veri model performansını doğrulamak için 5 katmanlı çapraz doğrulama kullanılarak bölünmüştür.

### Model Mimarisi

Bu çalışmada kullanılan modeller:

- Google ViT-B16
- Microsoft BeiT
- LeViT
- DEViT
- Swin

Her model önceden eğitilmiş ağırlıklarla kullanılmıştır. Modeller dört sınıfı içerecek şekilde son katmanı değiştirilerek veri setimiz üzerinde ince ayar yapılmıştır. Karışık hassasiyetli eğitim eğitim sürecini hızlandırmak için etkinleştirilmiştir.

### Model Eğitim Parametreleri

- Google ViT-B16
  - Epoch: 15
  - Batch Size: 32
  - Öğrenme Oranı: 0.0001

- Microsoft BeiT
  - Epoch: 10
  - Batch Size: 32
  - Öğrenme Oranı: 0.0001

- LeViT 384
  - Epoch: 20
  - Batch Size: 32
  - Öğrenme Oranı: 0.0001

- DEViT base_patch_16_224
  - Epoch: 10
  - Batch Size: 32
  - Öğrenme Oranı: 0.0001

- Swin base_patch4_window7_224
  - Epoch: 15
  - Batch Size: 32
  - Öğrenme Oranı: 0.0001

## Sonuç

Bu çalışma Dönüştürücü (Transformer) modellerinin tıbbi görüntü sınıflandırma görevlerinde özellikle Alzheimer hastalığının tespitinde etkinliğini göstermektedir. Gelecekteki çalışmalar daha derin model mimarilerini ve alternatif veri artırma tekniklerini araştırarak performansı daha da artırabilir.
