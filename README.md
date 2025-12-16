# dog-emotions-deep-learning

ilk sonuclar:

Toplam resim sayisi: 4000
Siniflar: ['angry', 'happy', 'relaxed', 'sad']
DogEmotionCNN(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc): Sequential(
    (0): Linear(in_features=100352, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=512, out_features=4, bias=True)
  )
)
Egitim basliyor...
Epoch 1/50 | Train Loss: 8.9528 Acc: 25.21% | Val Loss: 1.3679 Acc: 25.83%
  --> Model Kaydedildi (Loss düştü)
Epoch 2/50 | Train Loss: 1.3610 Acc: 24.86% | Val Loss: 1.3325 Acc: 32.00%
  --> Model Kaydedildi (Loss düştü)
Epoch 3/50 | Train Loss: 1.3478 Acc: 30.18% | Val Loss: 1.3121 Acc: 32.83%
  --> Model Kaydedildi (Loss düştü)
Epoch 4/50 | Train Loss: 1.3345 Acc: 31.39% | Val Loss: 1.3195 Acc: 32.83%
  --> İyileşme yok. Sabır: 1/5
Epoch 5/50 | Train Loss: 1.3224 Acc: 33.89% | Val Loss: 1.3370 Acc: 30.67%
  --> İyileşme yok. Sabır: 2/5
Epoch 6/50 | Train Loss: 1.3253 Acc: 31.29% | Val Loss: 1.3456 Acc: 29.17%
  --> İyileşme yok. Sabır: 3/5
Epoch 7/50 | Train Loss: 1.3207 Acc: 31.96% | Val Loss: 1.3239 Acc: 31.33%
  --> İyileşme yok. Sabır: 4/5
Epoch 8/50 | Train Loss: 1.3132 Acc: 31.68% | Val Loss: 1.2901 Acc: 32.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 9/50 | Train Loss: 1.2956 Acc: 32.39% | Val Loss: 1.2902 Acc: 33.00%
  --> İyileşme yok. Sabır: 1/5
Epoch 10/50 | Train Loss: 1.2930 Acc: 34.29% | Val Loss: 1.2913 Acc: 34.00%
  --> İyileşme yok. Sabır: 2/5
Epoch 11/50 | Train Loss: 1.2917 Acc: 34.39% | Val Loss: 1.2869 Acc: 34.17%
  --> Model Kaydedildi (Loss düştü)
Epoch 12/50 | Train Loss: 1.2839 Acc: 34.07% | Val Loss: 1.2915 Acc: 31.83%
  --> İyileşme yok. Sabır: 1/5
Epoch 13/50 | Train Loss: 1.2805 Acc: 35.21% | Val Loss: 1.2912 Acc: 30.67%
  --> İyileşme yok. Sabır: 2/5
Epoch 14/50 | Train Loss: 1.2727 Acc: 35.21% | Val Loss: 1.2862 Acc: 33.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 15/50 | Train Loss: 1.2730 Acc: 35.93% | Val Loss: 1.2939 Acc: 31.00%
  --> İyileşme yok. Sabır: 1/5
Epoch 16/50 | Train Loss: 1.2736 Acc: 36.61% | Val Loss: 1.2858 Acc: 32.50%
  --> Model Kaydedildi (Loss düştü)
Epoch 17/50 | Train Loss: 1.2704 Acc: 36.04% | Val Loss: 1.2900 Acc: 33.67%
  --> İyileşme yok. Sabır: 1/5
Epoch 18/50 | Train Loss: 1.2584 Acc: 36.96% | Val Loss: 1.2804 Acc: 33.50%
  --> Model Kaydedildi (Loss düştü)
Epoch 19/50 | Train Loss: 1.2618 Acc: 37.25% | Val Loss: 1.2805 Acc: 32.33%
  --> İyileşme yok. Sabır: 1/5
Epoch 20/50 | Train Loss: 1.2617 Acc: 36.11% | Val Loss: 1.2829 Acc: 35.00%
  --> İyileşme yok. Sabır: 2/5
Epoch 21/50 | Train Loss: 1.2406 Acc: 38.54% | Val Loss: 1.2852 Acc: 32.83%
  --> İyileşme yok. Sabır: 3/5
Epoch 22/50 | Train Loss: 1.2470 Acc: 36.36% | Val Loss: 1.2786 Acc: 33.50%
  --> Model Kaydedildi (Loss düştü)
Epoch 23/50 | Train Loss: 1.2455 Acc: 37.14% | Val Loss: 1.2745 Acc: 37.17%
  --> Model Kaydedildi (Loss düştü)
Epoch 24/50 | Train Loss: 1.2357 Acc: 38.25% | Val Loss: 1.2674 Acc: 33.83%
  --> Model Kaydedildi (Loss düştü)
Epoch 25/50 | Train Loss: 1.2343 Acc: 37.04% | Val Loss: 1.2617 Acc: 36.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 26/50 | Train Loss: 1.2260 Acc: 37.86% | Val Loss: 1.2653 Acc: 36.67%
  --> İyileşme yok. Sabır: 1/5
Epoch 27/50 | Train Loss: 1.2160 Acc: 38.21% | Val Loss: 1.2560 Acc: 38.17%
  --> Model Kaydedildi (Loss düştü)
Epoch 28/50 | Train Loss: 1.2102 Acc: 40.43% | Val Loss: 1.2566 Acc: 37.50%
  --> İyileşme yok. Sabır: 1/5
Epoch 29/50 | Train Loss: 1.2082 Acc: 38.64% | Val Loss: 1.2545 Acc: 39.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 30/50 | Train Loss: 1.1967 Acc: 40.21% | Val Loss: 1.2489 Acc: 39.83%
  --> Model Kaydedildi (Loss düştü)
Epoch 31/50 | Train Loss: 1.1832 Acc: 42.18% | Val Loss: 1.2447 Acc: 38.50%
  --> Model Kaydedildi (Loss düştü)
Epoch 32/50 | Train Loss: 1.1752 Acc: 41.43% | Val Loss: 1.2470 Acc: 39.83%
  --> İyileşme yok. Sabır: 1/5
Epoch 33/50 | Train Loss: 1.1606 Acc: 42.82% | Val Loss: 1.2440 Acc: 40.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 34/50 | Train Loss: 1.1553 Acc: 43.64% | Val Loss: 1.2371 Acc: 41.67%
  --> Model Kaydedildi (Loss düştü)
Epoch 35/50 | Train Loss: 1.1377 Acc: 43.89% | Val Loss: 1.2445 Acc: 39.83%
  --> İyileşme yok. Sabır: 1/5
Epoch 36/50 | Train Loss: 1.1433 Acc: 44.04% | Val Loss: 1.2483 Acc: 39.83%
  --> İyileşme yok. Sabır: 2/5
Epoch 37/50 | Train Loss: 1.1287 Acc: 45.71% | Val Loss: 1.2321 Acc: 41.00%
  --> Model Kaydedildi (Loss düştü)
Epoch 38/50 | Train Loss: 1.1194 Acc: 45.96% | Val Loss: 1.2418 Acc: 41.17%
  --> İyileşme yok. Sabır: 1/5
Epoch 39/50 | Train Loss: 1.1059 Acc: 46.61% | Val Loss: 1.2355 Acc: 42.00%
  --> İyileşme yok. Sabır: 2/5
Epoch 40/50 | Train Loss: 1.1027 Acc: 45.86% | Val Loss: 1.2590 Acc: 40.50%
  --> İyileşme yok. Sabır: 3/5
Epoch 41/50 | Train Loss: 1.0919 Acc: 47.86% | Val Loss: 1.2492 Acc: 42.00%
  --> İyileşme yok. Sabır: 4/5
Epoch 42/50 | Train Loss: 1.0660 Acc: 48.61% | Val Loss: 1.2553 Acc: 42.17%
  --> İyileşme yok. Sabır: 5/5
Early Stopping tetiklendi! Eğitim durduruluyor.
Eğitim tamamlandı!
c:\Users\MONSTER\Documents\GitHub\dog-emotions-deep-learning\main.py:236: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('best_model.pth'))
Classification Report:
              precision    recall  f1-score   support

       angry       0.45      0.15      0.22       172
       happy       0.39      0.51      0.45       136
     relaxed       0.36      0.50      0.42       142
         sad       0.42      0.47      0.44       150

    accuracy                           0.39       600
   macro avg       0.40      0.41      0.38       600
weighted avg       0.41      0.39      0.37       600

Accuracy (Doğruluk): $\%39$. Rastgele sallasak $\%25$ tuttururduk. Yani model bir şeyler öğreniyor ama yeterli değil.

Overfitting (Ezberleme): Train Accuracy yükselmeye devam ediyor ($\%48$'lere çıkmış), ama Validation Accuracy $\%40$'larda takılıp kalmış. Hatta Loss grafiğinde Validation Loss hafiften artmaya başlamış. Bu, modelin veriyi ezberlemeye başladığının kanıtı.

Sınıf Karışıklığı: Modelin en büyük sorunu 'Angry' (Kızgın) sınıfı ile. 172 kızgın köpekten sadece 25'ini bilmiş!Angry olanları en çok Sad (Üzgün) (74 adet) sanmış. 




model mimarisini cnn den transfer learning e cekince cikan sonuc:

Egitim basliyor...
Epoch 1/15 | Train Loss: 0.9513 Acc: 62.18% | Val Loss: 0.9473 Acc: 59.17%
Epoch 2/15 | Train Loss: 0.6418 Acc: 74.46% | Val Loss: 0.6845 Acc: 71.83%
Epoch 3/15 | Train Loss: 0.5303 Acc: 79.61% | Val Loss: 0.7852 Acc: 69.33%
Epoch 4/15 | Train Loss: 0.3990 Acc: 85.32% | Val Loss: 1.3053 Acc: 59.67%
Epoch 5/15 | Train Loss: 0.3210 Acc: 88.04% | Val Loss: 0.8570 Acc: 71.67%
Epoch 6/15 | Train Loss: 0.2363 Acc: 92.21% | Val Loss: 0.8097 Acc: 73.00%
Epoch 7/15 | Train Loss: 0.1861 Acc: 93.25% | Val Loss: 1.0097 Acc: 69.67%
Epoch 8/15 | Train Loss: 0.1741 Acc: 94.57% | Val Loss: 1.2212 Acc: 70.17%
Epoch 9/15 | Train Loss: 0.1860 Acc: 93.86% | Val Loss: 0.8110 Acc: 73.83%
Epoch 10/15 | Train Loss: 0.1761 Acc: 93.71% | Val Loss: 1.3423 Acc: 66.67%
Epoch 10/15 | Train Loss: 0.1761 Acc: 93.71% | Val Loss: 1.3423 Acc: 66.67%
Epoch 11/15 | Train Loss: 0.1590 Acc: 94.29% | Val Loss: 0.9954 Acc: 73.50%
Epoch 11/15 | Train Loss: 0.1590 Acc: 94.29% | Val Loss: 0.9954 Acc: 73.50%
Epoch 12/15 | Train Loss: 0.0864 Acc: 97.14% | Val Loss: 1.0824 Acc: 71.50%
Epoch 13/15 | Train Loss: 0.1098 Acc: 95.96% | Val Loss: 0.9572 Acc: 71.67%
Egitim tamamlandi!
Model kaydedildi.




overfittingi iyilestirmenin sonucu:

Egitim basliyor...
Epoch 1/20 | Train Loss: 0.8926 Acc: 62.96% | Val Loss: 0.4924 Acc: 80.50%
Epoch 2/20 | Train Loss: 0.3075 Acc: 89.04% | Val Loss: 0.4440 Acc: 83.67%
Epoch 3/20 | Train Loss: 0.1036 Acc: 97.18% | Val Loss: 0.4925 Acc: 84.67%
Epoch 4/20 | Train Loss: 0.0472 Acc: 99.29% | Val Loss: 0.4957 Acc: 84.17%
Epoch 5/20 | Train Loss: 0.0226 Acc: 99.86% | Val Loss: 0.4632 Acc: 85.50%
Epoch 6/20 | Train Loss: 0.0182 Acc: 99.82% | Val Loss: 0.4951 Acc: 85.00%
Epoch 7/20 | Train Loss: 0.0110 Acc: 99.93% | Val Loss: 0.4578 Acc: 86.00%
Epoch 8/20 | Train Loss: 0.0093 Acc: 99.93% | Val Loss: 0.4586 Acc: 86.00%
Epoch 9/20 | Train Loss: 0.0094 Acc: 99.93% | Val Loss: 0.4776 Acc: 86.67%
Epoch 10/20 | Train Loss: 0.0084 Acc: 99.96% | Val Loss: 0.4624 Acc: 86.83%
Epoch 11/20 | Train Loss: 0.0060 Acc: 100.00% | Val Loss: 0.4814 Acc: 86.17%
Epoch 12/20 | Train Loss: 0.0082 Acc: 99.96% | Val Loss: 0.4530 Acc: 86.33%
Epoch 13/20 | Train Loss: 0.0062 Acc: 100.00% | Val Loss: 0.4653 Acc: 86.83%
Epoch 14/20 | Train Loss: 0.0073 Acc: 99.96% | Val Loss: 0.4841 Acc: 86.00%
Epoch 15/20 | Train Loss: 0.0061 Acc: 99.96% | Val Loss: 0.4574 Acc: 86.33%
Epoch 16/20 | Train Loss: 0.0071 Acc: 100.00% | Val Loss: 0.4596 Acc: 86.50%
Epoch 17/20 | Train Loss: 0.0066 Acc: 100.00% | Val Loss: 0.4696 Acc: 86.50%
Epoch 18/20 | Train Loss: 0.0072 Acc: 99.96% | Val Loss: 0.4678 Acc: 86.67%
Epoch 19/20 | Train Loss: 0.0081 Acc: 99.96% | Val Loss: 0.4766 Acc: 86.50%
Epoch 20/20 | Train Loss: 0.0061 Acc: 100.00% | Val Loss: 0.4470 Acc: 86.50%
Egitim tamamlandi!
Model kaydedildi.


resnet freezıng ekledım

Egitim basliyor...
Epoch 1/20 | Train Loss: 1.3040 Acc: 39.57% | Val Loss: 1.1469 Acc: 47.50%
Epoch 2/20 | Train Loss: 1.1564 Acc: 48.46% | Val Loss: 0.9999 Acc: 55.67%
Epoch 3/20 | Train Loss: 1.1218 Acc: 50.86% | Val Loss: 1.0085 Acc: 56.00%
Epoch 4/20 | Train Loss: 1.0999 Acc: 52.50% | Val Loss: 0.9580 Acc: 59.67%
Epoch 5/20 | Train Loss: 1.0855 Acc: 53.18% | Val Loss: 0.9629 Acc: 64.33%
Epoch 6/20 | Train Loss: 1.0665 Acc: 53.46% | Val Loss: 0.9338 Acc: 62.00%
Epoch 7/20 | Train Loss: 1.0354 Acc: 55.79% | Val Loss: 0.9428 Acc: 63.83%
Epoch 8/20 | Train Loss: 1.0654 Acc: 54.14% | Val Loss: 0.9302 Acc: 62.67%
Epoch 9/20 | Train Loss: 1.0546 Acc: 54.86% | Val Loss: 0.9689 Acc: 61.00%
Epoch 10/20 | Train Loss: 1.0450 Acc: 54.93% | Val Loss: 0.9688 Acc: 61.67%
Epoch 11/20 | Train Loss: 1.0230 Acc: 55.82% | Val Loss: 0.9063 Acc: 62.50%
Epoch 12/20 | Train Loss: 1.0318 Acc: 54.71% | Val Loss: 0.9487 Acc: 62.33%
Epoch 13/20 | Train Loss: 1.0361 Acc: 55.50% | Val Loss: 0.9211 Acc: 63.50%
Epoch 14/20 | Train Loss: 1.0251 Acc: 55.43% | Val Loss: 0.9291 Acc: 62.67%
Epoch 15/20 | Train Loss: 1.0066 Acc: 57.61% | Val Loss: 0.8997 Acc: 64.17%
Epoch 16/20 | Train Loss: 1.0153 Acc: 55.96% | Val Loss: 0.9156 Acc: 63.50%
Epoch 17/20 | Train Loss: 1.0152 Acc: 56.54% | Val Loss: 0.9409 Acc: 62.67%
Epoch 18/20 | Train Loss: 1.0153 Acc: 56.25% | Val Loss: 0.8873 Acc: 65.67%
Epoch 19/20 | Train Loss: 0.9890 Acc: 58.29% | Val Loss: 0.9275 Acc: 62.50%
Epoch 20/20 | Train Loss: 1.0043 Acc: 56.11% | Val Loss: 0.9301 Acc: 62.17%
Egitim tamamlandi!
Model kaydedildi.

