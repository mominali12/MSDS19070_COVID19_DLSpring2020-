# COVID-19 Detection via X-Ray images
## Description
This repository contains code and results for COVID-19 classification assignment by Deep Learning Spring 2020 course offered at Information Technology University, Lahore, Pakistan. This assignment is only for learning purposes and is not intended to be used for clinical purposes.

## Dependencies
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
```

## Dataset
We have two different types of experiements :
```python
1. COVID19 Classification part-1
2. COIVD19 Classification using Focal Loss to cater Samples Imbalance
```
Both the experiments utilize different dataset .

a) For COVID19 Classification part1 we have used an open source [data set 1](https://drive.google.com/drive/u/1/folders/1-FzZhQO9oHIT9SNOWYoKsuz7fe447vtR) of X-Ray images <br>

b) For COIVD19 Classification using Focal Loss we have used [data set 2](https://drive.google.com/open?id=1eytbwaLQBv12psV8I-aMkIli9N3bf8nO) <br>

In each dataset the data is divided into 3 parts:
```python
1. Train data
2. Test data
3. Validation data
```
Follwing are the details of Data set: <br>

![](Images/dataset_details.JPG)

Chest X-Ray images are taken in different views (AP or PA) depending on which side of the body is facing the X-Ray scanner. Images from different views have slightly different features. For this task, we will be using images without considering their views. A few sample images: <br><br>

![](Images/sample_images.JPG)

## Experiments Performed on Dataset-1
Case|Model|CNN layers|FC layers
-----|-----|----------|---------|
----------|----------|requires_grad|requires_grad
case1|VGG-16|all False|all True
case2|ResNet-18|all False|all True
case3|VGG-16|all True|all True
case4|ResNet-18|all True|all True

```python
1. Transfer learning on VGG-16 trained for ImageNet by freezing all CNN layers and replacing FC layers with new FC layers.
2. Transfer learning on ResNet-18 trained for ImageNet by freezing all CNN layers and replacing FC layers with new FC layers.
3. Transfer learning on VGG-16 trained for ImageNet by unfreezing all layers and replacing FC layers with  new FC layers.
4. Transfer learning on ResNet-18 trained for ImageNet by unfreezing all layers and replacing FC layers with new FC layers.
```
<br>

## Classification Results
<br>

```python
CASE 1:  (VGG-16 FC Layer Only)
```
**Accuracy**
Data Split|Accuracy
-|-|
test data|94%
train data|88.81
validation data|88.26

**Confusion Matrix Training data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|3901|1018|
actual(normal)|287|5565|

<br>

**Confusion Matrix Validation data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|510|105|
actual(normal)|71|814|

<br>

**Confusion Matrix Testing data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|553|62|
actual(normal)|17|868|

<br>

```python
CASE 2: (ResNet-18 FC Layer Only)
```

**Accuracy**
Data|Accuracy
---------|---------|
test data|93%
train data|85.99
validation data|86.53

**Confusion Matrix Training data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|4220|699|
actual(normal)|737|5115|

<br>

**Confusion Matrix Validation data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|548|67|
actual(normal)|135|750|

<br>

**Confusion Matrix Testing data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|566|49|
actual(normal)|56|829|

<br>

```python
CASE 3: (VGG-16 All Layers )
```

**Accuracy**
Data|Accuracy
---------|---------|
test data|96%
train data|92.3%
validation data|91.4%

**Confusion Matrix Training data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|4289|630|
actual(normal)|238|6073|

<br>

**Confusion Matrix Validation data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|532|83|
actual(normal)|46|839|

<br>

**Confusion Matrix Testing data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|578|37|
actual(normal)|9|876|

<br>

```python
CASE 4: (ResNet-18 All Layers)
```

**Accuracy**
Data|Accuracy
---------|---------|
test data|96%
train data|91.57
validation data|86.53

**Confusion Matrix Training data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|4178|741|
actual(normal)|194|5658|

<br>

**Confusion Matrix Validation data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|535|80|
actual(normal)|59|826|

<br>

**Confusion Matrix Testing data**
N|prediction(infected)|predicted(normal)
---------|---------|---------|
actual(infected)|575|40|
actual(normal)|15|870|

<br>

## Experiments Performed on Dataset-2

Case|Model|Without Focal Loss|With Focal Loss
-----|-----|----------|---------|
----------|----------|requires_grad|requires_grad
case1|VGG-16|all layers True|all layers True
case2|ResNet-18|all layers True|all layers True

```python
1. Transfer learning on VGG-16 trained for ImageNet by unfreezing all CNN layers and replacing FC layers with new FC layers without focal loss.
2. Transfer learning on ResNet-18 trained for ImageNet by freezing all CNN layers and replacing FC layers with new FC layers without focal loss.
3. Transfer learning on VGG-16 trained for ImageNet by unfreezing all layers and replacing FC layers with  new FC layers with focal loss.
4. Transfer learning on ResNet-18 trained for ImageNet by unfreezing all layers and replacing FC layers with new FC layers with focal loss..
```

<br>

## Classification Results
<br>

### Without Focal Loss

<br>

**For ResNet18 without Focal Loss (Training Data)**

                 precision    recall  f1-score   support

               0       1.00      0.72      0.84       200
               1       0.99      0.94      0.97      4000
               2       0.87      0.99      0.93      2000

        accuracy                           0.95      6200
       macro avg       0.96      0.89      0.91      6200
    weighted avg       0.96      0.95      0.95      6200
 
 <br>
 
 **For ResNet18 without Focal Loss (Validation Data)**
 
                  precision    recall  f1-score   support

               0       0.93      0.50      0.65        28
               1       0.96      0.90      0.93       400
               2       0.79      0.94      0.86       200

        accuracy                           0.89       628
       macro avg       0.89      0.78      0.81       628
    weighted avg       0.90      0.89      0.89       628

<br>

**For VGG16 without Focal Loss (Training Data)**

                  precision    recall  f1-score   support

               0       0.99      0.88      0.93       200
               1       0.98      1.00      0.99      4000
               2       0.99      0.96      0.97      2000

        accuracy                           0.98      6200
       macro avg       0.99      0.95      0.97      6200
    weighted avg       0.98      0.98      0.98      6200

<br>

**For VGG16 without Focal Loss (Validation Data)**

                  precision    recall  f1-score   support

               0       1.00      0.61      0.76        28
               1       0.94      0.98      0.96       400
               2       0.94      0.91      0.92       200

        accuracy                           0.94       628
       macro avg       0.96      0.83      0.88       628
    weighted avg       0.94      0.94      0.94       628
    
<br>

### With Focal Loss <br>
**For ResNet18 with Focal Loss (Training Data)**

                  precision    recall  f1-score   support

               0       0.96      0.88      0.92       200
               1       0.97      0.98      0.98      4000
               2       0.96      0.95      0.95      2000

        accuracy                           0.97      6200
       macro avg       0.96      0.94      0.95      6200
    weighted avg       0.97      0.97      0.97      6200

<br>

**For ResNet18 with Focal Loss (Validation Data)**

                  precision    recall  f1-score   support

               0       1.00      0.64      0.78        28
               1       0.93      0.95      0.94       400
               2       0.89      0.89      0.89       200

        accuracy                           0.92       628
       macro avg       0.94      0.83      0.87       628
    weighted avg       0.92      0.92      0.92       628

<br>

**For VGG16 with Focal Loss (Training Data)**

                  precision    recall  f1-score   support

               0       0.96      0.82      0.89       200
               1       0.97      0.99      0.98      4000
               2       0.97      0.95      0.96      2000

        accuracy                           0.97      6200
       macro avg       0.97      0.92      0.94      6200
    weighted avg       0.97      0.97      0.97      6200
    
<br>

**For VGG16 with Focal Loss (Validation Data)**

                  precision    recall  f1-score   support

               0       0.94      0.57      0.71        28
               1       0.94      0.95      0.95       400
               2       0.88      0.91      0.89       200

        accuracy                           0.92       628
       macro avg       0.92      0.81      0.85       628
    weighted avg       0.92      0.92      0.92       628
    
<br>


Please find the links to fine tuned weights for different models:
## Part 1
1. [VGG-16 FC layer Only ](https://drive.google.com/open?id=1G5xKYoc8hRypjmjImlyX4xc-e2asyMsm)
2. [ResNet-18 FC layer Only ](https://drive.google.com/open?id=1BmEuMgLekNIuDYvmnUDW4PJBDl7ON5Sc)
3. [VGG-16 All layers ](https://drive.google.com/open?id=1t3FihFki06-1r7pgccD2gm9VVXdOQltq)
4. [ResNet-18 All layers ](https://drive.google.com/open?id=1-EKMaQjdWpUC-M04a7A0vshtxkwhgakw)

## Part 2
1. [VGG-16 Without Focal Loss](https://drive.google.com/open?id=1--1T75uyhNGEVINuDrx2Rkpc1OeX7Zdr)
2. [ResNet-18 Without Focal Loss](https://drive.google.com/open?id=19zvuexsZY0TBas1G48rchZ1gipei7oUh)
3. [VGG-16 With Focal Loss](https://drive.google.com/open?id=1--oVgQp3Wf1ZmtLBxc83WdYNBznAGgbj)
4. [ResNet-18 With Focal Loss](https://drive.google.com/open?id=1o8bKKwvngbnuApEX9qaQVZnN47UDIJUx)

