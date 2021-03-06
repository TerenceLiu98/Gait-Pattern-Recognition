# Gait-Pattern-Recognition

## Features 

In Raw data, we get four features:

* pressure 
* acceleration - x 
* acceleration - y
* acceleration - z

However, inside the raw data, the data is not linear, or the data is not obtained from sensors on people, but from sensors on the ground (tiles). There will be interference between tiles, that is, if the second tile will feel vibration (weak) when stepping on the first tile. Also, the sensor itself contains noise.


This is the trajectory:
```
(Group 1)-> [][][][][][][][][][][][]
            [][][][][][][][][][][][] <- (Group 2)
(Group 3)-> [][][][][][][][][][][][]
```
In this experiment, in total, 4 people have participated, and there were 3 groups of data for each person as shown in the figure above. Because some of the data is missing, we choose the common data of 4 individuals (Group 2).

```
# example of the data

[p_1, x_1, y_1, z_1]
.
.
.
[p_12, x_12, y_12, z_12]
```

## Pre-Processing


1. Data fusion: We regard these 12 tiles as one tile, and such a long tile has 12 pressure and three-dimensional accelerometer We can process these 12 * 4 sensors' data, and then get the subjects' motion trajectory, with these data, we can get their pressure waveform during walking and the three-dimensional acceleration waveform.

2. Low pass Filter: Butterworth filter


## Result

*(may have small different for each run)*


```
{SVM - Result}
acc = 99.35%
Accuracy: 1.00 (+/- 0.00) {5-fold validation}
              precision    recall  f1-score   support

         gao       1.00      1.00      1.00       178
          li       0.99      0.99      0.99       238
        wang       0.99      0.99      0.99       183
         yan       0.99      1.00      0.99       169

    accuracy                           0.99       768
   macro avg       0.99      0.99      0.99       768
weighted avg       0.99      0.99      0.99       768

{kNN (k = 3) - Result}
acc = 99.22%
Accuracy: 1.00 (+/- 0.01) {5-fold validation}
              precision    recall  f1-score   support

         gao       1.00      0.99      1.00       178
          li       0.99      0.99      0.99       238
        wang       0.99      0.99      0.99       183
         yan       0.99      1.00      0.99       169

    accuracy                           0.99       768
   macro avg       0.99      0.99      0.99       768
weighted avg       0.99      0.99      0.99       768
```


Repo Structure:
``` 
|───────────────────────────
│   .gitignore
│   main.py
│   README.md
│
├───data
│      G1.csv
│      L1.csv
│      W1.csv
│      Y1.csv
│   
├───model
│       KNN_Model.joblib
│       Scaler.joblib
│       SVM_Model.joblib
│
└───src
        extract_features.py
        json2csv.py
        model.py
```


load pre-trained model & the Scaler: 
```python
from joblib import load
SVM = load('model/SVM_Model.joblib')
KNN = load('model/KNN_Model.joblib')
scaler = load('model/Scaler.joblib')
```
do training with new data

```shell
python -B main.py
```

step of testing:

1. pre-processing: follow the steps in `src/extract_feature.py`
2. load sacler and the pre-train model
3. do the testing, e.g. `SVM.predict(X_test)`
