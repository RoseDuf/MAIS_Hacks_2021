# MAIS 2021 Hackathon

## Generate the ASL dataset

Images used to generate the dataset come from [Akash's ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)

Executing `python database.py` will then genereate a `training.csv` file.
Using a maxium of 2800 training images per letter/sign, it was possible to generate [this dataset](https://drive.google.com/file/d/16cAQvTVGYrsoDzOqPR6zceIFB1uKm72G/view?usp=sharing).

## Training the ASL NN

With the previously generated dataset, 6 training algorithms (Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gaussian NB) were tried in order to obtain the highest accuracy from the testing set.

```
================================================
EVALUATING MODEL RANDOM FOREST CLASSIFIER
================================================
              precision    recall  f1-score   support

           a       0.97      0.97      0.97        95
           b       0.98      1.00      0.99       113
           c       0.95      1.00      0.97        97
           d       0.97      0.93      0.95       100
         del       0.95      0.96      0.96       106
           e       1.00      0.99      0.99        99
           f       0.99      0.98      0.98        96
           g       1.00      1.00      1.00        95
           h       1.00      1.00      1.00       103
           i       0.98      1.00      0.99        90
           j       1.00      0.97      0.98       100
           k       0.98      1.00      0.99        93
           l       0.98      0.96      0.97       113
           m       0.86      0.98      0.92        98
           n       0.95      0.91      0.93       104
           o       0.96      0.95      0.96       103
           p       1.00      0.98      0.99       100
           q       0.97      0.99      0.98        94
           r       0.99      0.98      0.99       110
           s       0.97      0.95      0.96        93
       space       0.98      0.97      0.98       111
           t       0.99      0.93      0.96       101
           u       0.95      0.93      0.94        85
           v       0.95      0.98      0.97       106
           w       0.99      1.00      1.00       109
           x       0.97      0.95      0.96        92
           y       0.97      0.98      0.97        87
           z       1.00      1.00      1.00       107

    accuracy                           0.97      2800
   macro avg       0.97      0.97      0.97      2800
weighted avg       0.97      0.97      0.97      2800
```