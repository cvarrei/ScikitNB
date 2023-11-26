# ScikitNB: A Scikit-Learn-Inspired Naive Bayes Classifier for R

ScikitNB is an R package that implements three types of Naive Bayes Classifiers (Gaussian, Categorical, and Bernoulli) inspired by Scikit-Learn's machine learning algorithms. The package includes R6 objects for each model, providing a seamless integration with the R ecosystem. Additionally, the package includes visualization,  feature importance and feature selection tools, making it easy to interpret and understand the results. To optimize performance, parallelization has been integrated into the package

## Installation

To install the package, please use the following code lines: 

```R
library(devtools)

install_github("cvarrei/ScikitNB")
```

And then to import it: 
```R
library(ScikitNB)
```


## Documentation
In case of any difficulties using one of the function, you can refer to the documentation of the function with these two possibilities: 


```R
help(GaussianNB)

?GaussianNB
```

To view the documentation of the R6 class's functions, you must directly call it. Calling an R6 object function specifically ("gnb$fit(), assuming gnb <- GaussianNB$new()) will not work.

## Datasets 

Five datasets are included within the package in order to test the different models. The link refers to the source of the dataset:
- "numerical" - Numerical dataset ([Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)) <---- Gaussian Naïve Bayes Classifier
- "categorical" - Categorical dataset ([Link](https://www.kaggle.com/datasets/uciml/mushroom-classification)) <---- Categorical Naïve Bayes Classifier
- "mixed" - Mixed (Numerical + Categorical) dataset ([Link](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)) <---- Categorical & Gaussian Naïve Bayes Classifier
- "mixed_big" - Mixed (Numerical + Categorical) dataset ([Link](https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii)) <---- Categorical & Gaussian Naïve Bayes Classifier
- "text" - Text Dataset ([Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)) <---- Bernouilli Naïve Bayes Classifier

To import a dataset, use the function : 

```R
data("mixed")
```

## GaussianNB: Gaussian Naïve Bayes Classifier

The Gaussian Naive Bayes Classifier is suitable for numerical and mixed (qualitative and numerical datasets). 

First, you have to create your object: 

```R
gnb <- GaussianNB$new(n_cores=4)
```
Because, the Gaussian Naïve Bayes Classifier has to work on quantitative data, you have to use the preprocess_train() function. If the original dataset includes only quantitative data, no preprocessing is done, if it includes mixed (categorical + numerical) features, it will realize a Factorial Mixed Data Analysis to project the feature into a factorial space and (if not specified) automatically keep the most informative components. 

Be careful, it is necessary to activate the library "caret" before preprocessing mixed data: 
```R
library(caret)
```
```R
X_train_scaled <- gnb$preprocess_train(X_train)
```
### Factor analysis of mixed data (FAMD)

The GaussianNB objects inherit from another R6 object: "AFDM". This allows the classifier to manage automatically mixed data (categorical & numerical). It is based on the FactoMineR package. 
If you are interested in diving into the detail of the factorial projection, these are the different functions available within the AFDM object: 

This function initialize an instance of the AFDM object: 
```R
afdm <- AFDM$new()
```

This function will return the individual coordinates from the FAMD or in other words the transformed dataset into the factorial space. We can select the number of components we want to keep or we can let the function automatically try to detect the elbow: 
```R
# Explicitely choose the number of components
transformed_data <- afdm$fit_transform(data, n_compon = 5)
```
```R
# Automatically choose the number of components
transformed_data <- afdm$fit_transform(data)
```
This automatic detection of the elbow is based on the distance between each point of the curve of the eigen values and a straight line linking the highest and smallest eigen values. This can be visually observed using the following function:

Example on the "mixed" dataset.
```R
afdm$plot_diagonal_auto()
```
<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/diagonal.png" width="600" height="400">

Other common plots can be displayed such as the scree plot: 
Example on the "mixed" dataset.
```R
afdm$plot_eigen()
```
<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/screeplot.png" width="600" height="300">

Or the cumulated sum of the variance explained by the components: 
Example on the "mixed" dataset.
```R
afdm$plot_cumsumprop()
```
<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/cumsumplot.png" width="600" height="300">

The different attributes of the FAMD can be accessed with the function **afdm$get_attributes()**. 

Finally, we can project supplementary individuals onto the factorial space using: 
```R
X_test_sc <- afdm$coord_supplementaries(X_test)
```

The GaussianNB project use the fit_transform() and coord_supplementaries() function to internally preprocess train and test dataset. 

### Fit the training data
Then, you have to fit the model on your data: 
```R
gnb$fit(X_train_scaled, y_train)
```


###  Brief and Detailed summary on the training dataset

The generic function "print" and "summary" have been included to obtain, thus, each possibilities will work: 
```R
print(gnb)
```
```R
gnb$print()
```
```
The target variable has 2 classes. The sample has 538 observations.
The model has an error rate on the training sample of 0.264 [0.227;0.301] at 95%CI
```
```R
summary(gnb)
```
```R
gnb$summary()
```

This is an example on the "numerical" dataset.
```
################ GAUSSIAN NAIVE BAYES CLASSIFIER ################ 


Number of classes:  2 
Labels of classes: 0 1 
Number of features:  8 
Number of observations in the training sample:  538 
Prior of each class: 
        0         1 
0.6319703 0.3680297 


CLASSIFIER PERFORMANCE -------------- 

       predict_train
y_train   0   1
      0 278  62
      1  80 118


Error Rate = 0.264 [0.227;0.301] at 95%CI
Recall : 
   0    1 
0.82 0.60 
Precision : 
   0    1 
0.78 0.66 


CLASSIFICATION FUNCTIONS --------------

                         0         1         F(1,536)  P-Value     Signif.
Intercept                -29.87554 -41.49773                              
Pregnancies              0.2954    0.45203   33.13426  1.44890e-08 **     
Glucose                  0.1402    0.17872   147.04945 4.47450e-30 **     
BloodPressure            0.20298   0.20643   0.50945   4.75688e-01        
SkinThickness            0.07769   0.08934   4.29713   3.86541e-02 **     
Insulin                  0.00511   0.00732   8.05106   4.71991e-03 **     
BMI                      0.56189   0.64662   48.46997  9.84424e-12 **     
DiabetesPedigreeFunction 4.32151   5.3609    13.68443  2.38552e-04 **     
Age                      0.24066   0.28896   37.50045  1.76872e-09 **     

** : Statistically significant at p-value < 0.05
```


Using the following code line allows to get the results into a list: 

```R
results <- gnb$summary()
```

### Prediction on the test dataset

First, it is necessary to realize the same preprocessing than the one used and fitted on the training set. We will use the preprocessing_test() for this purpose. It will return a transformed version of the X_test dataset:

```R
X_test_sc <- gnb$preprocessing_test(X_test)
```

You can then extract the predicted class of the test sample:
```R
gnb$predict(X_test_sc)
```

The posterior probabilities and log-probabilities of each individual can be extracted with the following code lines: 

```R
gnb$predict_proba(X_test_sc)
```
```
                  0           1
  [1,] 9.208192e-01 0.079180780
  [2,] 8.006172e-01 0.199382818
  [3,] 9.620447e-01 0.037955263
  [4,] 4.958214e-01 0.504178566
  [5,] 8.613981e-02 0.913860191
  [6,] 9.261912e-01 0.073808840
  [7,] 9.118819e-06 0.999990881
  [8,] 1.642109e-01 0.835789130
  [9,] 8.842191e-01 0.115780871
 [10,] 9.480604e-01 0.051939586
etc.
```
```R
gnb$predict_log_proba(X_test_sc)
```
```
                  0             1
  [1,]  -0.082491549 -2.536022e+00
  [2,]  -0.222372371 -1.612529e+00
  [3,]  -0.038694326 -3.271347e+00
  [4,]  -0.701539429 -6.848248e-01
  [5,]  -2.451783614 -9.007768e-02
  [6,]  -0.076674629 -2.606277e+00
  [7,] -11.605170204 -9.118861e-06
  [8,]  -1.806603887 -1.793789e-01
  [9,]  -0.123050363 -2.156056e+00
 [10,]  -0.053337051 -2.957674e+00
etc.
```
Then, we can assess how the model will classify the test dataset and compare it with the true classes: 

```R
gnb$score(X_test_sc, y_test)
```
```
Confusion Matrix      y_pred
y_test   0   1
     0 136  24
     1  24  46


Error rate = 0.209 [0.156;0.261] CI 95%

Recall : 
   0    1 
0.85 0.66 
Precision : 
   0    1 
0.85 0.66 
```

Several visualizations are included to better understand the predictive ability of the model: 

**Confusion Matrix**

Example on the "numerical" dataset.
```R
gnb$plot_confusionmatrix(X_test_sc, y_test)
```

<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/cm.png" width="400" height="400">

**ROC Curve**

There are two possibilies with the ROC Curve plot: 
- If there are two classes, it is a simple ROC curve with its AUC which is displayed with the specified positive class:
Example on the "numerical" dataset.

```R
plot_roccurve(X_test_sc, y_test, positive=1)
```
<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/roc.png" width="600" height="400">

- If there are more than two classes, it is a one-versus-rest multiclass ROC curve which is displayed. In brief, each class is compared with the rest and a ROC curve is displayed with corresponding AUC. Then, no "positive" argument is needed.
Example on the "mixed_big" dataset.

```R
plot_roccurve(X_test_sc, y_test)
```

<img src="https://github.com/cvarrei/ScikitNB_Shiny/blob/main/images/readme_package/roc_multi.png" width="600" height="400">

## CategoricalNB: Categorical Naïve Bayes Classifier

The Categorical Naive Bayes classifier is suitable for classification with qualitative features. 

First, you have to create your object: 

```R
cnb <- CategoricalNB$new()
```

### Supervised discretization

Because the Categorical Naïve Bayes Classifier has to work on qualitative data, you have to always use the `preprocess()` function. If the original dataset includes only qualitative data, no preprocessing is done, if it includes mixed (categorical + numerical) features, it will realize a supervised discretization. It will return a transformed version of the dataset:

```R
X_train_d <- cnb$preprocessing_train(X_train, y_train)[[1]]
```

### Fit the training data

Then, you have to fit the model on your data. You can specify the Laplace smoothing parameter (`lmbd`) which is one by default.

```R
cnb$fit(X_train_d, y_train, lmbd=1)
```

###  Brief and Detailed summary on the training dataset

The generic functions `print()` and `summary()` have been overloaded, thus each of the following possibilities will work. 

For the `print()` function:

```R
print(cnb)
```
or 

```R
cnb$print()
```

```
The target variable has 2 classes. The sample has 5687 observations.

The model has an error rate on the training sample of 0.048 [0.042;0.054] at CI 95%
```


For the `summary()` function:

```R
summary(cnb)
```

or

```R
cnb$summary()
```

```
################ CATEGORICAL NAIVE BAYES CLASSIFIER ################ 

Number of classes:  2 
Labels of classes: e p 
Number of features:  22 
Number of observations in the validation sample:  5687 

CLASSIFIER PERFORMANCES ---------------------------------------
       
y_train    e    p
      e 2924   15
      p  258 2490

Error Rate = 0.048 [0.042;0.054] at CI 95%
Recall : 
   e    p 
0.99 0.91 
Precision : 
   e    p 
0.92 0.99 

CLASSIFIER CHARACTERISTICS -------------------------------------
Model coefficients -----------------------------------------------
      Descriptor                 e       p      
 [1,] constant                   -68.424 -76.663
 [2,] cap.shape_b                -1.524  -3.535 
 [3,] cap.shape_c                -7.208  -5.704 
 [4,] cap.shape_f                -0.196  -0.101 
 [5,] cap.shape_k                -2.096  -1.029 
 [6,] cap.shape_s                -4.03   -7.09  
 [7,] cap.surface_f              0.006   -0.883 
 [8,] cap.surface_g              -6.973  -5.512 
 [9,] cap.surface_s              -0.285  -0.217 
[10,] cap.color_b                -2.045  -1.727 

etc.

Conditionnal distribution : P(descriptor / class attribute) ----
                                      e            p
cap.shape_b                0.0998302207 0.0127087872
cap.shape_c                0.0003395586 0.0014524328
cap.shape_f                0.3769100170 0.3939724038
cap.shape_k                0.0563667233 0.1557734205
cap.shape_s                0.0081494058 0.0003631082
cap.shape_x                0.4584040747 0.4357298475
cap.surface_f              0.3645939517 0.1860465116
cap.surface_g              0.0003397893 0.0018168605
cap.surface_s              0.2725110432 0.3622819767
cap.surface_y              0.3625552158 0.4498546512
cap.color_b                0.0125466260 0.0300942712

etc.
```

### Prediction on the test dataset

First, it is necessary to realize the same preprocessing than the one used and fitted on the training set. We will use the preprocessing_test() for this purpose. It will return a transformed version of the X_test dataset:

```R
X_test_d <- cnb$preprocessing_test(X_test)
```

The predict(), score() and the the visualization functions (plot_confusionmatrix() and plot_roccurve()) of the CategoricalNB follow the same pattern as the Gaussian NB.


## BernoulliNB: Bernouilli Naïve Bayes Classifier

The Bernouilli Naive Bayes Classifier is more suitable for text data. You can use the "text" dataset included in the package. 

First, you have to create your object:

```R
bernoulli_nb <- BernoulliNB$new(n_cores=4)
```

The Bernoulli Naive Bayes Classifier has to perform text preprocessing on training and test data. The goal is to convert raw text data into a binary bag-of-words representation suitable for our tasks.

```R
preprocess_X = bernoulli_nb$preprocess(X_train, X_test)
X_train_vectorized = preprocess_X[[1]]
X_test_vectorized = preprocess_X[[2]]
```

### Fit the training data

Then, you have to fit the model on your data:

```R
bernoulli_nb$fit(X_train_vectorized, y_train)
```

### Brief and Detailed summary on the training dataset

The generic function "print" and "summary" have been included to obtain, thus, each possibilities will work:

```R
print(bernoulli_nb)
```

```R
bernoulli_nb$print()
```

```
The target variable has 2 classes. The sample has 3869 observations.
The model has an error rate on the training sample of 0.014 [0.011;0.018] at 95%CI
```

```R
summary(bernoulli_nb)
```

```R
bernoulli_nb$summary()
```

```
################ BERNOULLI NAIVE BAYES CLASSIFIER ################

Number of classes:  2
Labels of classes: ham spam
Number of features:  6600
Number of observations in the training sample:  3869
Prior probabilities for each class:
      ham      spam
0.8705092 0.1294908

CLASSIFIER PERFORMANCE --------------

predict_train  ham spam
         ham  3367   55
         spam    1  446

Error Rate = 0.014 [0.011;0.018] at 95%CI
Recall :
 ham spam
0.98 1.00
Precision :
 ham spam
1.00 0.89

LOG-PROBABILITIES --------------

The 10 tokens with the highest log-proba :
[1] "ham"

   Word Log_Probability
1   can       -2.535419
2  will       -2.654608
3   get       -2.747390
4   now       -2.804548
5  ltgt       -2.844553
6  just       -2.849668
7  dont       -2.854810
8  call       -2.963613
9  like       -3.016723
10  ill       -3.028918

[1] "spam"

     Word Log_Probability
1    call      -0.7695517
2    free      -1.2369835
3     now      -1.3530557
4     txt      -1.6458792
5   claim      -1.8017496
6  mobile      -1.8385635
7    text      -1.8767847
8    stop      -1.9031021
9   reply      -1.9720949
10  prize      -2.0462029
```

Using the following code line allows to get the results into a list:

```R
results <- bernoulli_nb$summary()
```

### Prediction on the test dataset

The predict(), score() and the the visualization functions (plot_confusionmatrix() and plot_roccurve()) of the BernoulliNB follow the same pattern as the Gaussian NB

### Words with Highest Log Probabilities

It is possible to display the top 10 tokens with the highest log_probability of a specific class from the target variable.

```R
spam_top_features = bernoulli_nb$top_logproba("spam")
```

```
Word Log_Probability
1    call      -0.7695517
2    free      -1.2369835
3     now      -1.3530557
4     txt      -1.6458792
5   claim      -1.8017496
6  mobile      -1.8385635
7    text      -1.8767847
8    stop      -1.9031021
9   reply      -1.9720949
10  prize      -2.0462029
```
