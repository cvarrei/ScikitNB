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

[REVOIR AU CAS OU] 

```R
help(gnb$fit())

?gnb$fit()
```

## Datasets 

Three datasets are included within the package in order to test the different model:
- "numerical" - Numerical dataset: xxxxx (link) <---- Gaussian Naïve Bayes Classifier
- "categorical" - Categorical dataset: xxxxx (link) <---- Categorical Naïve Bayes Classifier
- "mixed" - Mixed (Numerical + Categorical) dataset: xxxxx (link) <---- Categorical & Gaussian Naïve Bayes Classifier
- "text" - Text Dataset <---- Bernouilli Naïve Bayes Classifier

To import a dataset, use the function : 

```R
dataset(mixed)
```

## GaussianNB: Gaussian Naïve Bayes Classifier

First, you have to create your object: 

```R
gnb <- GaussianNB$new(n_cores=4)
```
Because, the Gaussian Naïve Bayes Classifier has to work on quantitative data, you have to use the preprocess_train() function. If the original dataset includes only quantitative data, no preprocessing is done, if it includes mixed (categorical + numerical) features, it will realize a Factorial Mixed Data Analysis to project the feature into a factorial space and (if not specified) automatically keep the most informative components. 

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
```R
afdm$plot_diagonal_auto()
```
<img src="images/diagonalplot.png" width="800" height="500">

Other common plots can be displayed such as the scree plot: 
<img src="images/screeplot.png" width="600" height="300">
```R
afdm$plot_eigen()
```

Or the cumulated sum of the variance explained by the components: 
<img src="images/cumsumplot.png" width="600" height="300">

```R
afdm$plot_cumsumprop()
```
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



###  Brief and Detailed summary of the classifier characteristics anf performance on the training dataset

The generic function "print" and "summary" have been included to obtain, thus, each possibilities will work: 
```R
print(gnb)
```
```R
gnb$print()
```
```
The target variable has 2 classes. The sample has 303 observations.
The model has an error rate on the training sample of 0.172 [0.129;0.214] at 95%CI
```
```R
summary(gnb)
```
```R
gnb$summary()
```

```
############### GAUSSIAN NAIVE BAYES CLASSIFIER ################ 


Number of classes:  2 
Labels of classes: 0 1 
Number of features:  13 
Number of observations in the training sample:  303 
Prior of each class: 
        0         1 
0.4554455 0.5445545 


CLASSIFIER PERFORMANCE -------------- 

       predict_train
y_train   0   1
      0 101  37
      1  15 150


Error Rate = 0.172 [0.129;0.214] at 95%CI
Recall : 
   0    1 
0.73 0.91 
Precision : 
   0    1 
0.87 0.80 


CLASSIFICATION FUNCTIONS --------------

          0          1         F(1,301) P-Value     Signif.
Intercept -102.04727 -99.25775                             
age       0.72055    0.6683    16.1167  7.52480e-05 **     
sex       4.11622    2.80848   25.79219 6.67869e-07 **     
cp        0.55127    1.58576   69.77227 2.46971e-15 **     
trtbps    0.44484    0.42798   6.45817  1.15461e-02 **     
chol      0.09384    0.09053   2.20298  1.38790e-01        
fbs       1.25332    1.09588   0.23694  6.26778e-01        
restecg   1.6504     2.18182   5.77721  1.68399e-02 **     
thalachh  0.32143    0.36617   65.1201  1.69734e-14 **     
exng      3.07326    0.77787   70.95244 1.52081e-15 **     
oldpeak   1.43918    0.52922   68.55144 4.08535e-15 **     
slp       3.47825    4.75211   40.90207 6.10161e-10 **     
caa       1.31352    0.40941   54.55983 1.49154e-12 **     
thall     7.67007    6.39669   40.4077  7.62488e-10 **     

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

Then, we can assess how the model will classify the test dataset and compare it with the true classes: 

```R
gnb$score(X_test_sc, y_test)
```
<img src="images/score.png" width="300" height="250">
Several visualizations are included to better understand the predictive ability of the model: 

**Confusion Matrix**
```R
gnb$plot_confusionmatrix(X_test_sc, y_test)
```

<img src="images/cm.png" width="400" height="400">

**ROC Curve**

There are two possibilies with the ROC Curve plot: 
- If there are two classes, it is a simple ROC curve with its AUC which is displayed with the specified positive class:
```R
plot_roccurve(X_test_sc, y_test, positive="presence")
```
<img src="images/bin_roc.png" width="600" height="400">

- If there are more than two classes, it is a one-versus-rest multiclass ROC curve which is displayed. In brief, each class is compared with the rest and a ROC curve is displayed with corresponding AUC. Then, no "positive" argument is needed.

```R
plot_roccurve(X_test_sc, y_test)
```

<img src="images/multi_roc.png" width="600" height="400">

## Categorical Naïve Bayes

## Bernouilli Naïve Bayes
