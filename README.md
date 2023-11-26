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
data(mixed)
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
