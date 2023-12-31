% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/GaussianNB.R
\name{GaussianNB}
\alias{GaussianNB}
\title{Gaussian Naive Bayes Classifier}
\description{
A R6 object to realize classification of numerical or mixed (numerical and qualitative) datasets using Gaussian Naive Bayes Classifier.
}
\examples{

## ------------------------------------------------
## Method `GaussianNB$new`
## ------------------------------------------------

# Initialize the Gaussian Naive Bayes Classifier with 2 cores and verbose output
gaussian_nb <- GaussianNB$new(n_cores = 2)


## ------------------------------------------------
## Method `GaussianNB$preprocessing_train`
## ------------------------------------------------

# Assuming'X_train' and 'y_train' are your explanatory and target variables respectively

X_train_sc <- gaussian_nb$preprocessing_train(X_train, y_train)


## ------------------------------------------------
## Method `GaussianNB$preprocessing_test`
## ------------------------------------------------

# Assuming 'X_test' is your test dataset

X_test_sc <- gaussian_nb$preprocessing_test(X_test)


## ------------------------------------------------
## Method `GaussianNB$fit`
## ------------------------------------------------

# Assuming 'X_train_sc' is your preprocessed training dataset

gaussian_nb$fit(X_train_sc)


## ------------------------------------------------
## Method `GaussianNB$predict`
## ------------------------------------------------

# Assuming 'X_test_sc' is your preprocessed test dataset
# Ensure that the model has been trained using fit method

y_pred <- gaussian_nb$predict(X_test_sc, type="comblin")


## ------------------------------------------------
## Method `GaussianNB$predict_log_proba`
## ------------------------------------------------

# Assuming 'X_test_sc' is your preprocessed test dataset

log_probs <- gaussian_nb$predict_log_proba(X_test_sc)


## ------------------------------------------------
## Method `GaussianNB$predict_proba`
## ------------------------------------------------

# Assuming 'X_test_sc' is your preprocessed test dataset

probs <- gaussian_nb$predict_proba(X_test_sc)


## ------------------------------------------------
## Method `GaussianNB$score`
## ------------------------------------------------

# Assuming 'X_test' and 'y_test' are your test dataset and labels
# Ensure that the model has been trained using fit method

results <- gaussian_nb$score(X_test, y_test)


## ------------------------------------------------
## Method `GaussianNB$print`
## ------------------------------------------------

# Assuming the Gaussian Naive Bayes model has been trained

gaussian_nb$print()

# OR

print(gaussian_nb)


## ------------------------------------------------
## Method `GaussianNB$summary`
## ------------------------------------------------

# Assuming the Gaussian Naive Bayes model has been trained

gaussian_nb$summary()

# OR

summary(gaussian_nb)


## ------------------------------------------------
## Method `GaussianNB$plot_confusionmatrix`
## ------------------------------------------------

# Assuming 'X_test_sc' and 'y_test' are your preprocessed test dataset and actual labels

gaussian_nb$plot_confusionmatrix(X_test_sc, y_test)


## ------------------------------------------------
## Method `GaussianNB$plot_roccurve`
## ------------------------------------------------

# Assuming 'X_test_sc' and 'y_test' are your preprocessed test dataset and actual labels
# This is the case of a binary classification

gaussian_nb$plot_roccurve(X_test_sc, y_test, positive = "YourPositiveClassLabel")

}
\section{Super class}{
\code{\link[ScikitNB:AFDM]{ScikitNB::AFDM}} -> \code{GaussianNB}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-GaussianNB-new}{\code{GaussianNB$new()}}
\item \href{#method-GaussianNB-preprocessing_train}{\code{GaussianNB$preprocessing_train()}}
\item \href{#method-GaussianNB-preprocessing_test}{\code{GaussianNB$preprocessing_test()}}
\item \href{#method-GaussianNB-fit}{\code{GaussianNB$fit()}}
\item \href{#method-GaussianNB-predict}{\code{GaussianNB$predict()}}
\item \href{#method-GaussianNB-predict_log_proba}{\code{GaussianNB$predict_log_proba()}}
\item \href{#method-GaussianNB-predict_proba}{\code{GaussianNB$predict_proba()}}
\item \href{#method-GaussianNB-score}{\code{GaussianNB$score()}}
\item \href{#method-GaussianNB-print}{\code{GaussianNB$print()}}
\item \href{#method-GaussianNB-summary}{\code{GaussianNB$summary()}}
\item \href{#method-GaussianNB-plot_confusionmatrix}{\code{GaussianNB$plot_confusionmatrix()}}
\item \href{#method-GaussianNB-plot_roccurve}{\code{GaussianNB$plot_roccurve()}}
\item \href{#method-GaussianNB-clone}{\code{GaussianNB$clone()}}
}
}
\if{html}{\out{
<details><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="coord_supplementaries"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-coord_supplementaries'><code>ScikitNB::AFDM$coord_supplementaries()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="fit_transform"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-fit_transform'><code>ScikitNB::AFDM$fit_transform()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="get_attributes"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-get_attributes'><code>ScikitNB::AFDM$get_attributes()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="get_var_characteristics"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-get_var_characteristics'><code>ScikitNB::AFDM$get_var_characteristics()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="plot_cumsumprop"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-plot_cumsumprop'><code>ScikitNB::AFDM$plot_cumsumprop()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="plot_diagonal_auto"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-plot_diagonal_auto'><code>ScikitNB::AFDM$plot_diagonal_auto()</code></a></span></li>
<li><span class="pkg-link" data-pkg="ScikitNB" data-topic="AFDM" data-id="plot_eigen"><a href='../../ScikitNB/html/AFDM.html#method-AFDM-plot_eigen'><code>ScikitNB::AFDM$plot_eigen()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-new"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-new}{}}}
\subsection{Method \code{new()}}{
This method initializes an instance of the Gaussian Naive Bayes Classifier.
It sets up the environment, including verifying and setting the number of processor cores
to be used for computation and optionally provides instructions for using the classifier.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$new(n_cores = 1, verbose = T)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{n_cores}}{The number of processor cores to use for calculations.
The method verifies if the specified number of cores is available.}

\item{\code{verbose}}{A logical value indicating whether to print detailed messages
about the initialization process. Default is TRUE.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
If the specified number of cores (`n_cores`) exceeds the number of available cores,
         the function throws an error. When `verbose` is TRUE, the function prints
         instructions on the console for preprocessing and using the classifier.
         The Gaussian Naive Bayes Classifier object inherit from the AFDM class,
         allowing for factor analysis of mixed data if necessary.
}

\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Initialize the Gaussian Naive Bayes Classifier with 2 cores and verbose output
gaussian_nb <- GaussianNB$new(n_cores = 2)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-preprocessing_train"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-preprocessing_train}{}}}
\subsection{Method \code{preprocessing_train()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$preprocessing_train(X_df, n_compon = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_df}}{A dataframe containing the explanatory variables.}

\item{\code{n_compon}}{The number of components for the FAMD. Default: NULL, the elbow will be automatically detect.}

\item{\code{y_vec}}{A vector containing the target variable.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The function performs several checks: it ensures that the sizes of `X_df` and `y_vec` match,
         that `X_df` is indeed a dataframe and not entirely composed of categorical variables,
         and that `y_vec` contains more than one class. The method handles mixed data by
         applying FAMD via the inherited `AFDM` class.
}

\subsection{Returns}{
A preprocessed dataframe suitable for training the Gaussian Naive Bayes Classifier.
        If the dataset includes mixed data (both categorical and numerical),
        a Factor Analysis of Mixed Data (FAMD) is applied and only numerical data is returned.
        If the data is purely numerical, it return the dataset.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming'X_train' and 'y_train' are your explanatory and target variables respectively

X_train_sc <- gaussian_nb$preprocessing_train(X_train, y_train)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-preprocessing_test"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-preprocessing_test}{}}}
\subsection{Method \code{preprocessing_test()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$preprocessing_test(X_test)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_test}}{A dataframe containing the test set explanatory variables.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The function checks the type of variables in 'X_test' and applies the appropriate
         preprocessing. For mixed data, it uses the 'coord_supplementaries'method
         from the inherited AFDM class to transform the data without additional training.
}

\subsection{Returns}{
A preprocessed dataframe suitable for making predictions with the Gaussian Naive Bayes Classifier.
        If the dataset includes mixed data (both categorical and numerical),
        a transformation is applied without retraining the model.
        For purely numerical data, it returns the dataset.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test' is your test dataset

X_test_sc <- gaussian_nb$preprocessing_test(X_test)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-fit"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-fit}{}}}
\subsection{Method \code{fit()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$fit(X_sc, y_vec, verbose = T)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{verbose}}{A logical value indicating whether to print detailed messages
about the training process. Default is TRUE.}

\item{\code{X}}{A preprocessed dataframe of explanatory variables used for training.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The method employs parallel computing for efficient calculations.
         Training performance is evaluated using confusion matrix, error rate,
         confidence intervals, recall, and precision for each class.
         These metrics are stored in the model's private fields.
}

\subsection{Returns}{
The method does not return anything but updates the model's internal state
        with the training results, including feature statistics and performance metrics.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_train_sc' is your preprocessed training dataset

gaussian_nb$fit(X_train_sc)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-predict"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-predict}{}}}
\subsection{Method \code{predict()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$predict(X_test, type = "comblin")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_test}}{A dataframe containing the preprocessed test set explanatory variables.}

\item{\code{type}}{A string specifying the prediction mode: 'comblin' for linear combination
or 'proba' for probability-based classification. Default is 'comblin'.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The method first verifies if the model has been trained with the `fit` method.
         In 'comblin' mode, it calculates the scores for each class using the learned
         coefficients and intercepts, and assigns each observation to the class with the
         highest score. In 'proba' mode, it assigns class labels based on the highest
         predicted probabilities.
}

\subsection{Returns}{
A vector of predicted class labels for each observation in the test dataset.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test_sc' is your preprocessed test dataset
# Ensure that the model has been trained using fit method

y_pred <- gaussian_nb$predict(X_test_sc, type="comblin")

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-predict_log_proba"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-predict_log_proba}{}}}
\subsection{Method \code{predict_log_proba()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$predict_log_proba(X_test)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_test}}{A dataframe containing the preprocessed test set explanatory variables.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
Before calculating the log probabilities, the method checks if the model
         has been trained. It uses parallel processing to calculate the log probabilities
         for efficiency.
         The values of our function differ slightly from those in scikit-learn because we employ the
         bias-corrected variance  divided by n - 1, whereas scikit-learn uses the variance divided by n
         (numpy's var function). This discrepancy does not impact the classification.
}

\subsection{Returns}{
A matrix of log probabilities, where each row corresponds to an observation
        and each column to a class. The log-probabilities are normalized using the logsumexp trick.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test_sc' is your preprocessed test dataset

log_probs <- gaussian_nb$predict_log_proba(X_test_sc)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-predict_proba"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-predict_proba}{}}}
\subsection{Method \code{predict_proba()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$predict_proba(X_test)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_test}}{A dataframe containing the preprocessed test set explanatory variables.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The method first checks if the model has been trained.
         It then calls the 'predict_log_proba' method to obtain the log probabilities
         and applies the exponential function to these values to get the posterior probabilities.
}

\subsection{Returns}{
A matrix of posterior probabilities, where each row corresponds to an observation
        and each column to a class. These probabilities are derived from the log probabilities
        computed by the 'predict_log_proba' method.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test_sc' is your preprocessed test dataset

probs <- gaussian_nb$predict_proba(X_test_sc)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-score"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-score}{}}}
\subsection{Method \code{score()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$score(X_test, y_test, type = "comblin", verbose = T)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X_test}}{A dataframe containing the test set explanatory variables.}

\item{\code{y_test}}{A vector containing the actual class labels of the test set.}

\item{\code{type}}{A string specifying the prediction mode: 'comblin' (default) for linear
combination, 'proba' for probability-based classification.}

\item{\code{verbose}}{A boolean indicating whether to print detailed output (default is TRUE).}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The method checks if the model has been trained, predicts class labels for the test set,
         constructs a confusion matrix, and calculates error rate, recall, and precision.
         It also computes a 95% confidence interval for the error rate.
         The results are returned as a list and can optionally be printed in detail.
}

\subsection{Returns}{
A list containing the confusion matrix, error rate, 95% confidence interval for the error rate,
        recall, and precision for each class.
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test' and 'y_test' are your test dataset and labels
# Ensure that the model has been trained using fit method

results <- gaussian_nb$score(X_test, y_test)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-print"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-print}{}}}
\subsection{Method \code{print()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$print()}\if{html}{\out{</div>}}
}

\subsection{Details}{
The method checks if the model has been trained and then prints
         the number of classes in the target variable, the number of observations
         in the sample, and the error rate on the training set along with its 95%
         confidence interval. This provides a quick overview of the model's training performance.
}

\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming the Gaussian Naive Bayes model has been trained

gaussian_nb$print()

# OR

print(gaussian_nb)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-summary"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-summary}{}}}
\subsection{Method \code{summary()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$summary(print_afdm = F)}\if{html}{\out{</div>}}
}

\subsection{Details}{
The method checks if the model has been trained.
         It prints details such as the number of classes, class labels, number of features,
         size of the training sample, prior probabilities of each class,
         classifier performance metrics (confusion matrix, error rate, recall, precision) on the training dataset,
         and the classification function (coefficients, intercepts, F-values, p-values).
         Statistical significance is indicated for features with p-value < 0.05.
         The summary provides a deep insight into the model's characteristics and performance.
}

\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming the Gaussian Naive Bayes model has been trained

gaussian_nb$summary()

# OR

summary(gaussian_nb)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-plot_confusionmatrix"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-plot_confusionmatrix}{}}}
\subsection{Method \code{plot_confusionmatrix()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$plot_confusionmatrix(X, y_test, type = "comblin")}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X}}{A dataframe containing the test set explanatory variables.}

\item{\code{y_test}}{A vector containing the actual class labels of the test set.}

\item{\code{type}}{A string specifying the prediction type: 'comblin' (default) for linear
combination based classification, 'proba' for probability based classification.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
The method first checks if the model has been trained.
         It then predicts the class labels for the test set using the specified prediction type
         and constructs a confusion matrix. This matrix is visualized using the 'corrplot' package,
         with a color gradient representing the frequency of each cell in the matrix. The method
         handles missing classes by adding zero-filled columns to ensure a complete matrix.
}

\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test_sc' and 'y_test' are your preprocessed test dataset and actual labels

gaussian_nb$plot_confusionmatrix(X_test_sc, y_test)

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-plot_roccurve"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-plot_roccurve}{}}}
\subsection{Method \code{plot_roccurve()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$plot_roccurve(X, y_test, positive = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{X}}{A dataframe containing the preprocessed test set explanatory variables.}

\item{\code{y_test}}{A vector containing the actual class labels of the test set.}

\item{\code{positive}}{An optional parameter specifying the positive class label in a binary classification scenario.
It is required when the model is trained on a binary classification task.}
}
\if{html}{\out{</div>}}
}
\subsection{Details}{
For multi-class classification (more than two classes), the method generates a One-vs-Rest (OVR)
         multi-class ROC curve. For binary classification, it plots a ROC curve for the specified positive class.
         The Area Under the Curve (AUC) is calculated for each class. The method checks if the model
         has been trained before generating the ROC curve.
}

\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{# Assuming 'X_test_sc' and 'y_test' are your preprocessed test dataset and actual labels
# This is the case of a binary classification

gaussian_nb$plot_roccurve(X_test_sc, y_test, positive = "YourPositiveClassLabel")

}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-GaussianNB-clone"></a>}}
\if{latex}{\out{\hypertarget{method-GaussianNB-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{GaussianNB$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
