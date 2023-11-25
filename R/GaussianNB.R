GaussianNB <- R6Class("GaussianNB",
                      inherit = AFDM, # Ancestor class to create the AFDM if necessary
                      public = list(
                        #' Initialize Gaussian Naive Bayes Classifier
                        #'
                        #' This method initializes an instance of the Gaussian Naive Bayes Classifier.
                        #' It sets up the environment, including verifying and setting the number of processor cores
                        #' to be used for computation and optionally provides instructions for using the classifier.
                        #'
                        #' @param n_cores The number of processor cores to use for calculations.
                        #'        The method verifies if the specified number of cores is available.
                        #' @param verbose A logical value indicating whether to print detailed messages
                        #'        about the initialization process. Default is TRUE.
                        #'
                        #' @details If the specified number of cores (`n_cores`) exceeds the number of available cores,
                        #'          the function throws an error. When `verbose` is TRUE, the function prints
                        #'          instructions on the console for preprocessing and using the classifier.
                        #'          The Gaussian Naive Bayes Classifier object inherit from the AFDM class,
                        #'          allowing for factor analysis of mixed data if necessary.
                        #'
                        #' @examples
                        #' # Initialize the Gaussian Naive Bayes Classifier with 2 cores and verbose output
                        #' gaussian_nb <- GaussianNB$new(n_cores = 2)
                        #'
                        #' @export
                        initialize = function(n_cores=1, verbose=T) {


                          n_cores_avail <- detectCores()
                          if (n_cores > n_cores_avail){
                            stop(paste0("Only ", n_cores_avail, " are available, please change the 'n_cores' argument"))
                          }
                          private$n_cores <- n_cores # Number of cores chosen

                          if (verbose == T){
                            cat("The Gaussian Naïve Bayes Classifier has been correctly instanciated. \n")
                            cat(paste0("The calculation will be distributed on ", private$n_cores, " cores."))
                            cat("\n")
                            cat("\n")
                            cat("Please, follows these steps: \n")
                            cat("   1) Preprocess your explanatory variables 'X_train_sc <- preprocessing_train(X_train)' \n")
                            cat("   2) Fit your data with the following 'fit(X_train_sc, y_train)' \n")
                            cat("   3) Preprocess your explanatory variables from your test sample 'X_test_sc <- preprocessing_test(X_test)' \n")
                            cat("   4) Predict on your test sample 'predict(X_test_sc)' \n")
                            cat("   5) Test the performance 'score(X_test_sc, y_test)'")
                          }

                        },

                        #' Preprocess Training Data for Gaussian Naive Bayes Classifier
                        #'
                        #' This method preprocesses the training dataset for use with the Gaussian Naive Bayes Classifier.
                        #' It includes checks for data consistency and applies necessary transformations.
                        #'
                        #' @param X_df A dataframe containing the explanatory variables.
                        #' @param y_vec A vector containing the target variable.
                        #' @param n_compon The number of components for the FAMD. Default: NULL, the elbow will be automatically detect.
                        #'
                        #' @return A preprocessed dataframe suitable for training the Gaussian Naive Bayes Classifier.
                        #'         If the dataset includes mixed data (both categorical and numerical),
                        #'         a Factor Analysis of Mixed Data (FAMD) is applied and only numerical data is returned.
                        #'         If the data is purely numerical, it return the dataset.
                        #'
                        #' @details The function performs several checks: it ensures that the sizes of `X_df` and `y_vec` match,
                        #'          that `X_df` is indeed a dataframe and not entirely composed of categorical variables,
                        #'          and that `y_vec` contains more than one class. The method handles mixed data by
                        #'          applying FAMD via the inherited `AFDM` class.
                        #'
                        #' @examples
                        #' # Assuming `X_train` and `y_train` are your explanatory and target variables respectively
                        #'
                        #' X_train_sc <- gaussian_nb$preprocessing_train(X_train, y_train)
                        #'
                        #' @export
                        preprocessing_train = function(X_df, n_compon=NULL){

                          # Extract the qualitative feature
                          non_numeric_columns <- sapply(X_df, function(x) !is.numeric(x))
                          # Check if the qualitative features are factors
                          if (any(non_numeric_columns) && !all(sapply(X_df[non_numeric_columns], is.factor))) {
                            stop("All non-numeric explanatory variables must be factors. You can apply the following function from dplyr package (you must load 'dplyr' or tidyverse' package): X_train <- X_train %>% mutate_if(is.character, as.factor)")
                          }

                          # Check that the explanatory variables are not all categorical
                          if (all(sapply(X_df, function(x) is.character(x) || is.factor(x)))) {
                            stop("All the explanatory variables are either character or factor.")
                          }
                          # Check that the explanatory variables are in a dataframe.
                          if (is.data.frame(X_df) == FALSE) {
                            stop("The explanatory variables must be within a dataframe.")
                          }


                          # Check if there are missing values within the columns
                          if (any(is.na(X_df))) {
                            stop("Missing values detected in the explanatory variables.")
                          }

                          # Check if one of the explanatory variables have a unique value (constant across all individuals)
                          if (any(sapply(X_df, function(x) length(unique(x)) == 1))) {
                            warning("One or more explanatory variables have a constant value.")
                          }

                          # Fields are detailed in the private section
                          private$X <- X_df
                          private$features <- colnames(X_df)
                          private$p <- ncol(X_df)
                          private$n <- nrow(X_df)

                          # If the data is mixed, an AFDM is applied to obtain only numerical data.
                          if (any(sapply(private$X, is.numeric)) && any(sapply(private$X, is.factor))) {

                            # Call the fit_transform function of the ancestor class "AFDM".
                            private$afdm_value <- T # Boolean that a FAMD has been done


                            if (is.null(n_compon)){
                              # If there is no number of components specified, the number of components is automatically detected.
                              private$X_scaled <- as.data.frame(super$fit_transform(private$X))
                            } else {
                              private$X_scaled <- as.data.frame(super$fit_transform(private$X, n_compon=n_compon))
                            }

                            # Adjust the number of variables
                            private$p <- ncol(private$X_scaled)
                            # Adjust variable names
                            private$features <- colnames(private$X_scaled)
                            print(paste0("Your dataset includes mixed data (categorical and numerical). A FMDA has been realized, ",  private$p, " components have been kept."))

                            # If the data is numeric only, the original dataset is returned.
                          } else if (all(sapply(private$X, is.numeric))){
                            private$X_scaled <- private$X
                          }

                          return(private$X_scaled)
                        },

                        #' Preprocess Test Data for Gaussian Naive Bayes Classifier
                        #'
                        #' This method preprocesses the test dataset for use with the Gaussian Naive Bayes Classifier.
                        #' It applies necessary transformations based on the nature of the data (mixed or purely numerical).
                        #'
                        #' @param X_test A dataframe containing the test set explanatory variables.
                        #'
                        #' @return A preprocessed dataframe suitable for making predictions with the Gaussian Naive Bayes Classifier.
                        #'         If the dataset includes mixed data (both categorical and numerical),
                        #'         a transformation is applied without retraining the model.
                        #'         For purely numerical data, it returns the dataset.
                        #'
                        #' @details The function checks the type of variables in `X_test` and applies the appropriate
                        #'          preprocessing. For mixed data, it uses the `coord_supplementaries` method
                        #'          from the inherited AFDM class to transform the data without additional training.
                        #'
                        #' @examples
                        #' # Assuming `X_test` is your test dataset
                        #'
                        #' X_test_sc <- gaussian_nb$preprocessing_test(X_test)
                        #'
                        #' @export
                        preprocessing_test = function(X_test){
                          # Transform every character columns into factor
                          X_test <- X_test  %>%
                            mutate_if(is.character, as.factor)

                          # If data types are mixed, we project the new individuals into the factorial space already created
                          if (any(sapply(X_test, is.numeric)) && any(sapply(X_test, is.factor))) {
                            X_test_sc <- super$coord_supplementaries(X_test)

                            # Otherwise, we return the dataset.
                          } else if (all(sapply(X_test, is.numeric))){
                            X_test_sc <- X_test

                          }
                          return(X_test_sc)
                        },

                        #' Train Gaussian Naive Bayes Classifier
                        #'
                        #' This method trains the Gaussian Naive Bayes Classifier using the provided training data.
                        #'
                        #' @param X A preprocessed dataframe of explanatory variables used for training.
                        #' @param verbose A logical value indicating whether to print detailed messages
                        #'        about the training process. Default is TRUE.
                        #'
                        #' @return The method does not return anything but updates the model's internal state
                        #'         with the training results, including feature statistics and performance metrics.
                        #'
                        #' @details The method employs parallel computing for efficient calculations.
                        #'          Training performance is evaluated using confusion matrix, error rate,
                        #'          confidence intervals, recall, and precision for each class.
                        #'          These metrics are stored in the model's private fields.
                        #'
                        #' @examples
                        #' # Assuming `X_train_sc` is your preprocessed training dataset
                        #'
                        #' gaussian_nb$fit(X_train_sc)
                        #'
                        #' @export
                        fit = function(X_sc, y_vec, verbose = T){

                          # Check if the target variable is a factor
                          if (!is.factor(y_vec)) {
                            stop("The target variable (y_vec) must be a factor.")
                          }

                          # Check if there is more than one class within the target variable
                          if(nlevels(y_vec) < 2){
                            stop("The target variable must have more than one class.")
                          }

                          # Check that the explanatory variable dataframe and the target variable vector have the same size
                          if (nrow(X_sc) != length(y_vec)) {
                            stop("The vector of the target variable does not have the same size as the dataframe of explanatory variables.")
                          }

                          # Check if there are missing values within the target variable
                          if (any(is.na(y_vec))) {
                            stop("Missing values detected in the target variable.")
                          }

                          # Indicate that learning has been completed.
                          private$fit_val <- T
                          private$y <- y_vec
                          private$k <- nlevels(y_vec)
                          private$n_k <- as.vector(table(private$y))
                          private$prior <- private$n_k / private$n
                          private$class <- levels(y_vec)

                          if (verbose == T){
                            cat("Fitting in progress...")
                          }

                          private$features <- colnames(X_sc)
                          private$p <- ncol(X_sc)

                          # Create clusters for the parallelization
                          cl <- makeCluster(private$n_cores)

                          # Calculate the conditional means
                          private$mean_k <- parSapply(cl, X_sc, function(col) {
                            tapply(col, private$y, mean)
                          })

                          # On calcule les variances conditionnelles
                          private$var_k <- parSapply(cl, X_sc, function(col) {
                            tapply(col, private$y, var)
                          })

                          # Calculate the conditional variances
                          private$var_pooled <- parApply(cl, private$var_k, 2, function(x) {
                            sum((private$n_k - 1) * x) / (private$n - private$k)
                          })

                          ### CALCULATE THE CLASSIFICATION FUNCTIONS
                          # Calculate the coefficients of the linear classification functions
                          private$coef <- t(parApply(cl, private$mean_k, 1, function(x){x/private$var_pooled}))

                          # Calculate the intercepts of the linear classification functions
                          private$intercept <- log(private$prior) - 0.5 * colSums(apply(private$mean_k,1,function(x){x^2/private$var_pooled}))

                          ### CALCULATE THE F-VALUE FOR EACH VARIABLE
                          # Calculate the averages of the variables
                          private$mean_var <- parApply(cl, X_sc, 2, mean)
                          # Calculate the first part of the F-value formula for each variable
                          F_1 <- parSapply(cl, 1:private$p,function(j){sum(private$n_k*(private$mean_k[,j]-private$mean_var[j])^2)})/(private$k-1)
                          # Divide by the pooled variance
                          private$F_value <- F_1 / private$var_pooled
                          # Extract the p-value for each variable
                          private$feature_pval <- pf(private$F_value,private$k-1,private$n-private$k,lower.tail=FALSE)

                          stopCluster(cl) # Close the clusters and print a warning message.
                          if (verbose == T){
                            cat("Clusters Closed")
                          }


                          ### CALCULATE THE PERFORMANCE OF THE TRAINING
                          # Extract the predicted classes for the explanatory variables from the training sample
                          predict_train <- self$predict(X_sc)
                          #
                          y_train <- private$y

                          #  Extract the true classes from the training sample
                          private$cm_train <- table(y_train,predict_train)

                          # Find unique classes in y_train and predict_train
                          unique_y_train <- rownames(private$cm_train)
                          unique_predict_train <- colnames(private$cm_train)

                          # Check for missing classes
                          missing_classes <- setdiff(unique_y_train, unique_predict_train)

                          # Add a column of zeros for each missing class
                          for (missing_class in missing_classes) {
                            private$cm_train <- cbind(private$cm_train, rep(0, nrow(private$cm_train)))
                            colnames(private$cm_train)[ncol(private$cm_train)] <- missing_class
                          }
                          # Reorder the columns
                          private$cm_train <- private$cm_train[,unique_y_train]

                          # Calculate the error rate for the training sample
                          private$error_train <- 1 - sum(diag(private$cm_train))/sum(private$cm_train)
                          # Calculate the 95% confidence interval for the training sample
                          et <- sqrt(private$error_train*(1-private$error_train)/private$n)
                          z <- qnorm(0.975)
                          private$li_train <- private$error_train - z*et
                          private$ls_train <- private$error_train + z*et

                          # Calculate the recall for each class in the training sample
                          private$recall_train <- diag(private$cm_train)/rowSums(private$cm_train)
                          # Calculate the precision for each class in the training sample
                          private$precision_train <- diag(private$cm_train)/colSums(private$cm_train)

                          if (verbose == T){
                            cat("\n")
                            cat("The training has been completed correctly!")
                          }
                        },


                        #' Predict Class Labels Using Gaussian Naive Bayes Classifier
                        #'
                        #' This method predicts the class labels for each observation in the test dataset
                        #' using the Gaussian Naive Bayes Classifier. It offers two modes of prediction:
                        #' linear combination ('comblin') and probability-based ('proba').
                        #'
                        #' @param X_test A dataframe containing the preprocessed test set explanatory variables.
                        #' @param type A string specifying the prediction mode: 'comblin' for linear combination
                        #'        or 'proba' for probability-based classification. Default is 'comblin'.
                        #'
                        #' @return A vector of predicted class labels for each observation in the test dataset.
                        #'
                        #' @details The method first verifies if the model has been trained with the `fit` method.
                        #'          In 'comblin' mode, it calculates the scores for each class using the learned
                        #'          coefficients and intercepts, and assigns each observation to the class with the
                        #'          highest score. In 'proba' mode, it assigns class labels based on the highest
                        #'          predicted probabilities.
                        #'
                        #' @examples
                        #' # Assuming `X_test_sc` is your preprocessed test dataset
                        #' # Ensure that the model has been trained using fit method
                        #'
                        #' y_pred <- gaussian_nb$predict(X_test_sc, type="comblin")
                        #'
                        #' @export
                        predict = function(X_test, type="comblin"){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }
                          # Check that the type argument is correct
                          if (!(type %in% c("comblin", "proba"))){
                            stop("Specify a valid type to be based on for the prediction: 'comblin' for linear combination or 'proba' for probability")
                          }

                          # If the predicted class are predicted from the classification functions,
                          if (type == "comblin") {
                            # Create an empty matrix which will receive the scores for each ranking function (the individual's score for class y_k)
                            mat_scores <- matrix(nrow=nrow(X_test), ncol=nrow(private$coef))
                            # Assign the class names as column names
                            colnames(mat_scores) <- levels(private$y)
                            # For every class,
                            for (i in 1:nrow(private$coef)){
                              # Calculate the sum of the product of the individual's values and the coefficient for each variable
                              coef_mult = apply(X_test, 1, function(x){x * private$coef[i, ]})
                              # Add the intercept to this score
                              mat_scores[,i] = colSums(coef_mult) + private$intercept[i]
                            }
                            #  Retrieve the index of the column with the maximum score for this individual
                            predict_class_ind <- max.col(mat_scores)
                            # Retrieve the label for this column with the maximum score
                            private$predict_class <- colnames(mat_scores)[predict_class_ind]

                            # If the predicted class are predicted from the probabilities,
                          } else if (type == "proba"){
                            # Create a dataframe with the probability per class for each individual

                            proba_df <- as.data.frame(self$predict_proba(X_test))
                            #  Retrieve the index of the column with the maximum probability for this individual
                            max_values_indices <- max.col(proba_df, ties.method = "first")
                            # Get the class for each index
                            private$predict_class <- colnames(proba_df)[max_values_indices]
                          }

                          return(private$predict_class)

                        },

                        #' Calculate Log Probabilities for Each Class
                        #'
                        #' This method calculates the logarithm of the probabilities for each class
                        #' for each observation in the test dataset using the Gaussian Naive Bayes Classifier.
                        #'
                        #' @param X_test A dataframe containing the preprocessed test set explanatory variables.
                        #'
                        #' @return A matrix of log probabilities, where each row corresponds to an observation
                        #'         and each column to a class. The log-probabilities are normalized using the logsumexp trick.
                        #'
                        #' @details Before calculating the log probabilities, the method checks if the model
                        #'          has been trained. It uses parallel processing to calculate the log probabilities
                        #'          for efficiency.
                        #'          The values of our function differ slightly from those in scikit-learn because we employ the
                        #'          bias-corrected variance  divided by n - 1, whereas scikit-learn uses the variance divided by n
                        #'          (numpy's var function). This discrepancy does not impact the classification.
                        #'
                        #'
                        #' @examples
                        #' # Assuming `X_test_sc` is your preprocessed test dataset
                        #'
                        #' log_probs <- gaussian_nb$predict_log_proba(X_test_sc)
                        #'
                        #' @export
                        predict_log_proba = function(X_test) {
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }

                          # Create a matrix that will receive the log_proba for each class
                          mat_log_likelihood <- matrix(NA, nrow = nrow(X_test), ncol = private$k)
                          # Assign the class names as column names.
                          colnames(mat_log_likelihood) <- levels(private$y)

                          cl <- makeCluster(private$n_cores) # Create the clusters

                          # For each class,
                          for (i in 1:private$k) {

                            mat_log_likelihood[, i] <- parApply(cl, X_test, 1, function(x) {
                              # Calculate the log of the prior for this class
                              jointi <- log(private$prior[i])
                              # [REVOIR] https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes formula logarithmed
                              # First part of the likelihood calculation formula
                              likelihood <- (-0.5 * sum(log(2 * pi * private$var_k[i, ])))
                              # Second part
                              likelihood <- likelihood - 0.5 * sum(((x - private$mean_k[i, ])^2) / private$var_k[i, ])
                              # Sum of log proba and the log prior
                              joint_log_likelihood <- likelihood + jointi
                              # Return the log_proba matrix
                              return(joint_log_likelihood)
                            })
                          }

                          # [REVOIR]logsumexp trick' to retrieve posterior probabilities as in the scikit-learn function - ref: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/naive_bayes.py
                          private$mat_log_likelihood_norm <- t(parApply(cl, mat_log_likelihood, 1, function(x) {
                            max_x <- max(x)
                            x - (max_x + log(sum(exp(x - max_x))))
                          }))

                          stopCluster(cl) # Close the clusters and print a warning message.

                          # Return the normalized log_proba matrix
                          return(private$mat_log_likelihood_norm)
                        },

                        #' Calculate Posterior Probabilities for Each Class
                        #'
                        #' This method computes the posterior probabilities for each class
                        #' for each observation in the test dataset using the Gaussian Naive Bayes Classifier.
                        #'
                        #' @param X_test A dataframe containing the preprocessed test set explanatory variables.
                        #'
                        #' @return A matrix of posterior probabilities, where each row corresponds to an observation
                        #'         and each column to a class. These probabilities are derived from the log probabilities
                        #'         computed by the `predict_log_proba` method.
                        #'
                        #' @details The method first checks if the model has been trained.
                        #'          It then calls the `predict_log_proba` method to obtain the log probabilities
                        #'          and applies the exponential function to these values to get the posterior probabilities.
                        #'
                        #' @examples
                        #' # Assuming `X_test_sc` is your preprocessed test dataset
                        #'
                        #' probs <- gaussian_nb$predict_proba(X_test_sc)
                        #'
                        #' @export
                        predict_proba = function(X_test){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }

                          # Calculate the exponential of the log_proba matrix
                          private$mat_proba <-  exp(self$predict_log_proba(X_test))

                          return(private$mat_proba)
                        },

                        #' Evaluate Performance of Gaussian Naive Bayes Classifier
                        #'
                        #' This method evaluates the performance of the Gaussian Naive Bayes model on a test dataset.
                        #' It calculates the error rate, recall, precision, and provides the confusion matrix.
                        #'
                        #' @param X_test A dataframe containing the test set explanatory variables.
                        #' @param y_test A vector containing the actual class labels of the test set.
                        #' @param type A string specifying the prediction mode: 'comblin' (default) for linear
                        #'        combination, 'proba' for probability-based classification.
                        #' @param verbose A boolean indicating whether to print detailed output (default is TRUE).
                        #'
                        #' @return A list containing the confusion matrix, error rate, 95% confidence interval for the error rate,
                        #'         recall, and precision for each class.
                        #'
                        #' @details The method checks if the model has been trained, predicts class labels for the test set,
                        #'          constructs a confusion matrix, and calculates error rate, recall, and precision.
                        #'          It also computes a 95% confidence interval for the error rate.
                        #'          The results are returned as a list and can optionally be printed in detail.
                        #'
                        #' @examples
                        #' # Assuming `X_test` and `y_test` are your test dataset and labels
                        #' # Ensure that the model has been trained using fit method
                        #'
                        #' results <- gaussian_nb$score(X_test, y_test)
                        #'
                        #' @export
                        score = function(X_test,  y_test, type = "comblin",verbose=T){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }
                          if (type == "comblin"){
                            y_pred <- self$predict(X_test, type="comblin")
                          } else if (type == "proba"){
                            y_pred <- self$predict(X_test, type="proba")
                          }

                          # Construction of the confusion matrix
                          cm <- table(y_test, y_pred)

                          ## CHECK IN CASE ONE CLASS IS NOT PREDICTED
                          # Find unique classes in y_train and predict_train
                          unique_y_test <- rownames(cm)
                          unique_predict_test <- colnames(cm)

                          # Check for missing classes
                          missing_classes <- setdiff(unique_y_test, unique_predict_test)

                          # Add a column of zeros for each missing class
                          for (missing_class in missing_classes) {
                            cm <- cbind(cm, rep(0, nrow(cm)))
                            colnames(cm)[ncol(cm)] <- missing_class
                          }
                          # Reorder the columns
                          cm <- cm[,unique_y_test]

                          # Calculate the error rate
                          error <- 1 - sum(diag(cm))/sum(cm)

                          # Calculate the recall for each class for the test sample
                          recall <- diag(cm)/rowSums(cm)
                          # Calculate the precision for each class for the test sample
                          precision <- diag(cm)/colSums(cm)

                          # Calculate the 95% confidence interval for the test samplet
                          n <- sum(cm)
                          et <- sqrt(error*(1-error)/n)
                          z <- qnorm(0.975)
                          li <- error - z*et
                          ls <- error + z*et

                          if(verbose == T){
                            cat("Confusion Matrix")
                            print(cm)
                            cat("\n")
                            cat("\n")
                            cat(paste0("Error rate = ", round(error, 3), " [",round(li,3),";",round(ls,3),"] CI 95%"))
                            cat("\n")
                            cat("\n")
                            cat("Recall : \n")
                            print(round(recall,2))
                            cat("Precision : \n")
                            print(round(precision,2))
                          }

                          # Create a list of results that can be extracted by calling this function
                          results <- list(
                            confusion_matrix = cm,
                            error = error,
                            confidence_interval = c(li, ls),
                            recall = recall,
                            precision = precision
                          )

                          # Returns the list without displaying it
                          invisible(results)
                        },

                        #' Print Short Summary of Gaussian Naive Bayes Model
                        #'
                        #' This method prints a short summary of the Gaussian Naive Bayes model,
                        #' including details about the target variable, sample size, and error rate on the training set.
                        #'
                        #' @details The method checks if the model has been trained and then prints
                        #'          the number of classes in the target variable, the number of observations
                        #'          in the sample, and the error rate on the training set along with its 95%
                        #'          confidence interval. This provides a quick overview of the model's training performance.
                        #'
                        #' @examples
                        #' # Assuming the Gaussian Naive Bayes model has been trained
                        #'
                        #' gaussian_nb$print()
                        #'
                        #' # OR
                        #'
                        #' print(gaussian_nb)
                        #'
                        #' @export
                        print = function(){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }
                          cat("The target variable has", private$k, "classes. The sample has", private$n, "observations.")
                          cat("\n")
                          cat(paste0("The model has an error rate on the training sample of ", round(private$error_train, 3), " [",round(private$li_train,3),";",round(private$ls_train,3),"] at 95%CI"))

                        },

                        #' Detailed Summary of Gaussian Naive Bayes Model
                        #'
                        #' This method displays a comprehensive summary of the Gaussian Naive Bayes model,
                        #' including details about the model's classes, features, performance metrics,
                        #' and classification function.
                        #'
                        #' @details The method checks if the model has been trained.
                        #'          It prints details such as the number of classes, class labels, number of features,
                        #'          size of the training sample, prior probabilities of each class,
                        #'          classifier performance metrics (confusion matrix, error rate, recall, precision) on the training dataset,
                        #'          and the classification function (coefficients, intercepts, F-values, p-values).
                        #'          Statistical significance is indicated for features with p-value < 0.05.
                        #'          The summary provides a deep insight into the model's characteristics and performance.
                        #'
                        #' @examples
                        #' # Assuming the Gaussian Naive Bayes model has been trained
                        #'
                        #' gaussian_nb$summary()
                        #'
                        #' # OR
                        #'
                        #' summary(gaussian_nb)
                        #'
                        #' @export
                        summary = function(print_afdm=F){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }
                          if (private$afdm_value != T & print_afdm == T){
                            stop("No FAMD was used in this algorithm. Please, assign FALSE to print_afdm.")
                          }
                          cat("################ GAUSSIAN NAIVE BAYES CLASSIFIER ################ \n")
                          cat("\n")
                          cat("\n")
                          cat("Number of classes: ", private$k, "\n")
                          cat("Labels of classes:", private$class, "\n")
                          cat("Number of features: ", private$p, "\n")
                          cat("Number of observations in the training sample: ", private$n, "\n")
                          cat("Prior of each class: \n")
                          prior <- private$prior
                          names(prior) <- private$class
                          print(prior)
                          cat("\n")
                          cat("\n")
                          cat("CLASSIFIER PERFORMANCE -------------- \n")
                          cat("\n")
                          print(private$cm_train)
                          cat("\n")
                          cat("\n")
                          cat(paste0("Error Rate = ", round(private$error_train, 3), " [",round(private$li_train,3),";",round(private$ls_train,3),"] at 95%CI"))
                          cat("\n")
                          cat("Recall : \n")
                          print(round(private$recall_train,2))
                          cat("Precision : \n")
                          print(round(private$precision_train,2))
                          cat("\n")
                          cat("\n")
                          if (private$afdm_value == T & print_afdm == T){
                            cat("A FAMD has been done. --------------\n")
                            afdm_results <-  super$get_var_characteristics()
                          }

                          cat("CLASSIFICATION FUNCTIONS --------------")
                          cat("\n")
                          cat("\n")
                          table_classif <- t(round(private$coef,5)) # Extract the coefficients
                          table_classif <- rbind(round(private$intercept, 5), table_classif) # Join the constants
                          rownames(table_classif)[1] <- "Intercept" # Assign the name "Interept" as an index to the line of constants
                          f_stat <- c("", round(private$F_value,5))  # Extract the F-values
                          p_val <- c("", sprintf("%.5e", private$feature_pval)) # Adjust the number of decimal places in the values, while keeping the scientific format
                          signif <- ifelse(private$feature_pval < 0.05, "**","") # Create a variable that indicates "**" if the p-value is less than 0.05
                          signif <- c("", signif)
                          # These values are added to the results table
                          table_classif <- cbind(table_classif, f_stat)
                          column_index <- which(colnames(table_classif) == "f_stat")
                          # Assign the column name "F" with the degrees of freedom
                          colnames(table_classif)[column_index] <- paste0("F(", private$k-1, ",", private$n-private$k, ")")

                          table_classif <- cbind(table_classif, "P-Value" = p_val)
                          table_classif <- cbind(table_classif, "Signif." = signif)

                          print(table_classif, quote = FALSE)
                          cat("\n")
                          cat("** : Statistically significant at p-value < 0.05")

                          # Create a list to extract the results by calling the function
                          results_summary <- list(
                            n_classes = private$k,
                            label_classes = private$class,
                            prior = prior,
                            features = private$p,
                            n = private$n,
                            cm_train = private$cm_train,
                            error_train = private$error_train,
                            conf_error_train = c(private$li_train, private$ls_train),
                            recall_train = private$recall_train,
                            precision_train = private$precision_train,
                            classification_function = list(coef = private$coef,
                                                           intercept  = private$intercept,
                                                           F_value = private$F_value,
                                                           pval = private$feature_pval)
                          )

                          # Returns the list without displaying it
                          invisible(results_summary)
                        },

                        #' Visualize Confusion Matrix for Gaussian Naive Bayes Classifier
                        #'
                        #' This method creates a visual representation of the confusion matrix for the test dataset
                        #' using the Gaussian Naive Bayes Classifier. It allows selection between class label prediction
                        #' based on linear combination (`comblin`) or probabilities (`proba`).
                        #'
                        #' @param X A dataframe containing the test set explanatory variables.
                        #' @param y_test A vector containing the actual class labels of the test set.
                        #' @param type A string specifying the prediction type: 'comblin' (default) for linear
                        #'        combination based classification, 'proba' for probability based classification.
                        #'
                        #' @details The method first checks if the model has been trained.
                        #'          It then predicts the class labels for the test set using the specified prediction type
                        #'          and constructs a confusion matrix. This matrix is visualized using the `corrplot` package,
                        #'          with a color gradient representing the frequency of each cell in the matrix. The method
                        #'          handles missing classes by adding zero-filled columns to ensure a complete matrix.
                        #'
                        #' @examples
                        #' # Assuming `X_test_sc` and `y_test` are your preprocessed test dataset and actual labels
                        #'
                        #' gaussian_nb$plot_confusionmatrix(X_test_sc, y_test)
                        #'
                        #' @export
                        plot_confusionmatrix = function(X, y_test , type="comblin"){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }
                          if (type == "comblin"){
                            y_pred <- self$predict(X, type="comblin")
                          } else if (type == "proba"){
                            y_pred <- self$predict(X, type="proba")
                          }

                          # Construction of the confusion matrix
                          cm <- table(y_test, y_pred)

                          # Find unique classes in y_train and predict_train
                          unique_y_test <- rownames(cm)
                          unique_predict_test <- colnames(cm)

                          # Check for missing classes
                          missing_classes <- setdiff(unique_y_test, unique_predict_test)

                          # Add a column of zeros for each missing class
                          for (missing_class in missing_classes) {
                            cm <- cbind(cm, rep(0, nrow(cm)))
                            colnames(cm)[ncol(cm)] <- missing_class
                          }
                          # Reorder the columns
                          cm <- cm[,unique_y_test]

                          # Create a colour palette
                          palette_col <- c('#fbff82', '#f9ef8a', '#f6df90', '#f3cf97', '#efbf9c', '#ebafa1', '#e79fa6', '#e28eab', '#dc7daf', '#d66bb3')
                          # Create a visualisation using corrplot
                          corrplot(cm, method="color", col = palette_col,  addCoef.col = 'black',  is.corr = FALSE,addgrid.col = 'white', tl.col = 'black',  tl.srt = 45, cl.pos = 'n', title="Confusion Matrix", mar=c(2,0,2,0))
                        },

                        #' Generate ROC Curve for Gaussian Naive Bayes Classifier
                        #'
                        #' This method plots the Receiver Operating Characteristic (ROC) curve for the test dataset
                        #' using the Gaussian Naive Bayes Classifier. It handles both binary and multi-class scenarios.
                        #'
                        #' @param X A dataframe containing the preprocessed test set explanatory variables.
                        #' @param y_test A vector containing the actual class labels of the test set.
                        #' @param positive An optional parameter specifying the positive class label in a binary classification scenario.
                        #'        It is required when the model is trained on a binary classification task.
                        #'
                        #' @details For multi-class classification (more than two classes), the method generates a One-vs-Rest (OVR)
                        #'          multi-class ROC curve. For binary classification, it plots a ROC curve for the specified positive class.
                        #'          The Area Under the Curve (AUC) is calculated for each class. The method checks if the model
                        #'          has been trained before generating the ROC curve.
                        #'
                        #' @examples
                        #' # Assuming `X_test_sc` and `y_test` are your preprocessed test dataset and actual labels
                        #'  This is the case of a binary classification
                        #'
                        #' gaussian_nb$plot_roccurve(X_test_sc, y_test, positive = "YourPositiveClassLabel")
                        #'
                        #' @export
                        plot_roccurve = function(X, y_test, positive=NULL){
                          # Check that the training has been carried out correctly
                          if (private$fit_val != T){
                            stop("You have to fit your model first! ( 1) gnb <- GaussianNB$new(X,y) ;2) gnb$fit()")
                          }

                          # If there are more than 2 classes, an OVR multi-class Roc curve will be displayed.
                          if (private$k > 2){
                            # Extract the classification probabilities
                            proba_df <- as.data.frame(self$predict_proba(X))
                            # Extract the true classes of individuals
                            true_classes = levels(y_test)[y_test]

                            #  For each class of the target variable,
                            roc_data <- lapply(private$class, function(current_class) {
                              # Extract the classification probabilities
                              proba_class <- proba_df[[current_class]]
                              # Create a dataframe with the probabilities and the true class
                              df_class <- data.frame(proba_class = proba_class, class = true_classes)
                              # Create a variable which takes 1 if the individual has the class studied as his real class
                              df_class <- df_class %>%
                                mutate(positive = ifelse(class == current_class, 1, 0))

                              # Arrange the dataframe in descending order of probability of classification
                              df_class <- df_class[order(-df_class$proba_class), ]

                              # Calculate the number of individuals with the class under study as their true class
                              npos <- sum(df_class$positive)
                              # Calculate the number of individuals who do not have the class studied as their true class
                              nneg <- sum(1-df_class$positive)
                              tvp <- cumsum(df_class$positive) / npos # True positives
                              tfp <- cumsum(1 - df_class$positive) / nneg # False positives

                              # Each individual is assigned a rank
                              rank <- seq(nrow(df_class),1,-1)
                              # Calculate the sum of positive ranks using "positive" (which equals 1 if positive).
                              sum_rkpos <- sum(rank * df_class$positive)
                              # Calculate the Mann-Whitney statistic
                              U = sum_rkpos - (npos*(npos+1))/2
                              # Calculate the AUC
                              auc <- U / (npos*nneg)

                              # Create a label showing the class studied and the AUC value
                              auc_class = paste0(current_class, " (AUC =", round(auc,2),")")

                              # Create a dataframe with the current class, the rate of true positives and false positives.
                              data.frame(
                                class = auc_class,
                                tfp = tfp,
                                tvp = tvp
                              )
                            }) %>% bind_rows() # We combine the dataframes one below the other

                            # Create a graph with one ROC curve for each class
                            ggplot(roc_data, aes(x = tfp, y = tvp, color = class)) +
                              geom_line(linewidth=1, alpha=0.8) +
                              geom_abline(slope = 1, intercept = 0, linetype="dashed", linewidth=0.7) +
                              labs(title = "One-vs-Rest Multi-class ROC Curve", x = "False Positive Rate", y = "True Positive Rate", color = "Classes") +
                              theme_minimal()

                            # If there are only two classes, only one curve is displayed
                          } else if (private$k == 2){

                            # Check that the user has indicated the class considered as positive
                            if (is.null(positive)){
                              stop("You must indicate the positive label")
                            }

                            # Check that the user has indicated an existing class
                            if (!(positive %in% private$class)){
                              stop(paste("You must use one of the existing classe: ", private$class))
                            }

                            # The "positive" argument is transformed into a character
                            label_positive = as.character(positive)

                            # The processing is the same as above.
                            proba_df <- as.data.frame(self$predict_proba(X))
                            proba_positive <- proba_df[[label_positive]]
                            df_class <- data.frame(proba_positive = proba_positive,
                                                   y_test  = y_test)
                            df_class <- df_class %>%
                              mutate(positive = ifelse(y_test == label_positive, 1, 0))
                            df_class <- df_class[order(-df_class$proba_positive), ]

                            npos <- sum(df_class$positive)
                            nneg <- sum(1-df_class$positive)
                            tvp <- cumsum(df_class$positive) / npos
                            tfp <- cumsum(1 - df_class$positive) / nneg

                            rank <- seq(nrow(df_class),1,-1)
                            sum_rkpos <- sum(rank * df_class$positive)
                            U = sum_rkpos - (npos*(npos+1))/2
                            auc <- U / (npos*nneg)

                            auc_title <- paste0("ROC Curve - Positive: ", label_positive," (AUC =", round(auc,2),")")

                            roc_graph <- data.frame(
                              tfp = tfp,
                              tvp = tvp)

                            ggplot(roc_graph, aes(x = tfp, y = tvp)) +
                              geom_line(linewidth=1, color="#d66bb3") +
                              geom_abline(slope = 1, intercept = 0, linetype="dashed", linewidth=0.7) +
                              labs(title = auc_title, x = "False Positive Rate", y = "True Positive Rate") +
                              theme_minimal()
                          }


                        }

                      ),
                      private = list(
                        # The boolean indicating if an AFDM (Analysis of Discriminant Factorial Method) has been performed.
                        afdm_value = F,
                        # Indicates the classes of the target variable.
                        class = NA,
                        # The confusion matrix of the training sample.
                        cm_train = NA,
                        # The coefficients of the ranking functions.
                        coef = NA,
                        # The error rate of the training sample.
                        error_train = NA,
                        # The p-values for the variables.
                        feature_pval = NA,
                        # The names of the variables.
                        features = NA,
                        # The F values of the variables.
                        F_value = NA,
                        # The boolean indicating if the training has been carried out.
                        fit_val = F,
                        # The constants of the ranking functions.
                        intercept = NA,
                        # The number of classes.
                        k = NA,
                        # Lower bound of the confidence interval for the training sample.
                        li_train = NA,
                        # Upper bound of the confidence interval for the training sample.
                        ls_train = NA,
                        #' Mixed data type. The matrix of normalized log-probabilities.
                        mat_log_likelihood_norm = NA,
                        #' The matrix of posterior probabilities.
                        mat_proba = NA,
                        # The conditional means.
                        mean_k = NA,
                        # The global means of the variables.
                        mean_var = NA,
                        # The total sample size.
                        n = NA,
                        # The number of chosen cores.
                        n_cores = NA,
                        # The sample size per class.
                        n_k = NA,
                        # The number of variables.
                        p = NA,
                        # The precision for each class for the training sample.
                        precision_train = NA,
                        # The predicted classes.
                        predict_class = NA,
                        # The prior probabilities.
                        prior = NA,
                        # The recall for each class for the training sample.
                        recall_train = NA,
                        # The conditional variances.
                        var_k = NA,
                        # The common variance.
                        var_pooled = NA,
                        # The dataframe of explanatory variables of the training sample.
                        X = NA,
                        # The dataframe of explanatory variables of the training sample after preprocessing.
                        X_scaled = NA,
                        # The vector of the target variable of the training sample.
                        y = NA
                      )
)
