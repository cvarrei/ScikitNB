BernoulliNB <- R6Class("BernoulliNB",

  public = list(

    #' Initialize Bernoulli Naive Bayes Classifier
    #'
    #' This method initializes an instance of the Bernoulli Naive Bayes Classifier.
    #' It sets up the environment, including verifying and setting the number of processor cores
    #' to be used for computation and optionally provides instructions for using the classifier.
    #'
    #' @param alpha Sets the Laplace smoothing parameter
    #' @param fit_prior Sets whether to learn class prior probabilities
    #' @param class_prior Sets the prior probabilities of the classes
    #' @param n_cores The number of processor cores to use for calculations.
    #'        The method verifies if the specified number of cores is available.
    #' @param verbose A logical value indicating whether to print detailed messages
    #'        about the initialization process. Default is TRUE.
    #'
    #' @details If the specified number of cores (n_cores) exceeds the number of available cores,
    #'          the function throws an error. When verbose is TRUE, the function prints
    #'          instructions on the console for preprocessing and using the classifier.
    #'          This constructor allows you to create an instance of your BernoulliNB class with
    #'          specified parameters.
    #'
    #' @examples
    #' # Initialize the Bernoulli Naive Bayes Classifier with 2 cores and verbose output
    #' bernoulli_nb <- BernoulliNB$new(n_cores = 2)
    #'
    #' @export
    # Constructeur de l'objet
    initialize = function(alpha=1.0, fit_prior=TRUE, class_prior=NULL, n_cores=1, verbose=T ) {
      private$alpha <- alpha
      private$fit_prior <- fit_prior
      private$class_prior <- class_prior
      private$is_fitted <- FALSE

      n_cores_avail <- detectCores()
      if (n_cores > n_cores_avail){
        stop(paste0("Only ", n_cores_avail, " are available, please change the 'n_cores' argument"))
      }
      private$n_cores <- n_cores

      if (verbose == T){
        cat("The Bernoulli Naive Bayes Classifier has been correctly instanciated. \n")
        cat(paste0("The calculation will be distributed on ", private$n_cores, " cores."))
        cat("\n")
        cat("\n")
        cat("Please, follows these steps: \n")
        cat("   1) Preprocess your explanatory variables '(1) preprocess_X = bernoulli_nb$preprocess(X_train, X_test);
            (2) X_train_vectorized = preprocess_X[[1]]; (3) X_test_vectorized = preprocess_X[[2]]' \n")
        cat("   2) Fit your data with the following 'fit(X_train_vectorized, y_train)' \n")
        cat("   4) Predict on your test sample 'predict(X_test_vectorized)'")
      }
    },

    #' Preprocess Data for Bernoulli Naive Bayes Classifier
    #'
    #' This method performs text preprocessing on training and test data.
    #' The goal is to convert raw text data into a binary bag-of-words
    #' representation suitable for the Bernoulli Naive Bayes.
    #'
    #' @param X_train A vector containing the explanatory variables.
    #' @param X_test A vector containing the target variable.
    #' @param to_lower A boolean to change our corpus to lowercase.
    #' @param rm_punctuation A boolean to remove punctuation from our corpus.
    #' @param rm_numbers A boolean to remove numbers from our corpus.
    #' @param rm_stopwords Stopwords to remove or NULL.
    #' @param strip_whitespace A boolean to strip white spaces from our corpus
    #'
    #' @return A list containing the vectorized representations of the training
    #' and test data.
    #'
    #' @details The function contains two other functions :
    #'          'preprocess_corpus(X)' Takes a vector of text documents, creates a text
    #'          corpus, and applies various preprocessing steps such as converting to
    #'          lowercase, removing punctuation, removing numbers, removing stopwords,
    #'          and stripping whitespace. It returns the preprocessed corpus.
    #'          'get_bow(X,vocabulary)' takes a vector of text documents and a vocabulary.
    #'          It uses the 'preprocess_corpus' function to preprocess the text, creates
    #'          a binary bag-of-words. It returns the bag-of-words as a sparse matrix.
    #'
    #' @examples
    #' # Assuming 'X_train' and 'X_test' are your explanatory and target variables respectively
    #'
    #' bernoulli_nb <- BernoulliNB$new(n_cores = 2)
    #' preprocess_X = bernoulli_nb$preprocess(X_train, X_test)
    #' X_train_vectorized = preprocess_X[[1]]
    #' X_test_vectorized = preprocess_X[[2]]
    #'
    #' @export
    preprocess = function(X_train, X_test, to_lower=T, rm_punctuation=T, rm_numbers=T,
                          rm_stopwords=stopwords('english'), strip_whitespace=T) {

      # Preprocess text data
      preprocess_corpus <- function(X) {
        corpus <- Corpus(VectorSource(X))
        if(to_lower){
          corpus <- tm_map(corpus, content_transformer(tolower))
        }
        if(rm_punctuation){
          corpus <- tm_map(corpus, removePunctuation)
        }
        if(rm_numbers){
          corpus <- tm_map(corpus, removeNumbers)
        }
        if(!is.null(rm_stopwords)){
          corpus <- tm_map(corpus, removeWords, rm_stopwords)
        }
        if(strip_whitespace){
          corpus <- tm_map(corpus, stripWhitespace)
        }
        return(corpus)
      }

      # Vectorize text data using a binary bag-of-words representation
      get_bow <- function(X, vocabulary) {
        corpus <- preprocess_corpus(X)
        dtm <- DocumentTermMatrix(corpus, control = list(dictionary = vocabulary, binary = TRUE))
        # Transformation en une matrice exploitable
        dtm_matrix <- as(dtm, "matrix")
        dtm_sparse <- Matrix(dtm_matrix, sparse = TRUE)
        return(dtm_sparse)
      }

      # Extract vocabulary from training set
      X_train_corpus <- preprocess_corpus(X_train)
      vocabulary <- colnames(DocumentTermMatrix(X_train_corpus, control = list(binary = TRUE)))

      # Vectorize training and test sets
      X_train_vectorized = get_bow(X_train, vocabulary = vocabulary)
      X_test_vectorized = get_bow(X_test, vocabulary = vocabulary)

      return(list(X_train_vectorized, X_test_vectorized))
    },

    #' Train Bernoulli Naive Bayes Classifier
    #'
    #' This method trains the Bernoulli Naive Bayes Classifier using the provided training data.
    #'
    #' @param X A vectorized matrix of explanatory features used for training.
    #' @param y a vector of our target variable containing our classes.
    #' @param verbose A logical value indicating whether to print detailed messages
    #'        about the training process. Default is TRUE.
    #'
    #' @return The method does not return anything but updates the model's internal state
    #'         with the training results, including feature statistics and performance metrics.
    #'
    #' @details The methof performs Bernoulli Naive Bayes model training by counting feature
    #'          occurrences, applying Laplace smoothing, removing problematic features, and updating model
    #'          parameters.
    #'
    #' @examples
    #' # Assuming 'X_train_vectorized' is your preprocessed training dataset
    #'
    #' bernoulli_nb <- BernoulliNB$new(n_cores = 2, verbose = TRUE)
    #' bernoulli_nb$fit(X_train_vectorized, y_train)
    #'
    #' @export
    fit = function(X, y, verbose = T) {

      if (verbose == T){
        cat("Fitting in progress...")
      }

      # Convert class labels to a binary matrix
      Y = as.matrix(model.matrix(~y - 1))
      private$classes = levels(factor(y))
      private$train_features = colnames(X)

      # Initialize lists of 0
      n_classes = ncol(Y)
      n_features = ncol(X)

      # ------------------------------- count ---------------------------------

      # Count and smooth feature occurrences and class occurences
      private$feature_count = as.matrix(t(Y) %*% X)
      private$class_count = colSums(Y)

      # ----------------------- update feature log prob ------------------------

      # Update the log probabilities of the features using the
      # Laplace smoothing technique to avoid null probabilities.
      smoothed_fc = private$feature_count + private$alpha # Apply Laplace smoothing to feature counts
      smoothed_cc = private$class_count + private$alpha * 2 # Apply Laplace smoothing to class counts

      # Compute log probabilities for feature log_prob
      private$feature_log_prob = log(smoothed_fc) - log(outer(smoothed_cc,1))[, rep(1,ncol(log(smoothed_fc)))]

      # Identify features where at least one value is greater than 0
      features <- colnames(private$feature_log_prob)
      features_to_remove <- features[colSums(private$feature_log_prob > 0) > 0]

      # Remove the features where a log_prob is > 0
      private$feature_log_prob <- private$feature_log_prob[, !(colnames(private$feature_log_prob) %in% features_to_remove)]

      # Rename the rows
      rownames(private$feature_log_prob) = private$classes

      # -------------------- update class log prior ----------------------------
      if (!is.null(private$class_prior)) {
        if (length(private$class_prior) != n_classes) {

          stop("Number of priors must match number of classes.")
        }

        private$class_log_prior <- log(private$class_prior)

      } else if (private$fit_prior) {

        log_class_count = log(private$class_count)
        # empirical prior, with sample_weight taken into account
        private$class_log_prior <- log_class_count - log(sum(private$class_count))

      } else {
        #Update the log priorities of the classes using an empirical approach
        # taking into account the sampling weight.
        private$class_log_prior <- rep(-log(n_classes), n_classes)
      }

      # -------------------- update some model attributes ----------------------------
      private$is_fitted = TRUE
      private$k <- length(private$class_count) # The number of classes to predict
      private$n = nrow(X)
      features = colnames(private$feature_log_prob)
      X = X[, features]
      private$p = ncol(X)
      private$X_train = X
      private$y_train = y
      private$train_features = colnames(X)


      # Calculate the performance of the training
      self$performances()

      if (verbose == T){
        cat("\n")
        cat("The training has been completed correctly!")
        cat("\n")
      }
    },

    #' Calculate the joint log probability
    #'
    #' This method calculates the logarithm of the probabilities for each class
    #' for each observation in the test dataset using the Bernoulli Naive Bayes Classifier.
    #'
    #' @param X A matrix containing the vectorized test set explanatory features.
    #'
    #' @return A matrix of joint log probabilities, where each row corresponds to an observation
    #'         and each column to a class.
    #'
    #' @details The method checks and selects features, ensures the compatibility of feature numbers,
    #'          calculates inverse log probabilities, computes joint log probabilities, updates them with class
    #'          priors, and returns the results.
    #'
    #'
    #' @examples
    #' # Assuming 'X' is your vectorized test dataset
    #' jll <- bernoulli_nb$joint_log_likelihood(X)
    #'
    #' @export
    joint_log_likelihood = function(X) {
      # Make sure that we have the same features as our calculated feature_log_prob
      features = colnames(private$feature_log_prob)
      X = X[, features]

      # Calculate the posterior log probability of the samples X -- _joint_log_likelihood()
      # Extracts the features used in the trained model
      n_features = ncol(private$feature_log_prob)
      n_features_X = ncol(X)

      if (n_features != n_features_X){
       stop("Number of features in the input does not match the trained model.")
      }

      # Calculation of Inverse Log Probabilities
      neg_prob <- log(1-exp(private$feature_log_prob))

      # Difference between the log probabilities of features and the inverse
      # log probabilities
      jll <- as.matrix(X %*% t(private$feature_log_prob - neg_prob))

      # Update the joint log likelihood
      jll <- jll + matrix(private$class_log_prior + rowSums(neg_prob), nrow=nrow(jll), ncol=ncol(jll), byrow=TRUE)

      return(jll)
    },

    #' Predict Class Labels Using Bernoulli Naive Bayes Classifier
    #'
    #' This method predicts the class labels for each observation in the test dataset
    #' using the Bernoulli Naive Bayes Classifier.
    #'
    #' @param X A matrix containing the vectorized test set explanatory features.
    #'
    #' @return A vector of predicted class labels for each observation in the test dataset.
    #'
    #' @details The method first verifies if the model has been trained with the `fit` method.
    #'          Then it calculates the joint_log_likelihood for the input samples, find the indices of
    #'          the predicted classes using parallel processing, maps them to class labels, and returns
    #'          the predicted classes.
    #'
    #' @examples
    #' # Assuming 'X' is your vectorized test dataset
    #' # Ensure that the model has been trained using fit method
    #' y_pred <- bernoulli_nb$predict(X)
    #'
    #' @export
    predict = function(X) {

      # Check that the fit has been done
      if (!private$is_fitted) {
        stop("You must fit your classifier first. Try (1) bernoulli_nb <- BernoulliNB$new() (2) bernoulli_nb$fit(X_train_vectorized, y_train)")
      }

      # Calculate the joint log likelihood
      jll = self$joint_log_likelihood(X)

      # Register a parallel cluster
      cl <- makeCluster(private$n_cores)

      # Find the index of the maximum value in each row
      predicted_indices <- parApply(cl, jll, 1, which.max)

      # Stop the clusters
      stopCluster(cl)

      # Map indices to class labels
      predicted_classes <- private$classes[predicted_indices]

      return(predicted_classes)
    },

    #' Calculate Log Probabilities for Each Class
    #'
    #' This method calculates the logarithm of the probabilities for each class
    #' for each observation in the test dataset using the Bernoulli Naive Bayes Classifier.
    #'
    #' @param X A vectorized matrix containing test explanatory features.
    #'
    #' @return A matrix of log probabilities, where each row corresponds to an observation
    #'         and each column to a class.
    #'
    #' @details The method calculates the joint_log_likelihood for the input samples. Then it normalizes
    #'          the jll by the marginal probability P(x), and calculates and return the log_probabilities.
    #'
    #' @examples
    #' # Assuming 'X' is your vectorized test dataset
    #' log_probs <- bernoulli_nb$predict_log_proba(X)
    #'
    #' @export
    predict_log_proba = function(X) {

      jll = self$joint_log_likelihood(X)

      # normalize by P(x) = P(f_1, ..., f_n)
      # The marginal probability P(x) is calculated by exponentially summing the joint probabilities over all classes for each observation:
      log_prob_x = log(rowSums(exp(jll)))

      return(jll - as.matrix(log_prob_x)[, rep(1,ncol(jll))])

    },

    #' Calculate Posterior Probabilities for Each Class
    #'
    #' This method computes the posterior probabilities for each class
    #' for each observation in the test dataset using the Bernoulli Naive Bayes Classifier.
    #'
    #' @param X A vectorized matrix containing test explanatory features.
    #'
    #' @return A matrix of posterior probabilities, where each row corresponds to an observation
    #'         and each column to a class. These probabilities are derived from the log probabilities
    #'         computed by the 'predict_log_proba' method.
    #'
    #' @details The method first checks if the model has been trained.
    #'          It then calls the 'predict_log_proba' method to obtain the log probabilities
    #'          and applies the exponential function to these values to get the posterior probabilities.
    #'
    #' @examples
    #' # Assuming `X` is your preprocessed test dataset
    #' probs <- bernoulli_nb$predict_proba(X)
    #'
    #' @export
    predict_proba = function(X) {

      # Check that the fit has been done
      if (!private$is_fitted) {
        stop("You must fit your classifier first. Try (1) bernoulli_nb <- BernoulliNB$new() (2) bernoulli_nb$fit(X_train_vectorized, y_train)")
      }

      return(exp(self$predict_log_proba(X)))
    },

    #' Evaluate Performance of Bernoulli Naive Bayes Classifier
    #'
    #' This method evaluates the performance of the Bernoulli Naive Bayes model on a test dataset.
    #' It calculates the error rate, recall, precision, and provides the confusion matrix.
    #'
    #' @param X A vectorized matrix containing test explanatory features.
    #' @param y A vector containing the actual class labels of the test set.
    #' @param verbose A boolean indicating whether to print detailed output (default is TRUE).
    #'
    #' @return A list containing the confusion matrix, error rate, 95% confidence interval for the error rate,
    #'         recall, and precision for each class.
    #'
    #' @details The method predicts class labels for the test set, constructs a confusion matrix,
    #'          and calculates error rate, recall, and precision. It also computes a 95% confidence interval
    #'          for the error rate. The results are returned as a list and can optionally be printed in detail.
    #'
    #' @examples
    #' # Assuming 'X' and 'y' are your test dataset and labels
    #' # Ensure that the model has been trained using fit method
    #' results <- bernoulli_nb$score(X, y)
    #'
    #' @export
    score = function(X, y, verbose=T) {

      y_pred <- self$predict(X)

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

      # Calculate the 95% confidence interval for the test sample
      n <- sum(cm)
      et <- sqrt(error*(1-error)/n)
      z <- qnorm(0.975)
      li <- error - z*et
      ls <- error + z*et

      if(verbose == T){
        cat("Confusion Matrix")
        cat("\n")
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

    #' Performances of our training dataset
    #'
    #' This method evaluates the performances of our training dataset.
    #' It calculates the error rate, recall, precision, and provides the confusion matrix.
    #'
    #' @details The method predicts class labels for the train set, constructs a confusion matrix,
    #'          and calculates error rate, recall, and precision. It also computes a 95% confidence interval
    #'          for the error rate.
    #'
    #' @examples
    #' # Assuming 'X_test' and 'y_test' are your test dataset and labels
    #' # Ensure that the model has been trained using fit method
    #' results <- bernoulli_nb$score(X_test, y_test)
    #'
    #' @export
    performances = function() {
      # Extract the predicted classes for the exp. variables in the training sample
      predict_train <- self$predict(private$X_train)

      # Extract the confusion matrix for the training sample
      private$cm_train <- table(predict_train, private$y_train)

      # Calculate the error rate for the training sample
      private$error_train <- 1 - sum(diag(private$cm_train))/sum(private$cm_train)
      # We calculate the 95% confidence interval for the training sample
      et <- sqrt(private$error_train*(1-private$error_train)/private$n)
      z <- qnorm(0.975)
      private$li_train <- private$error_train - z*et
      private$ls_train <- private$error_train + z*et

      # Calculate the recall for each class for the training sample
      private$recall_train <- diag(private$cm_train)/rowSums(private$cm_train)
      # We calculate the precision for each class for the training sample
      private$precision_train <- diag(private$cm_train)/colSums(private$cm_train)
    },

    #' Print Short Summary of Bernoulli Naive Bayes Model
    #'
    #' This method prints a short summary of the Bernoulli Naive Bayes model,
    #' including details about the target variable, sample size, and error rate on the training set.
    #'
    #' @details The method checks if the model has been trained and then prints
    #'          the number of classes in the target variable, the number of observations
    #'          in the sample, and the error rate on the training set along with its 95%
    #'          confidence interval. This provides a quick overview of the model's training performance.
    #'
    #' @examples
    #' # Assuming the Bernoulli Naive Bayes model has been trained
    #' bernoulli_nb <- BernoulliNB$new(n_cores = 2, verbose = TRUE)
    #' bernoulli_nb$fit(X_train_vectorized)
    #' bernoulli_nb$print()
    #'
    #' # OR
    #' print(bernoulli_nb)
    #'
    #' @export
    print = function() {

      # Check that the fit has been done
      if (!private$is_fitted) {
        stop("You must fit your classifier first. Try Bernoulli_classifier$fit(X_train_vectorized, y_train)")
      }

      cat("The target variable has", private$k, "classes. The sample has", private$n, "observations.")
      cat("\n")
      cat(paste0("The model has an error rate on the training sample of ", round(private$error_train, 3), " [",round(private$li_train,3),";",round(private$ls_train,3),"] at 95%CI"))
    },

    #' Detailed Summary of Bernoulli Naive Bayes Model
    #'
    #' This method displays a comprehensive summary of the Bernoulli Naive Bayes model,
    #' including details about the model's classes, features, performance metrics,
    #' and classification function.
    #'
    #' @details The method checks if the model has been trained.
    #'          It prints details such as the number of classes, class labels, number of features,
    #'          size of the training sample, prior probabilities of each class,
    #'          classifier performance metrics (confusion matrix, error rate, recall, precision) on the training dataset,
    #'          and the classification function (features significance).
    #'          The summary provides a deep insight into the model's characteristics and performance.
    #'
    #' @examples
    #' # Assuming the Bernoulli Naive Bayes model has been trained
    #'
    #' bernoulli_nb$summary()
    #'
    #' # OR
    #'
    #' summary(bernoulli_nb)
    #'
    #' @export
      summary = function() {

       # Check that the fit has been done
       if (!private$is_fitted) {
         stop("You must fit your classifier first. Try (1) bernoulli_nb <- BernoulliNB$new() (2) bernoulli_nb$fit(X_train_vectorized, y_train)")
       }

       cat("################ BERNOULLI NAIVE BAYES CLASSIFIER ################ \n")
       cat("\n")
       cat("\n")
       cat("Number of classes: ", private$k, "\n")
       cat("Labels of classes:", private$classes, "\n")
       cat("Number of features: ", private$p, "\n")
       cat("Number of observations in the training sample: ", private$n, "\n")
       cat("Logarithm of the prior probabilities for each class: \n")
       prior <- private$class_log_prior
       names(prior) <- private$classes
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
       cat("CLASSIFICATION FUNCTIONS --------------")
       cat("\n")
       cat("\n")
       cat("The 10 tokens with the highest log-proba :")
       for (i in 1:private$k){
         features = self$top_logproba(private$classes[i])
         cat("\n")
         print(private$classes[i])
         cat("\n")
         print(features)
       }

       #Create a list to extract the results by calling the function
        results_summary <- list(
          n_classes = private$k,
          label_classes = private$class,
          prior = private$class_log_prior,
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

    #' Generate ROC Curve for Bernoulli Naive Bayes Classifier
    #'
    #' This method plots the Receiver Operating Characteristic (ROC) curve for the test dataset
    #' using the Bernoulli Naive Bayes Classifier. It handles both binary and multi-class scenarios.
    #'
    #' @param X A matrix containing the preprocessed test set explanatory variables.
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
    #' # Assuming 'X' and 'y_test' are your preprocessed test dataset and actual labels
    #' # This is the case of a binary classification
    #'
    #' bernoulli_nb$plot_roccurve(X, y_test, positive = "YourPositiveClassLabel")
    #'
    #' @export
    plot_roccurve = function(X, y_test, positive=NULL) {
      # Check that the fit has been done
      if (!private$is_fitted) {
        stop("You must fit your classifier first. Try (1) bernoulli_nb <- BernoulliNB$new() (2) bernoulli_nb$fit(X_train_vectorized, y_train)")
      }

      # If there are more than 2 classes, we will display a multi-class OVR Roc curve
      if (private$k > 2){
        # We extract the assignment probabilities
        proba_df <- as.data.frame(self$predict_proba(X))
        # We extract the real classes of the individuals
        true_classes = factor(y_test)


        # For each class of the target variable,
        roc_data <- lapply(private$classes, function(current_class) {
          # We extract the assignment probabilities
          proba_class <- proba_df[[current_class]]
          # We create a dataframe with the probability and the real class
          df_class <- data.frame(proba_class = proba_class, class = true_classes)
          # We create a variable which takes 1 if the individual has the class studied as the real class
          df_class <- df_class %>%
           mutate(positive = ifelse(class == current_class, 1, 0))

          # We organize the dataframe in descending order of probability of assignment
          df_class <- df_class[order(-df_class$proba_class), ]

          # We calculate the number of individuals having the class studied as a real class
          npos <- sum(df_class$positive)
          # We calculate the number of individuals not having the class studied as a real class
          nneg <- sum(1-df_class$positive)
          tvp <- cumsum(df_class$positive) / npos # We calculate the rate of true positives
          tfp <- cumsum(1 - df_class$positive) / nneg # We calculate the false positive rate

          # We assign a rank to each individual
          rank <- seq(nrow(df_class),1,-1)
          # We create the sum of positive ranks using "positive" (which is equal to 1 if positive)
          sum_rkpos <- sum(rank * df_class$positive)
          # We calculate the Mann-Whitney statistic
          U = sum_rkpos - (npos*(npos+1))/2
          # We calculate the AUC
          auc <- U / (npos*nneg)

          # We create a label which includes the class studied and the AUC value
          auc_class = paste0(current_class, " (AUC =", round(auc,2),")")

          # We create a dataframe with the current class, the rate of true positives and false positives.
          data.frame(
            class = auc_class,
            tfp = tfp,
            tvp = tvp
          )
        }) %>% bind_rows() # We combine the dataframes one below the other

        # We create the graph with a line for each class
        ggplot(roc_data, aes(x = tfp, y = tvp, color = class)) +
          geom_line(size=1, alpha=0.8) + # For each class, we visualize the ROC Curve
          geom_abline(slope = 1, intercept = 0, linetype="dashed", size=0.7) +
          labs(title = "One-vs-Rest Multi-class ROC Curve", x = "False Positive Rate", y = "True Positive Rate", color = "Classes") +
          theme_minimal()

      } else if (private$k == 2){ # If there are only two classes, only one curve is displayed
        if (is.null(positive)){ # We check that the user has correctly indicated the class considered positive
          stop("You must indicate the positive label")
        }
        if (!(positive %in% private$classes)){ # We check that the indicated class is indeed one of the classes of the target variable
         stop(paste("You must use one of the existing classe: ", private$classes))
        }

        label_positive = as.character(positive) # We transform the "positive" arguments into characters
        # We extract the assignment probabilities
        proba_df <- as.data.frame(self$predict_proba(X))
        # We extract the assignment probabilities for the class considered positive
        proba_positive <- proba_df[[label_positive]]
        # We create a dataframe with the real classes and the assignment probabilities for the class considered positive
        df_class <- data.frame(proba_positive = proba_positive, y_test  = y_test)
        # We create a variable which takes 1 if the real class of the individual is the class considered positive
        df_class <- df_class %>% mutate(positive = ifelse(y_test == label_positive, 1, 0))
        # We reorganize the dataframe in descending order of assignment probabilities
        df_class <- df_class[order(-df_class$proba_positive), ]

        # We calculate the number of individuals having the class considered positive as the real class
        npos <- sum(df_class$positive)
        # We calculate the number of individuals not having the class considered positive as a real class
        nneg <- sum(1-df_class$positive)
        tvp <- cumsum(df_class$positive) / npos # We calculate the rate of true positives
        tfp <- cumsum(1 - df_class$positive) / nneg # We calculate the false positive rate

        # We assign a rank to each individual
        rank <- seq(nrow(df_class),1,-1)
        # We create the sum of positive ranks using "positive" (which is equal to 1 if positive)
        sum_rkpos <- sum(rank * df_class$positive)
        # We calculate the Mann-Whitney statistic
        U = sum_rkpos - (npos*(npos+1))/2
        # We calculate the AUC
        auc <- U / (npos*nneg)

        # We create a title with the class considered positive and the AUC
        auc_title <- paste0("ROC Curve - Positive: ", label_positive," (AUC =", round(auc,2),")")

        # We create a dataframe with the current class, the rate of true positives and false positives.
        roc_graph <- data.frame(
          tfp = tfp,
          tvp = tvp)

        # We create the graph for the ROC curve
        ggplot(roc_graph, aes(x = tfp, y = tvp)) +
          geom_line(size=1, color="#d66bb3") +
          geom_abline(slope = 1, intercept = 0, linetype="dashed", size=0.7) +
          labs(title = auc_title, x = "False Positive Rate", y = "True Positive Rate") +
          theme_minimal()
      }
    },

    #' Visualize Confusion Matrix for Bernoulli Naive Bayes Classifier
    #'
    #' This method creates a visual representation of the confusion matrix for the test dataset
    #' using the Bernoulli Naive Bayes Classifier.
    #'
    #' @param X A matrix containing the test features.
    #' @param y_test A vector containing the actual class labels of the test set.
    #'
    #' @details The method first checks if the model has been trained.
    #'          It then predicts the class labels for the test set using the specified prediction type
    #'          and constructs a confusion matrix. This matrix is visualized using the `corrplot` package,
    #'          with a color gradient representing the frequency of each cell in the matrix. The method
    #'          handles missing classes by adding zero-filled columns to ensure a complete matrix.
    #'
    #' @examples
    #' # Assuming 'X' and 'y_test' are your preprocessed test dataset and actual labels
    #' bernoulli_nb$plot_confusionmatrix(X, y_test)
    #'
    #' @export
    plot_confusionmatrix = function(X, y_test) {
      # Extract the predicted classes
      y_pred = self$predict(X)

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
      corrplot(cm, method="color", col = palette_col, addCoef.col = 'black', is.corr = FALSE, addgrid.col = 'white', tl.col = 'black', tl.srt = 45, cl.pos = 'n', title="Confusion Matrix", mar=c(2,0,2,0))
    },

    #' Extract Most Significant Features
    #'
    #' This method extract and present the 10 most significant features (words) with their associated
    #' log-probabilities for a given class from the Bernoulli Naive Bayes model.
    #'
    #' @param class The class from which we want to extract the features.
    #'
    #' @return A dataframe of the best words and their log_probabiliities
    #'
    #' @details The method first checks if the model has been trained.
    #'          It then predicts the class labels for the test set and constructs a confusion matrix.
    #'          This matrix is visualized using the 'corrplot' package, with a color gradient representing
    #'          the frequency of each cell in the matrix. The method handles missing classes by adding zero-filled columns to ensure a complete matrix.
    #'
    #' @examples
    #' # Assuming 'class' is a specific class from our dataset.
    #' bernoulli_nb$top_logproba("YourClass")
    #'
    #' @export
    top_logproba = function(class) {

      # Check that the fit has been done
      if (!private$is_fitted) {
        stop("You must fit your classifier first. Try Bernoulli_classifier$fit(X_train_vectorized, y_train)")
      }

      # matrix of our features log_probabilities for the chosen class
      class_feature_log_prob = t(as.matrix(private$feature_log_prob[class,]))

      # index of the 10 words with the highest log_probabilities
      significant_words_index <- order(class_feature_log_prob, decreasing = TRUE)[1:10]

      # all of our features
      vocab = private$train_features

      # dataframe of our 10 most significant words with their log_probabilities
      highest_words_log_probs = data.frame(Word = vocab[significant_words_index],
                                                Log_Probability = class_feature_log_prob[significant_words_index])

      return(highest_words_log_probs)
    }
  ),
  private = list(
  # (Laplace/Lidstone) smoothing parameter
  alpha=1.0,

  # Threshold for binarizing (mapping to booleans) of sample features
  binarize=0.0,

  # Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
  fit_prior=TRUE,
  # Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
  class_prior=NULL,
  # The number of chosen cores.
  n_cores = 1,
  # Classes in my target variable
  classes = NULL,
  # Occurence of each class
  class_count = NULL,
  # Features in my train set
  train_features = NULL,
  # Matrix of occurences of each features in each class
  feature_count = NULL,
  # Coefficients of the ranking functions
  feature_log_prob = NULL,
  # Initiate log proba for each class
  class_log_prior = NULL,
  # Boolean to check if our model is fitted or not
  is_fitted = FALSE,
  # Number of columns
  p = 0,
  # Number of classes
  k = 0,
  # Number of documents
  n = 0,
  # Training explanatory variables
  X_train = NULL,
  # Training target variable
  y_train = NULL,
  # Confusion matrix of the train sample
  cm_train = NULL,
  # Error rate for the train sample
  error_train = NULL,
  # Lower bound of the confidence interval of the training sample
  li_train = NULL,
  # Higher bound of the confidence interval of the training sample
  ls_train = NULL,
  # Recall for each class for the learning sample
  recall_train = NULL,
  # Precision for each class for the training sample
  precision_train = NULL
  )
)
