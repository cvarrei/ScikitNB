CategoricalNB <- R6Class("CategoricalNB",
                         public = list(
                           #' Initialize Categorical Naive Bayes Classifier
                           #'
                           #' This method initializes an instance of the Categorical Naive Bayes Classifier.
                           #' It sets up the environment  to be used for computation and optionally provides
                           #' instructions for using the classifier.
                           #'

                           #' @param verbose A logical value indicating whether to print detailed messages
                           #'        about the initialization process. Default is TRUE.
                           #'
                           #' @details When `verbose` is TRUE, the function prints
                           #'          instructions on the console for preprocessing and using the classifier.
                           #'
                           #' @examples
                           #' # Initialize the Categorical Naive Bayes Classifier with verbose output
                           #' catnb <- CategoricalNB$new()
                           #'
                           #' @export
                           initialize = function(verbose=T) {
                             private$my_model <- NULL
                             private$my_model_preprocess <- NULL
                             private$my_df_class_log_proba <- data.frame()
                             private$my_n_observations <- NULL
                             private$my_y_predict <- NULL
                             private$my_cm <- NULL
                             private$my_error <- NULL
                             private$my_li <- NULL
                             private$my_ls <- NULL
                             private$my_k_class <- NULL
                             private$my_class <- NULL
                             private$my_p <- NULL
                             private$my_class_func  <- NULL
                             private$my_vector_unique_value <- c()

                             if (verbose == TRUE){
                               cat("The Categorical Na?ve Bayes Classifier has been correctly instanciated. \n")
                               cat("\n")
                               cat("\n")
                               cat("Please, follows these steps: \n")
                               cat("   1) Preprocess your explanatory variables 'train <- preprocessing_train(X_train, y_train)' \n")
                               cat("   X_train_d <- train[[1]] \n")
                               cat("   y_train <- train[[2]] \n")
                               cat("   2) Fit your data with the following 'fit(X_train_d, y_train)' \n")
                               cat("   3) Preprocess your explanatory variables from your test sample 'X_test_d <- preprocessing_test(X_test)' \n")
                               cat("   4) Predict on your test sample 'predict(X_test_d)' \n")
                               cat("   5) Test the performance 'score(X_test_d, y_test)' \n")
                             }
                           },

                           #' Convert continuous data into discrete data for the train data
                           #'
                           #' This method preprocesses the training dataset for use with the Categorical Naive Bayes Classifier.
                           #' It includes checks for data consistency and applies necessary transformations.
                           #'
                           #' @param X The explanatory variables.
                           #' @param y The target variables.
                           #'
                           #' @return A list containing the discretized variables and the y variable.
                           #'
                           #' @details The discretization of data is an important and essential step in the pre-processing of variables
                           #'          continuous before the application of the naive Bayesian categorical model.
                           #'          For this, we used the supervised discretization method based on the principle of Minimum
                           #'          Description Length Principle (MDLP). The MDLP seeks to find the optimal discretization by minimizing the
                           #'          length of the model description while maximizing the separation between classes.
                           #'          It may happen, in certain cases, that the MDLP function does not find cutting terminals
                           #'          and returns a single category. We have chosen to deal with this case by deleting these columnsas they do not add any information
                           #'          for the classification.
                           #'          The user will be notified when the summary is displayed.
                           #'
                           #' @examples
                           #' Assuming 'X_train' is your explanatory variables
                           #' and 'y_train' the target variable.
                           #' catNB <- CategoricalNB$new()
                           #' list_catNB <- catNB$preprocessing_train(X_train,y_train)
                           #' X_train_d <- list_catNB[[1]]
                           #' y_train <- list_catNB[[2]]
                           #'
                           #' @export
                           preprocessing_train = function(X, y){

                             if (all(sapply(X, is.numeric))) {
                               stop("All the explanatory variables are numerical, better use the object 'GaussianNB'.")
                             }
                             if (any(is.na(X))) {
                               stop("Missing values detected in the explanatory variables.")
                             }

                             # Check if one of the explanatory variables have a unique value (constant across all individuals)
                             if (any(sapply(X, function(x) length(unique(x)) == 1))) {
                               warning("One or more explanatory variables have a constant value.")
                             }

                             # For each numerical column do the discretization
                             for (col in colnames(X)) {
                               if (is.numeric(X[,col])) {
                                 # Discretization
                                 discPW <- mdlp(cbind(X[col], y))
                                 # Assignment of discretization without the y
                                 X[col] <- discPW$Disc.data[1]
                                 if(discPW$cutp == "All"){
                                   # Adding the column to the list of deleted columns
                                   private$my_vector_unique_value <- c(private$my_vector_unique_value, col)
                                   # Deleting the column
                                   X <- X[, !(names(X) %in% c(col))]
                                 } else {
                                   # Registration of bounds + infinite bounds
                                   # discPW$cutp[[1]] -> Sublist
                                   private$my_model_preprocess <- c(private$my_model_preprocess, list(c(-Inf, discPW$cutp[[1]], +Inf)))
                                 }
                               }
                             }
                             # Convert all columns to characters
                             X <- as.data.frame(lapply(X, as.character))
                             X <- X %>%
                               mutate_if(is.character, as.factor)
                             return(list(X, as.factor(y)))
                           },

                           #' Convert continuous data into discrete data for the predicted data
                           #'
                           #' @param X The explanatory variables.
                           #'
                           #' @return The discretized dataframe.
                           #'
                           #' @details This function will discretize the continuous
                           #'         explanatory columns according to the limits learnt
                           #'         during the preprocessing on the training sample (preprocessing_train()).
                           #'         Be careful, you must first preprocess your training sample.
                           #'
                           #' @examples
                           #' Assuming 'X_test' is your explanatory variables
                           #'
                           #' X_test_d <- catNB$preprocessing_test(X_test)
                           #'
                           #' @export
                           preprocessing_test = function(X){

                             if (all(sapply(X, is.numeric))) {
                               stop("All the explanatory variables are numerical, better use the object 'GaussianNB'.")
                             }

                             # Deletion of single-valued columns
                             for(col in private$my_vector_unique_value){
                               X <- X[, !(names(X) %in% c(col))]
                             }

                             # For each numeric column
                             cpt = 1
                             for (i in 1:ncol(X)) {
                               if (is.numeric(X[, i])) {
                                 # Discretization
                                 # include.lowest = FALSE --> Do not include bounds
                                 X[i] = cut(X[, i], breaks = private$my_model_preprocess[[cpt]], include.lowest = FALSE, labels=1:(length(private$my_model_preprocess[[cpt]])-1))
                                 cpt = cpt + 1
                               }
                             }
                             return(X)
                           },

                           #' Train Categorical Naive Bayes Classifier on training data
                           #'
                           #' This method trains the Categorical Naive Bayes Classifier using the provided training data.
                           #'
                           #' @param X_train The explanatory variables.
                           #' @param y_train The target variables.
                           #' @param lmbd Optional constant used for Laplace smoothing.
                           #'             By default 1.
                           #'
                           #'
                           #' @return The method does not return anything but updates the model's internal state
                           #'         with the training results, including feature statistics and performance metrics.
                           #'
                           #' @examples
                           #' Assuming 'X_train' is your preprocessed training dataset
                           #' and 'y_train'is your training target.
                           #'
                           #' catNB$fit(X_train, y_train)
                           #'
                           #' @export
                           fit = function(X_train, y_train, lmbd = 1) {
                             # Checks
                             if (is.factor(y_train) == FALSE) {
                               stop("The target variables must be factorized.")
                             }
                             if (is.data.frame(X_train) == FALSE) {
                               stop("The explanatory variables must be within a dataframe.")
                             }
                             if (nrow(X_train) != length(y_train)) {
                               stop("The vector of the target variable does not have the same size as the dataframe of explanatory variables.")
                             }
                             if(nlevels(y_train) < 2){
                               stop("The target variable must have more than one class.")
                             }
                             if (lmbd < 0) {
                               stop("The lmbd variable must be greater than or equal to 0.")
                             }
                             if (all(sapply(X_train, is.numeric))) {
                               stop("All the explanatory variables are numerical, better use the object 'GaussianNB'.")
                             }
                             if (any(is.na(y_train))) {
                               stop("Missing values detected in the target variable.")
                             }

                             # Initialization of global variables
                             private$my_k_class = nlevels(y_train) # The number of classes to predict
                             private$my_class <- levels(y_train)
                             private$my_p <- ncol(X_train)
                             private$my_n_observations <- nrow(X_train) # The number of individuals

                             # Class frequency table
                             class_counts <- table(y_train)

                             # Model initialization
                             nb_model <- list()

                             # A priori probabilities of classes -> %
                             nb_model$class_probs <- (class_counts + lmbd) / (sum(class_counts) + lmbd * private$my_k_class)

                             # For each variable, calculation of conditional probabilities
                             for (feature_name in names(X_train)) {
                               feature_probs <- table(X_train[, feature_name], y_train)

                               # Counts the number of categories of the feature_name variable
                               num_modalities <- length(unique(X_train[[feature_name]]))

                               for (i in 1:length(class_counts)) {
                                 feature_probs[,i] <- (feature_probs[,i] + lmbd) /
                                   (class_counts[i] + lmbd*num_modalities)
                               }
                               nb_model[[feature_name]] <- feature_probs
                             }

                             # Classification functions
                             class_func <- data.frame()

                             constant <- log(nb_model$class_probs)

                             # Variable traversal
                             for (feature_name in names(nb_model)[-1]) {
                               # Number of modalities of the variable
                               nb_mod = nrow(nb_model[[feature_name]])
                               if (nb_mod > 1) {
                                 for (i in 1:(nb_mod - 1)) {
                                   desc = paste(feature_name, rownames(nb_model[[feature_name]])[i], sep = "_")

                                   # Calculations on model values by modality
                                   func = round(log(nb_model[[feature_name]][i,]) - log(nb_model[[feature_name]][nb_mod,]),3)
                                   func[is.infinite(func)] <- NaN
                                   class_func <- rbind(class_func, c(desc, func))
                                 }
                               }
                               constant <- round(constant + log(nb_model[[feature_name]][nb_mod,]),3)
                             }

                             # Adding the constant
                             class_func <- rbind(c("constant", constant), class_func)
                             colnames(class_func) <- c("Descriptor", private$my_class)

                             # Saving the class_func
                             private$my_class_func <- class_func

                             # Saving the model
                             private$my_model <- nb_model

                             # Predictions
                             private$my_y_predict <- self$predict(X_train)

                             # Confusion matrix
                             private$my_cm <- table(y_train, private$my_y_predict)

                             # Error rate
                             private$my_error <- 1 - sum(diag(private$my_cm))/sum(private$my_cm)

                             # 95% confidence interval
                             et <- sqrt(private$my_error*(1-private$my_error)/private$my_n_observations)
                             z <- qnorm(0.975)
                             private$my_li <- private$my_error - z*et
                             private$my_ls <- private$my_error + z*et

                             cat("The training has been completed correctly!\n\n")
                           },

                           #' Predict Class Labels Using Categorical Naive Bayes Classifier
                           #'
                           #' This method predicts the class labels for each observation in the test dataset
                           #' using the Categorical Naive Bayes Classifier.
                           #'
                           #' @param data_test The dataset on which we wish to predict.
                           #'
                           #' @return The prediction as a dataframe.
                           #'
                           #'
                           #' @examples
                           #' Assuming 'X_test' is your preprocessed test dataset
                           #' Ensure that the model has been trained using fit method
                           #'
                           #' y_pred <- catnb$predict(X_test)
                           #'
                           #' @export
                           predict = function(data_test) {

                             if (is.null(private$my_model)) {
                               stop("You need to run the 'fit' method first to get predictions.")
                             }

                             # Initialization of proba
                             private$my_df_class_log_proba <- NULL

                             predicted_test <- sapply(1:nrow(data_test), function(i) {
                               new_data <- data_test[i, ]

                               class_probs <- log(private$my_model$class_probs)

                               # For each observation characteristic
                               for (feature_name in names(new_data)) {
                                 feature_probs <- private$my_model[[feature_name]]
                                 feature_value <- new_data[[feature_name]]
                                 # Addition because log(a*b)=log(a)+log(b)
                                 class_probs <- class_probs + log(feature_probs[feature_value, ])
                               }

                               # Calculation of probabilities
                               private$my_df_class_log_proba <- rbind(private$my_df_class_log_proba, class_probs - log(sum(exp(class_probs))))

                               # Selection of the class with the highest percentage
                               predicted_class <- names(class_probs)[which.max(class_probs)]
                               return(predicted_class)
                             })

                             return(as.factor(predicted_test))
                           },

                           #' Calculate Log Probabilities for Each Class
                           #'
                           #' This method calculates the logarithm of the probabilities for each class
                           #' for each observation in the test dataset using the Categorical Naive Bayes Classifier.
                           #'
                           #' @param X A dataframe containing the preprocessed test set explanatory variables.
                           #'
                           #' @return A matrix of log probabilities, where each row corresponds to an observation
                           #'         and each column to a class.
                           #'
                           #' @examples
                           #' Assuming 'X_test' is your preprocessed test dataset
                           #' logs_probas <- catnb$predict_log_proba(X_test)
                           #'
                           #' @export
                           predict_log_proba = function(X) {
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }
                             self$predict(X)

                             cat("The log probabilities were calculated.\n\n")
                             return(private$my_df_class_log_proba)
                           },

                           #' Calculate Posterior Probabilities for Each Class
                           #'
                           #' This method computes the posterior probabilities for each class
                           #' for each observation in the test dataset using the Categorical Naive Bayes Classifier.
                           #'
                           #' @param X A dataframe containing the preprocessed test set explanatory variables.
                           #'
                           #' @return The dataframe which contains the probabilities.
                           #'
                           #' @examples
                           #' Assuming 'X_test' is your preprocessed test dataset
                           #' probas <- catnb$predict_proba(X_test)
                           #'
                           #' @export
                           predict_proba = function(X) {
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }

                             # Exponential log_proba
                             cat("The probabilities were calculated.\n\n")
                             return(exp(self$predict_log_proba(X)))
                           },

                           #' Print Short Summary of Categorical Naive Bayes Model
                           #'
                           #' This method prints a short summary of the Categorical Naive Bayes model,
                           #' including details about the target variable, sample size, and error rate on the training set.
                           #'
                           #' @details The method checks if the model has been trained and then prints
                           #'          the number of classes in the target variable, the number of observations
                           #'          in the sample, and the error rate on the training set along with its 95%
                           #'          confidence interval. This provides a quick overview of the model's training performance.
                           #'
                           #' @examples
                           #' # Assuming the Categorical Naive Bayes model has been trained
                           #'
                           #' catnb$print()
                           #'
                           #' # OR
                           #'
                           #' print(catnb)
                           #'
                           #' @export
                           print = function(){
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }
                             cat("The target variable has", private$my_k_class,
                                 "classes. The sample has", private$my_n_observations, "observations.")
                             cat("\n")

                             error = 1 - sum(diag(private$my_cm))/sum(private$my_cm)

                             cat(paste0("The model has an error rate on the training sample of ", round(error, 3), " [",round(private$my_li,3),";",round(private$my_ls,3),"] at CI 95%"))
                             cat("\n")
                             cat("\n")
                           },

                           #' Detailed Summary of Categorical Naive Bayes Model
                           #'
                           #' This method displays a comprehensive summary of the Categorical Naive Bayes model,
                           #' including details about the model's classes, features, performance metrics,
                           #' and classification function.
                           #'
                           #' @details The method checks if the model has been trained.
                           #'          It prints details such as the number of classes, class labels, number of features,
                           #'          size of the training sample, prior probabilities of each class,
                           #'          classifier performance metrics (confusion matrix, error rate, recall, precision) on the training dataset,
                           #'          and the classification function (coefficients, intercepts)
                           #'          and conditionnal distribution.
                           #'
                           #' @examples
                           #' # Assuming the Categorical Naive Bayes model has been trained
                           #'
                           #' catnb$summary()
                           #'
                           #' # OR
                           #'
                           #' summary(catnb)
                           #'
                           #' @export
                           summary = function(){
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }

                             cat("################ CATEGORICAL NAIVE BAYES CLASSIFIER ################ \n")
                             if (!is.null(private$my_vector_unique_value) && length(private$my_vector_unique_value) > 0) {
                               cat("\n")
                               cat("\n")
                               cat("For at least one of the variables, the supervised discretisation on the numerical columns below found no cut-off limits.")
                               cat("The following variables were eliminated because they did not provide new information (having a unique category after discretization).  : \n", private$my_vector_unique_value, "\n")
                             }
                             cat("\n")
                             cat("\n")
                             cat("Number of classes: ", private$my_k_class, "\n")
                             cat("Labels of classes:", private$my_class, "\n")
                             cat("Number of features: ", private$my_p, "\n")
                             cat("Number of observations in the validation sample: ", private$my_n_observations, "\n")
                             cat("\n")
                             cat("Prior for each class: \n")
                             prior <- private$my_model$class_probs
                             names(prior) <- private$classes
                             print(prior)

                             cat("CLASSIFIER PERFORMANCES ---------------------------------------")
                             cat("\n")
                             print(private$my_cm)
                             cat("\n")
                             cat("\n")
                             cat(paste0("Error Rate = ", round(private$my_error, 3), " [",round(private$my_li,3),";",round(private$my_ls,3),"] at CI 95%"))
                             cat("\n")
                             cat("Recall : \n")
                             print(round(diag(private$my_cm)/rowSums(private$my_cm),2))
                             cat("Precision : \n")
                             print(round(diag(private$my_cm)/colSums(private$my_cm),2))
                             cat("\n")
                             cat("\n")

                             cat("CLASSIFIER CHARACTERISTICS -------------------------------------")
                             cat("\n")
                             cat("Model description-----------------------------------------------")
                             cat("\n")
                             print(as.matrix(private$my_class_func), quote = FALSE)
                             cat("\n")
                             cat("Conditionnal distribution : P(descriptor / class attribute) ----")
                             cat("\n")
                             list_matrices <- private$my_model[-1]

                             # Modify row names and combine
                             combined_matrix <- do.call(rbind, lapply(names(list_matrices), function(name) {
                               matrix <- list_matrices[[name]]
                               rownames(matrix) <- paste(name, rownames(matrix), sep = "_")
                               return(matrix)
                             }))

                             # Print the combined matrix
                             print(combined_matrix)
                             cat("\n")
                             cat("\n")

                             #Create a list to extract the results by calling the function
                             results_summary <- list(
                               n_classes = private$my_k_class,
                               label_classes = private$my_class,
                               prior = prior,
                               features = private$my_p,
                               n = private$my_n_observations,
                               cm_train = private$my_cm,
                               error_train = private$my_error,
                               conf_error_train = c(private$my_li, private$my_ls),
                               recall_train = diag(private$my_cm)/rowSums(private$my_cm),
                               precision_train = diag(private$my_cm)/colSums(private$my_cm)
                             )

                             # Returns the list without displaying it
                             invisible(results_summary)
                           },

                           #' Evaluate Performance of Categorical Naive Bayes Classifier
                           #'
                           #' This method evaluates the performance of the Categorical Naive Bayes model on a test dataset.
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
                           #' # Assuming 'X_test' and 'y_test' are your test dataset and labels
                           #' # Ensure that the model has been trained using fit method
                           #'
                           #' results <- catnb$score(X_test, y_test)
                           #'
                           #' @export
                           score = function(X_test, y_test, verbose=T){
                             # Check that the training has been carried out correctly
                             if (is.null(private$my_model)) {
                               stop("You have to fit your model first! ( 1) cnb <- Categorical$new() ;2) cnb$fit(X,y)")
                             }

                             y_pred <- self$predict(X_test)

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

                           #' Visualize Confusion Matrix for Categorical Naive Bayes Classifier
                           #'
                           #' This method creates a visual representation of the confusion matrix for the test dataset
                           #' using the Categorical Naive Bayes Classifier. It allows selection between class label prediction
                           #' based on linear combination (`comblin`) or probabilities (`proba`).
                           #'
                           #' @param X A dataframe containing the test set explanatory variables.
                           #' @param y_test A vector containing the actual class labels of the test set.
                           #'
                           #' @details The method first checks if the model has been trained.
                           #'          It then predicts the class labels for the test set using the specified prediction type
                           #'          and constructs a confusion matrix. This matrix is visualized using the `corrplot` package,
                           #'          with a color gradient representing the frequency of each cell in the matrix. The method
                           #'          handles missing classes by adding zero-filled columns to ensure a complete matrix.
                           #'
                           #' @examples
                           #' # Assuming 'X_test' and 'y_test' are your preprocessed test dataset and actual labels
                           #'
                           #' catnb$plot_confusionmatrix(X_test, y_test)
                           #'
                           #' @export
                           plot_confusionmatrix = function(X, y_test) {
                             # On v?rifie que l'apprentissage a bien ?t? r?alis?
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }

                             # Creation of the confusion matrix between the actual and the predicted
                             y_pred <- self$predict(X)

                             cm_plot <- table(y_pred, y_test)

                             # Color palette
                             palette_col <- c('#fbff82', '#f9ef8a', '#f6df90', '#f3cf97', '#efbf9c', '#ebafa1', '#e79fa6', '#e28eab', '#dc7daf', '#d66bb3')

                             # Visualization with corrplot
                             corrplot(cm_plot, method="color", col = palette_col,  addCoef.col = 'black',  is.corr = FALSE,addgrid.col = 'white', tl.col = 'black',  tl.srt = 45, cl.pos = 'n', title="Matrice de Confusion", mar=c(2,0,2,0))
                           },


                           #' Generate ROC Curve for Categorical Naive Bayes Classifier
                           #'
                           #' This method plots the Receiver Operating Characteristic (ROC) curve for the test dataset
                           #' using the Categorical Naive Bayes Classifier. It handles both binary and multi-class scenarios.
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
                           #' # Assuming 'X_test' and 'y_test' are your preprocessed test dataset and actual labels
                           #' # This is the case of a binary classification
                           #'
                           #' catnb$plot_roccurve(X_test, y_test, positive = "YourPositiveClassLabel")
                           #'
                           #' @export
                           plot_roccurve = function(X, y_test, positive=NULL){
                             # Check that the training has been carried out correctly
                             if (is.null(private$my_model)) {
                               stop("You have to fit your model first! ( 1) cnb <- Categorical$new(X,y) ;2) cnb$fit()")
                             }

                             # If there are more than 2 classes, an OVR multi-class Roc curve will be displayed.
                             if (private$my_k_class > 2){
                               # Extract the classification probabilities
                               proba_df <- as.data.frame(as.matrix(self$predict_proba(X)))
                               # Extract the true classes of individuals
                               true_classes = levels(y_test)[y_test]

                               # For each class of the target variable,
                               roc_data <- lapply(private$my_class, function(current_class) {
                                 # Extract the classification probabilities
                                 proba_class <- proba_df[[current_class]]

                                 # Create a dataframe with the probabilities and the true class
                                 df_class <- data.frame(proba_class = proba_class, class = true_classes)
                                 # Create a variable which takes 1 if the individual has the class studied as his real class
                                 df_class <- df_class %>% mutate(positive = ifelse(class == current_class, 1, 0))

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
                             } else if (private$my_k_class == 2){

                               # Check that the user has indicated the class considered as positive
                               if (is.null(positive)){
                                 stop("You must indicate the positive label")
                               }

                               # Check that the user has indicated an existing class
                               if (!(positive %in% private$my_class)){
                                 stop(paste("You must use one of the existing classe: ", private$my_class))
                               }

                               # The "positive" argument is transformed into a character
                               label_positive = as.character(positive)

                               # The processing is the same as above.
                               proba_df <- as.data.frame(self$predict_proba(X))
                               proba_positive <- proba_df[[label_positive]]
                               df_class <- data.frame(proba_positive = proba_positive, y_test = y_test)
                               df_class <- df_class %>% mutate(positive = ifelse(y_test == label_positive, 1, 0))
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

                               roc_graph <- data.frame(tfp = tfp, tvp = tvp)

                               ggplot(roc_graph, aes(x = tfp, y = tvp)) +
                                 geom_line(linewidth=1, color="#d66bb3") +
                                 geom_abline(slope = 1, intercept = 0, linetype="dashed", linewidth=0.7) +
                                 labs(title = auc_title, x = "False Positive Rate", y = "True Positive Rate") +
                                 theme_minimal()
                             }
                           }
                         ),
                         private = list(
                           # The model
                           my_model = NA,
                           # The model of the discretization
                           my_model_preprocess = NA,
                           # Probabilities
                           my_df_class_log_proba = NA,
                           # Number of observations
                           my_n_observations = NA,
                           # Prediction of y values
                           my_y_predict = NA,
                           # Confusion matrix
                           my_cm = NA,
                           # Error rate
                           my_error = NA,
                           # Lower bound of the confidence interval
                           my_li = NA,
                           # Upper bound of the confidence interval
                           my_ls = NA,
                           # Number of classes
                           my_k_class = NA,
                           # Label of classes
                           my_class = NA,
                           # Number of features
                           my_p = NA,
                           # Model classification coefficients
                           my_class_func = NA,
                           # Columns deleted after discretization.
                           my_vector_unique_value = NA
                         )
)
