CategoricalNB <- R6Class("CategoricalNB",

                         public = list(
                           #' Create a new instance of the CategoricalNB class.
                           initialize = function(verbose=T) {
                             #' The model.
                             private$my_model <- NULL

                             #' The model of the discretization.
                             private$my_model_preprocess <- NULL

                             #' Probabilities.
                             private$my_df_class_log_proba <- data.frame()

                             #' Number of observations.
                             private$my_n_observations <- NULL

                             #' Prediction of y values.
                             private$my_y_predict <- NULL

                             #' Confusion matrix.
                             private$my_cm <- NULL

                             #' Error rate.
                             private$my_error <- NULL

                             #' Lower bound of the confidence interval.
                             private$my_li <- NULL

                             #' Upper bound of the confidence interval.
                             private$my_ls <- NULL

                             #' Number of classes.
                             private$my_k_class <- NULL

                             #' Label of classes.
                             private$my_class <- NULL

                             #' Number of features.
                             private$my_p <- NULL

                             #' Model description.
                             private$my_class_func  <- NULL

                             #' Columns deleted after discretization.
                             private$my_vector_unique_value <- c()

                             if (verbose == TRUE){
                               cat("The Categorical Naive Bayes Classifier has been correctly instanciated. \n")
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
                           #' @param X The explanatory variables.
                           #' @param y The target variables.
                           #' @return The discretized dataframe.
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
                           #' @return The discretized dataframe.
                           preprocessing_test = function(X){

                             if (all(sapply(X, is.numeric))) {
                               stop("All the explanatory variables are numerical, better use the object 'GaussianNB'.")
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

                           #' Fit the model according to the given training data.
                           #'
                           #' @param X_train The explanatory variables.
                           #' @param y_train The target variables.
                           #' @param lmbd Optional constant used for Laplace smoothing.
                           #' @return The model.
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

                           #' Predict function.
                           #'
                           #' @param data_test The dataset on which we wish to predict.
                           #' @return The prediction as a dataframe.
                           predict = function(data_test, n_cores = 4) {

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

                           #' Predict log-probability estimates.
                           #'
                           #' @param X The dataset on which we wish the probability prediction.
                           #' @return The dataframe which contains the log-probabilities.
                           predict_log_proba = function(X) {
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }
                             self$predict(X)

                             cat("The log probabilities were calculated.\n\n")
                             return(private$my_df_class_log_proba)
                           },

                           #' Probability estimates.
                           #'
                           #' @param X The dataset on which we wish the probability prediction.
                           #' @return The dataframe which contains the probabilities.
                           predict_proba = function(X) {
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }

                             # Exponential log_proba
                             cat("The probabilities were calculated.\n\n")
                             return(exp(self$predict_log_proba(X)))
                           },

                           #' Print function.
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

                           #' Summary function.
                           summary = function(){
                             if (is.null(private$my_model)) {
                               stop("You need to train the model.")
                             }

                             cat("################ CATEGORICAL NAIVE BAYES CLASSIFIER ################ \n")
                             if (!is.null(private$my_vector_unique_value) && length(private$my_vector_unique_value) > 0) {
                               cat("\n")
                               cat("\n")
                               cat("For at least one of the variables, the supervised discretisation on the numerical columns below found no cut-off limits.")
                               car("The following variables were eliminated because they did not provide new information (having a unique category after discretization).  : \n", private$my_vector_unique_value, "\n")
                             }
                             cat("\n")
                             cat("\n")
                             cat("Number of classes: ", private$my_k_class, "\n")
                             cat("Labels of classes:", private$my_class, "\n")
                             cat("Number of features: ", private$my_p, "\n")
                             cat("Number of observations in the validation sample: ", private$my_n_observations, "\n")
                             cat("\n")

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
                           },

                           #' Evaluate Performance of Categorical Naive Bayes Classifier.
                           #'
                           #' @param X_test A dataframe containing the test set explanatory variables.
                           #' @param y_test A vector containing the actual class labels of the test set.
                           #' @param verbose A boolean indicating whether to print detailed output (default is TRUE).
                           #'
                           #' @return A list containing the confusion matrix, error rate, 95% confidence interval for the error rate, recall, and precision for each class.
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
                           #' Visualization of the confusion matrix.
                           #'
                           #' @param X The dataset X.
                           #' @param y_test y_test.
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
                           my_model = NA,
                           my_model_preprocess = NA,
                           my_df_class_log_proba = NA,
                           my_n_observations = NA,
                           my_y_predict = NA,
                           my_cm = NA,
                           my_error = NA,
                           my_li = NA,
                           my_ls = NA,
                           my_k_class = NA,
                           my_class = NA,
                           my_p = NA,
                           my_class_func = NA,
                           my_vector_unique_value = NA
                         )
)

# R6 does not have a summary generic method, so we have to add a small S3 method function to call the summary function from R6 object.
summary.CategoricalNB <- function(objectCNB) {
  objectCNB$summary()
}
