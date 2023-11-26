AFDM <- R6Class("AFDM",
                public = list(
                  #' Initialize Method for AFDM
                  #'
                  #' @description
                  #' This method initializes an object of AFDM. It sets up the necessary
                  #' environment and variables for the object to function correctly. This method is
                  #' automatically called when a new object of this class is created.
                  #'
                  #' @details In the context of this R6 class, the initialize method may set up fields
                  #'          related to the FAMD model or other relevant data structures. It is essential
                  #'          for preparing the object for subsequent methods like 'fit_transform',
                  #'          'get_attributes', etc.
                  #'
                  #' @examples
                  #' # To create an instance of the class
                  #' afdm <- AFDM$new()
                  #'
                  #' @export
                  initialize = function(){
                  },
                  #' Fit and Transform Data using Factor Analysis of Mixed Data (FAMD)
                  #'
                  #' @description
                  #' This function performs a Factor Analysis of Mixed Data (FAMD) on a given dataset.
                  #' It either automatically determines the number of components to keep based on the
                  #' elbow method if 'n_compon' is not specified, or uses the specified number of components.
                  #'
                  #' @param X_df A dataframe containing the data to be analyzed.
                  #' @param n_compon An optional integer specifying the number of components to retain.
                  #'        If NULL, the function will automatically determine this number using the elbow method.
                  #' @return A dataframe containing the coordinates of individuals on the retained components.
                  #'         The number of columns corresponds to the number of components kept.
                  #' @details If 'n_compon' is NULL, the function performs FAMD on the entire dataset
                  #'          and uses the elbow method to determine the optimal number of components
                  #'          to retain. The elbow method involves finding the point where the
                  #'          addition of another component doesn't significantly explain more variance.
                  #'          This is done by calculating the maximum vertical distance between the eigen values curve
                  #'          and a straight line linking the highest and smallest eigen values.
                  #'          If 'n_compon' is specified, the function retains the specified number
                  #'          of components. The function always returns the individual coordinates
                  #'          based on the selected components.
                  #'
                  #' @examples
                  #' # Example usage with automatic component selection
                  #' df <- data.frame(a = rnorm(100), b = rnorm(100))
                  #' transformed_data <- afdm$fit_transform(df)
                  #'
                  #' # Example usage with specified number of components
                  #' transformed_data <- afdm$fit_transform(df, n_compon = 5)
                  #'
                  #' @export
                  fit_transform = function(X_df, n_compon=NULL){

                    X_df <- X_df %>%
                      mutate_if(is.character, as.factor)
                    X_df <- data.frame(X_df)
                    private$fit_value <- T
                    # Extract the training dataset and remove space in case.
                    colnames(X_df) <- gsub(" ", "_", colnames(X_df))
                    X_df <- X_df %>%
                          mutate_if(is.character, as.factor)
                    private$X <- X_df

                    # Extract the labels of the encoded one-hot categorical variables.
                    qualitative = Filter(is.factor, private$X) # On extrait seulement les variables qualitatives
                    # Apply one-hot encoding
                    dummy_qual <- dummyVars(" ~ .", data=qualitative)
                    qual_oc <- data.frame(predict(dummy_qual, newdata=qualitative))
                    # Extract the encoded one-hot variable names
                    private$qual_lab = colnames(qual_oc)

                    # If the component number has not been predefined, the algorithm will detect the location of the elbow
                    if (is.null(n_compon)){
                      # Fit the FAMD model
                      private$afdm_model <- FAMD(private$X, ncp = ncol(private$X), graph=F)

                      private$eigenvalue <- private$afdm_model$eig[,1] # Extract eigenvalues
                      private$prop <-  private$afdm_model$eig[,2] # Extract proportion of explained variance
                      private$cumsum <- private$afdm_model$eig[,3] # Extract cumulated proportion of variance explained
                      private$n_components <- length(private$eigenvalue) # Extract initial number of components

                      # Create a diagonal from the maximum value to the minimum value
                      diagonal <- seq(from = max(private$eigenvalue), to = min(private$eigenvalue), length.out = private$n_components)
                      # Calculate the distance between the diagonal and the eigenvalue to find the elbow
                      dist_fromdiag <- abs(diagonal - private$eigenvalue)
                      # Get the index of the eigenvalue which is farthest from the diagonal
                      private$kept_compon <- which.max(dist_fromdiag)
                      # Extract individual coordinates
                      return(private$afdm_model[["ind"]][["coord"]][,1:private$kept_compon])


                    } else {
                      # Realize the aFDM with the number of components specified.
                      private$afdm_model <- FAMD(private$X, ncp = n_compon, graph=F)

                      private$kept_compon <- n_compon
                      private$eigenvalue <- private$afdm_model$eig[,1]
                      private$prop <-  private$afdm_model$eig[,2]
                      private$cumsum <- private$afdm_model$eig[,3]
                      private$n_components <- n_compon

                      return(private$afdm_model[["ind"]][["coord"]][,1:private$kept_compon])

                    }

                  },

                  #' Get Attributes of the FAMD Model
                  #'
                  #' @description
                  #' This function returns the attributes of the Factor Analysis of Mixed Data (FAMD) model
                  #' that has been fitted to the dataset.
                  #'
                  #' @return Returns the FAMD model object with all its attributes.
                  #' @details This function is useful for retrieving the complete FAMD model object,
                  #'          which includes various components like eigenvalues, scores, loadings, etc.
                  #'          This object can be used for further analysis or inspection of the model.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' famd_model <- afdm$get_attributes()
                  #'
                  #' @export
                  get_attributes = function(){
                    if (private$fit_value == F){
                      stop("FAMD has not been fitted yet! ")
                    }
                    # Retrieve the attributes of the FAMD
                    return(private$afdm_model)
                  },

                  #' Get Characteristics of Variables in FAMD Model
                  #'
                  #' @description
                  #' This method retrieves the coordinates and squared cosines (cos2) of the dimensions retained
                  #' for both qualitative and quantitative variables in the FAMD model.
                  #'
                  #' @param verbose A logical value indicating whether to print the characteristics to the console.
                  #'        Default is TRUE.
                  #'
                  #' @return A list containing coordinates and cos2 for qualitative and quantitative variables.
                  #'         The list has four elements: `coord_qual`, `cos2_qual`, `coord_quant`, and `cos2_quant`.
                  #'
                  #' @details The method extracts the coordinates and cos2 from the FAMD model for both qualitative
                  #'          and quantitative variables. These characteristics are important for interpreting the
                  #'          factors in terms of the original variables. If `verbose` is TRUE, these details are
                  #'          printed to the console for easy viewing.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' var_characteristics <- afdm$get_var_characteristics()
                  #'
                  #' @export
                  get_var_characteristics = function(verbose=T){
                    # Retrieve the AFDM model attributes
                    afdm_model <- self$get_attributes()
                    # Extract coordinates and cos2 for qualitative variables
                    quali_coord <- afdm_model[["quali.var"]][["coord"]][,1:private$kept_compon]
                    rownames(quali_coord) <- private$qual_lab
                    quali_cos2 <- afdm_model[["quali.var"]][["cos2"]][,1:private$kept_compon]
                    rownames(quali_cos2) <- private$qual_lab

                    # Extract coordinates and cos2 for quantitative variables
                    quanti_coord <- afdm_model[["quanti.var"]][["coord"]][,1:private$kept_compon]
                    quanti_cos2 <- afdm_model[["quanti.var"]][["cos2"]][,1:private$kept_compon]

                    # Compile characteristics into a list
                    charac_var <- list(coord_qual = quali_coord,
                                       cos2_qual = quali_cos2,
                                       coord_quant = quanti_coord,
                                       cos2_quant = quanti_cos2
                    )

                    if (verbose == T){
                      cat("Coordinates of the qualitative features \n")
                      print(quali_coord)
                      cat("\n")
                      cat("Quality of Representation of the qualitative features - Cos2 \n")
                      print(quali_cos2)
                      cat("\n")
                      cat("Correlation of the quantitative features \n")
                      print(quanti_coord)
                      cat("\n")
                      cat("Quality of Representation of the quantitative features - Cos2 \n")
                      print(quanti_cos2)
                      cat("\n")
                    }

                    return(charac_var)

                  },



                  #' Project Supplementary Individuals in Factorial Plane
                  #'
                  #' @description
                  #' This function projects supplementary individuals (new observations) onto the factorial plane
                  #' defined by a previously fitted Factor Analysis of Mixed Data (FAMD) model.
                  #'
                  #' @param X_test A dataframe containing the new observations to be projected.
                  #' @return A dataframe containing the coordinates of the supplementary individuals
                  #'         in the space of the retained components of the FAMD model.
                  #' @details The function applies one-hot encoding to the test dataset and then
                  #'          scales it using the mean and standard deviation calculated from the training dataset.
                  #'          It then projects these scaled observations onto the factorial plane using the
                  #'          coefficients from the FAMD model.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' new_data <- data.frame(a = rnorm(50), b = rnorm(50))
                  #' projected_data <- afdm$coord_supplementaries(new_data)
                  #'
                  #' @export
                  coord_supplementaries = function(X_test){
                    # Remove spaces in case
                    X_train <- private$X
                    colnames(X_train) <- gsub(" ", "_", colnames(X_train))
                    colnames(X_test) <- gsub(" ", "_", colnames(X_test))
                    # Apply a one-hot encoding transformation
                    dummy <- dummyVars(" ~ .", data=X_train)
                    X_oc <- data.frame(predict(dummy, newdata=X_train))
                    # Extract the mean for each variable after the one-hot encoding transformation
                    mean_X_oc = apply(X_oc, 2, mean)
                    # Extract the standard deviation for each variable after the one-hot encoding transformation
                    sd_X_oc = apply(X_oc, 2, sd)

                    X_test_oc <- data.frame(predict(dummy, newdata=X_test))


                    # Scale all the variables using the means and standard deviations calculated on the training sample.
                    X_test_sc <- scale(X_test_oc, center=mean_X_oc, scale=sd_X_oc)


                    # Extract the transposed matrix of coefficients for categorical variables for the dimensions retained
                    coef_comp_qual <- t(private$afdm_model[["quali.var"]][["coord"]][,1:private$kept_compon])
                    # Apply the labels of the encoded one-hot categorical variables as variable names to match the formatting.
                    colnames(coef_comp_qual) <- private$qual_lab
                    # Extract the transposed matrix of coefficients for the quantitative variables for the retained dimensions
                    coef_comp_quant <- t(private$afdm_model[["quanti.var"]][["coord"]][,1:private$kept_compon])

                    # Global coefficient matrix
                    coef_afdm <- cbind(coef_comp_qual, coef_comp_quant)
                    # Extract the order of the variables from this matrix of coefficients
                    order_columns <- colnames(coef_afdm)
                    # Transpose the coefficient matrix for all variables
                    coef_afdm_t <- t(coef_afdm)

                    # Rearrange the matrix of scaled explanatory variables
                    X_test_sc_ordered <- X_test_sc[, colnames(coef_afdm)]

                    # Obtain the coordinates of our additional individuals by multiplying the matrices
                    coord_Xtest <- X_test_sc_ordered %*% coef_afdm_t

                    return(coord_Xtest)
                  },

                  #' Plot Eigenvalues (Scree Plot)
                  #'
                  #' @description
                  #' This function plots a scree plot of eigenvalues for each factor/component in the FAMD model.
                  #' It helps in visualizing the importance of each factor.
                  #'
                  #' @return A ggplot object representing the scree plot. The plot shows the eigenvalues
                  #'         of each factor as bars, making it easy to visualize the 'elbow'.
                  #' @details The scree plot is a bar plot of eigenvalues of each factor in descending order.
                  #'          It is used to determine the number of factors to keep in factor analysis.
                  #'          The 'elbow' in the plot indicates the point after which the addition of
                  #'          new factors contributes less to explaining the variance in the data.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' afdm$plot_eigen()
                  #'
                  #'
                  #' @export
                  plot_eigen = function(){
                    data_eigen <- data.frame(
                      num_compon = seq(1:private$n_components),
                      label = paste("Facteur", seq(1:private$n_components)),
                      Eigenvalue = private$eigenvalue
                    )

                    p <- ggplot(data_eigen, aes(x=as.factor(num_compon), y=Eigenvalue))+
                      geom_bar(stat="identity", fill="darkslategray4") +
                      geom_line(aes(x=num_compon, y=Eigenvalue),color="black", linewidth=0.7)+
                      geom_point(aes(x=num_compon, y=Eigenvalue),color="black", size=3)+
                      ylab("Eigenvalues") +
                      xlab("Dimensions")+
                      ggtitle("Scree Plot")+
                      scale_y_continuous(breaks = seq(0, max(private$eigenvalue), 1))+ # Nous param?trons les ticks de l'axe y.
                      theme_minimal()


                    return(p)
                  },

                  #' Scree Plot with Automatic Elbow Detection
                  #'
                  #' @description
                  #' This function extends the basic scree plot by adding a diagonal line to assist
                  #' in the automatic detection of the 'elbow' point.
                  #'
                  #' @return A ggplot object representing the scree plot with additional diagonal line.
                  #'         The plot shows the eigenvalues of each factor as bars and a diagonal line
                  #'         to facilitate the detection of the elbow point.
                  #' @details This plot is similar to the basic scree plot but includes a diagonal line
                  #'          that helps in identifying the elbow point automatically. This is useful for
                  #'          determining the optimal number of factors to keep in an automated manner.
                  #'          The plot also includes labels showing the distance between the eigenvalues
                  #'          and the diagonal, providing a clear visual cue for the elbow.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' afdm$plot_diagonal_auto()
                  #'
                  #'
                  #' @export
                  plot_diagonal_auto = function(){
                    # Extract the diagonal value
                    diagonal  = seq(from = max(private$eigenvalue), to = min(private$eigenvalue), length.out = private$n_components) # On extrait les valeurs de la diagonale
                    # Extract the distance between the diagonal and the eigenvalue curves
                    dist_diag = diagonal - private$eigenvalue

                    data_eigen <- data.frame(
                      num_compon = seq(1:private$n_components),
                      label = paste("Facteur", seq(1:private$n_components)),
                      Eigenvalue = private$eigenvalue,
                      diagonal = diagonal,
                      dist = as.character(round(dist_diag,2)),
                      y_label = private$eigenvalue + (dist_diag/2) # Label distance position
                    )

                    p <- ggplot(data_eigen, aes(x=as.factor(num_compon), y=Eigenvalue))+
                      geom_bar(stat="identity", fill="darkslategray4") +
                      geom_line(aes(x=num_compon, y=Eigenvalue),color="black", linewidth=0.7)+
                      geom_line(aes(x=num_compon, y=diagonal),color="red", linewidth=0.7)+
                      geom_point(aes(x=num_compon, y=Eigenvalue),color="black", size=3)+
                      geom_point(aes(x=num_compon, y=diagonal),color="red", size=3)+
                      # Display segments that join each eigenvalue with the corresponding value of the diagonal
                      geom_segment( aes(x=num_compon, xend=num_compon, y=Eigenvalue, yend=diagonal), color="red", alpha=0.5, linewidth=0.7)+
                      # Display labels with the distance between the diagonal and the eigenvalue
                      geom_label(aes(x = num_compon, y = y_label, label = as.character(dist)),size = 3, fill = "white")+
                      ylab("Eigenvalues") +
                      xlab("Dimensions") +
                      ggtitle("Scree Plot with diagonal distance - Automatic Elbow Selection")+
                      scale_y_continuous(breaks = seq(0, max(private$eigenvalue), 1))+
                      theme_minimal()

                    return(p)
                  },

                  #' Plot Cumulative Proportion of Inertia
                  #'
                  #' @description
                  #' This function creates a plot displaying the cumulative proportion of inertia explained by
                  #' each factor in the Factor Analysis of Mixed Data (FAMD) model.
                  #'
                  #' @return A ggplot object representing the cumulative proportions of inertia.
                  #'         The plot shows the cumulative proportion of inertia for each factor as bars,
                  #'         which helps in understanding the amount of variance explained by each factor.
                  #' @details The plot is particularly useful for assessing the contribution of each factor
                  #'          to the total variance explained.
                  #'
                  #' @examples
                  #' # Assuming 'fit_transform' has been already used to fit the model:
                  #' afdm$plot_cumsumprop()
                  #'
                  #'
                  #' @export
                  plot_cumsumprop = function(){
                    data_eigen <- data.frame(
                      num_compon = seq(1:private$n_components),
                      label = paste("Facteur", seq(1:private$n_components)),
                      cumul_prop_inertia = private$cumsum
                    )

                    p <- ggplot(data_eigen, aes(x=as.factor(num_compon), y=cumul_prop_inertia))+
                      geom_bar(stat="identity", fill="cadetblue1") +
                      geom_line(aes(x=num_compon, y=cumul_prop_inertia),color="black", linewidth=0.7)+
                      geom_point(aes(x=num_compon, y=cumul_prop_inertia),color="black", size=3)+
                      xlab("Dimensions") +
                      ylab("Cumulative proportion of explained variance (%)") +
                      ggtitle("Cumulative proportion of explained variance plot")+
                      scale_y_continuous(breaks = seq(0, 100, 10))+
                      theme_minimal()

                    return(p)
                  }

                ),
                private = list(
                  #  Mixed data type. The model object of the AFDM.
                  afdm_model = NA,
                  # The coordinates of the individuals in the AFDM space.
                  coord_ind = NA,
                  # The cumulative sums of explained variance.
                  cumsum = NA,
                  # The eigenvalues of the AFDM.
                  eigenvalue = NA,
                  # Boolean indicating whether the AFDM model has been fit.
                  fit_value = F,
                  # The components kept after the AFDM analysis.
                  kept_compon = NA,
                  # The number of components used in the AFDM.
                  n_components = NA,
                  # The proportion of variance explained by each component in the AFDM.
                  prop = NA,
                  # The names of the qualitative features
                  qual_lab = NA,
                  # The input data used for fitting the AFDM model.
                  X = NA,
                  # The one-hot encoded version of the supplementary individuals dataset
                  X_test_oc = NA
                )
)
