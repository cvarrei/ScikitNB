#' This function provides a summary for NaiveBayes R6 objects.
#'
#' @param nb An object of class GaussianNB, BernoulliNB, or CategoricalNB.
#' @param print_afdm A boolean to print the FAMD results in case the model is a Gaussian Naive Bayes Classifier on mixed data.
#' @details
#' R6 does not have a summary generic method, so we have to add a small S3 method function to call the summary function from R6 object.
#' @return Summary of the Naive Bayes Classifier object.
#' @export
summary <- function(nb, ...) {
  if (inherits(nb, "GaussianNB")) {
    nb$summary(...)
  } else if (inherits(nb, "BernoulliNB")) {
    nb$summary()
  } else if (inherits(nb, "CategoricalNB")) {
    nb$summary()
  } else {
    stop("The object is not of a recognized type (GaussianNB, BernoulliNB, or CategoricalNB).")
  }
}
