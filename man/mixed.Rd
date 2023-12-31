% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mixed.R
\docType{data}
\name{mixed}
\alias{mixed}
\title{Mixed dataset with Binary Classification - Campus recruitment dataset}
\format{
A data frame (215 individuals and 13 variables) with the following variables:
\describe{
  \item{gender}{Gender: "M" (Male), "F" (Female)}
  \item{ssc_p}{Numerical. Secondary Education percentage - 10th Grade}
  \item{ssc_b}{Board of Education: "Others", "Central"}
  \item{hsc_p}{Numerical. Higher Secondary Education percentage - 12th Grade}
  \item{hsc_b}{Board of Education: "Others", "Central"}
  \item{hsc_s}{Specialization in Higher Secondary Education: "Commerce", "Science", "Arts"}
  \item{degree_p}{Numerical. Degree Percentage}
  \item{degree_t}{Field of degree education: "Sci&Tech", "Comm&Mgmt", "Others"}
  \item{workex}{Work Experience: "No", "Yes"}
  \item{etest_p}{Numerical. Employability test percentage (conducted by college)}
  \item{specialisation}{Post Graduation(MBA) Specialization: "Mkt&HR", "Mkt&Fin"}
  \item{mba_p}{Numerical. MBA percentage}
  \item{status}{Status of placement (Targer Variable): "Placed", "Not Placed" }
}
}
\source{
\url{https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement}
}
\usage{
data(mixed)
}
\description{
This dataset attributes related to student academic backgrounds and employment outcomes.
It includes demographic information, academic performance at different education levels, work experience, and employability test scores,
along with MBA specialization and placement status.
}
\keyword{datasets}
