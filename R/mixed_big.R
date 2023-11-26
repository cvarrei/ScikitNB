#' Mixed Dataset with multiclass classification - Big Size - Healthcare Management Dataset
#'
#' This dataset is centered on healthcare management, specifically predicting Length of Stay (LOS) in hospitals during the Covid-19 pandemic. 
#' It includes various attributes of patients and hospital details aimed at optimizing treatment plans and resource allocation.
#'
#' @format A data frame of 83139 indivudals and 14 variables:
#' \describe{
#'   \item{Hospital_type_code}{"a", "b", "c", "d", "e", "f", "g"}
#'   \item{City_Code_Hospital}{"1", "2", ..., "13"}
#'   \item{Hospital_region_code}{"X", "Y", "Z"}
#'   \item{Available Extra Rooms in Hospital}{Numerical}
#'   \item{Department}{"anesthesia", "gynecology", "radiotherapy", "TB & Chest disease", "surgery"}
#'   \item{Ward_Type}{"S", "R", "Q", "P", "T", "U"}
#'   \item{Ward_Facility_Code}{"A", "B", "C", "D", "E", "F"}
#'   \item{Bed Grade}{"1", "2", "3", "4"}
#'   \item{Type of Admission}{"Trauma", "Urgent", "Emergency"}
#'   \item{Severity of Illness}{"Extreme", "Moderate", "Minor"}
#'   \item{Visitors with Patient}{Numerical. Number of Visitors with the patient}
#'   \item{Age}{"0-10", "11-20", ..., "91-100"}
#'   \item{Stay}{Target Variable : "0-10" "51-60" ">70"}
#' }
#' @docType data
#' @usage data(mixed_big)
#' @source \url{https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii}
#' 
"mixed_big"