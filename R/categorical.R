#' Categorical Dataset with Binary Classification - Mushrooms Dataset
#'
#' This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms
#' in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981).
#' Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.
#' This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining
#' the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy."
#'
#' Donated to UCI ML 27 April 1987
#'
#' The dataset contains the following variables:
#' \describe{
#'   \item{cap-shape:}{bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s}
#'   \item{cap-surface:}{fibrous=f, grooves=g, scaly=y, smooth=s}
#'   \item{cap-color:}{brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y}
#'   \item{bruises?:}{bruises=t, no=f}
#'   \item{odor:}{almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s}
#'   \item{gill-attachment:}{attached=a, descending=d, free=f, notched=n}
#'   \item{gill-spacing:}{close=c, crowded=w, distant=d}
#'   \item{gill-size:}{broad=b, narrow=n}
#'   \item{gill-color:}{black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y}
#'   \item{stalk-shape:}{enlarging=e, tapering=t}
#'   \item{stalk-root:}{bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?}
#'   \item{stalk-surface-above-ring:}{fibrous=f, scaly=y, silky=k, smooth=s}
#'   \item{stalk-surface-below-ring:}{fibrous=f, scaly=y, silky=k, smooth=s}
#'   \item{stalk-color-above-ring:}{brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y}
#'   \item{stalk-color-below-ring:}{brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y}
#'   \item{veil-type:}{partial=p, universal=u}
#'   \item{veil-color:}{brown=n, orange=o, white=w, yellow=y}
#'   \item{ring-number:}{none=n, one=o, two=t}
#'   \item{ring-type:}{cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z}
#'   \item{spore-print-color:}{black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y}
#'   \item{population:}{abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y}
#'   \item{habitat:}{grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d}
#' }
#'
#' Class Labels: edible=e, poisonous=p
#'
#' @docType data
#' @usage data(categorical)
#' @format An object of class \code{"cross"}; see \code{\link[qtl]{read.cross}}.
#' @keywords datasets
#' @references Audobon Society Field Guide (1981);
#' @source \url{https://www.kaggle.com/datasets/uciml/mushroom-classification}
"categorical"
