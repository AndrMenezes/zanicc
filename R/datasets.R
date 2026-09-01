#' @name pollen_climate
#'
#' @title Pollen-climate data set
#'
#' @description
#' The data comprises \eqn{7{,}832} samples collected from locations across the
#' Northern Hemisphere. It includes compositional pollen counts of of \eqn{28} taxa
#' together with contemporary climate measurements.
#'
#' @format A \code{\link{list}} with two matrices:
#'
#' \describe{
#'
#' \item{\code{Y}}{A \eqn{7{,}832 \times 28} matrix of compositional pollen counts,
#' where each row corresponds to a sample and each column to one of the \eqn{28}
#' pollen taxa.}
#'
#' \item{\code{X}}{
#'
#' A \eqn{7{,}832 \times 8} matrix of site coordinates and climate variables
#' with the following columns:
#'
#' \itemize{
#' \item \code{longitude}: Longitude of the sampling location.
#' \item \code{latitude}: Latitude of the sampling location.
#' \item \code{altitude}: Altitude of the sampling location.
#' \item \code{gdd0}: Growing degree days above \eqn{0^\circ\mathrm{C}}$,
#' the annual sum of daily temperatures exceeding this threshold (\eqn{^\circ\mathrm{C}} days).
#' \item \code{gdd5}: Growing degree days above \eqn{5^\circ\mathrm{C}}$,
#' the annual sum of daily temperatures exceeding this threshold (\eqn{^\circ\mathrm{C}} days).
#' \item \code{mtco}: Mean temperature of the coldest month (\eqn{^\circ\mathrm{C}}).
#' \item \code{mtwa}: Mean temperature of the warmest month (\eqn{^\circ\mathrm{C}}).
#' \item \code{aet.pet}: Ratio of actual to potential evapotranspiration.
#' }
#' }
#' }
#'
#' @details
#' This data is typically referred to as a modern pollen-climate calibration data set, as it is
#' used in pollen-based palaeoclimate reconstruction methods.
#' This data set is an expanded version of the RS10 data set described by Allen et al. (2000).
#' The compositional pollen counts were obtained from the uppermost 5--10mm of lake sediment.
#' Climate variables were computed from weighted averaging of observations from
#' nearby weather stations over climatological periods of approximately 30 years.
#'
#'
#' @author
#' André F. B. Menezes
#'
#' @usage
#' data(pollen_climate, package = "zanicc")
#'
#' @references
#' Allen, J. R. M., Watts, W. A., & Huntley, B. (2000).
#' Weichselian palynostratigraphy, palaeovegetation and palaeoenvironment:
#' the record from Lago Grande di Monticchio, southern Italy.
#' \emph{Quaternary International}, \strong{73--74}, 91--110.
#'
#'
"pollen_climate"

#' @name pollen_monticchio
#'
#' @title Fossil pollen counts from Lago Grande di Monticchio
#'
#' @description
#' The fossil data set contains \eqn{924} samples of compositional counts of the same
#' 28 pollen taxa as in the `pollen_climate` modern data.
#' The fossil pollen records were extracted from one site at
#' Lago Grande di Monticchio, situated in the crater of Monte Vulture in Basilicata,
#' southern Italy.
#'
#' @format A \code{\link{matrix}} with the compositional counts of the 28 pollen
#' taxa. It also contains in the attributes the `depth` and the `age` estimates
#' corresponding to each fossil sample.
#'
#' @details
#' The pollen records has progressively been developed and published as more of
#' the sediment column has become available for analysis.
#' Relevant references which study this data set are Allen et al. (2000),
#' Allen and Huntley (2009), and Parnell et al. (2016) and references cited therein.
#'
#'
#' @author
#' André F. B. Menezes
#'
#' @usage
#' data(pollen_monticchio, package = "zanicc")
#'
#' @references
#' Allen, J. R. M., Watts, W. A., & Huntley, B. (2000).
#' Weichselian palynostratigraphy, palaeovegetation and palaeoenvironment:
#' the record from Lago Grande di Monticchio, southern Italy.
#' \emph{Quaternary International}, \strong{73--74}, 91--110.
#'
#' Allen, J.R., Huntley, B., 2009, jul. Last Interglacial palaeovegetation,
#' palaeoenvironments and chronology: a new record from Lago Grande di Monticchio, southern Italy.
#' \emph{Quaternary Science Reviews}, \strong{28 (15e16)}, 1521e1538.
#'
#' Parnell, A. C., Haslett, J., Sweeney, J., Doan, T. K., Allen, J. R. M. and Huntley, B. (2016),
#' Joint palaeoclimate reconstruction from pollen data via forward models and climate histories,
#' \emph{Quaternary Science Reviews} \strong{151}, 111--126.
#'
"pollen_monticchio"



#' @name microbiome_gut
#'
#' @title Human gut microbiome data set
#'
#' @description
#' The data is from the study of Wu et al. (2011) consisting of faecal samples
#' from \eqn{98} healthy volunteers, along with their demographic data and diet
#' information.
#' Microbial operational taxonomic units (OTUs) were taxonomically classified up to
#' the genus level and taxa with fewer than two samples were removed,
#' resulting in \eqn{80} genera.
#'
#'
#' @format A \code{\link{list}} with four matrices:
#'
#' \describe{
#'
#' \item{\code{Y}}{A \eqn{96 \times 80} matrix of OTU compositional counts,
#' where each row corresponds to a sample and each column to one of the \eqn{80}
#' OTU.}
#'
#' \item{\code{X_ffq}}{A \eqn{96 \times 108} matrix with covariates related to
#' habitual long-term diet information.}
#'
#' \item{\code{X_ffq}}{A \eqn{96 \times 108} matrix with covariates related to
#' recent diet information.}
#'
#' \item{\code{BMI}}{A \eqn{96 \times 1} matrix with measures of the body mass index
#' of each individual.}
#'
#' }
#'
#' @author
#' André F. B. Menezes
#'
#' @usage
#' data(microbiome_gut, package = "zanicc")
#'
#' @references
#' Wu, G., Chen, J., Hoffmann, C., Bittinger, K., Chen, Y.-Y., Keilbaugh, S., Bewtra, M., Knights, D., W.A., W., Knight, R., Sinha, R., Gilroy, E., Gupta, K., Baldassano, R., Nessel, L., Li, H., Bushman, F. and Lewis, J. (2011), ‘Linking long-term dietary patterns with gut microbial enterotypes’, \emph{Science} 334(6052), 105–-108.
#'
"microbiome_gut"






