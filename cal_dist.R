rm(list = ls())

library(glmnet)
library(GUniFrac)
library(MiRKAT)
library(MiSPU)
library(randomForest)
library(ROCR)
library(caret)

# load throat data
data(throat.otu.tab)
data(throat.tree)
data(throat.meta)

# covariates
anti =  (throat.meta$AntibioticUsePast3Months_TimeFromAntibioticUsage != "None")*2 - 1
Male = (throat.meta$Sex == "Male")*2 - 1
cova = cbind(Male, anti)

# outcome
Smoker = (throat.meta$SmokingStatus == "Smoker") *2 - 1

## Create the UniFrac Distances
otu.tab.rff = Rarefy(throat.otu.tab)$otu.tab.rff
otu.tab.rff = otu.tab.rff/rowSums(otu.tab.rff)
unifracs = GUniFrac::GUniFrac(otu.tab.rff, throat.tree, alpha=c(0, 0.25, 0.5, 0.75, 1))$unifracs
weighted = unifracs[,,"d_1"]
unweighted = unifracs[,,"d_UW"]
w0 = unifracs[,,"d_0"]
w25 = unifracs[,,"d_0.25"]
w5 = unifracs[,,"d_0.5"]
w75 = unifracs[,,"d_0.75"]
bc= as.matrix(vegdist(otu.tab.rff, method="bray"))

setwd("/Volumes/GoogleDrive/My Drive/micob/mkmr")
library(R.matlab)
writeMat('input_from_r.mat', D.weighted = weighted, D.unweighted = unweighted, D.BC = bc, 
         D.0 = w0, D.0.25 = w25, D.0.75 = w75, D.0.5 = w5, y = Smoker, cova = cova,
         OTU = otu.tab.rff)
