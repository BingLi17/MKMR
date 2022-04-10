MKMR: A Multi-Kernel Machine Regression model to predict health outcomes using human microbiome data

----Description

MKMR was adapted from the "follow-path" package at https://www.di.ens.fr/~fbach/path/. Please visit https://www.di.ens.fr/~fbach/path/ for more details about the package.

In MKMR, we first calculate Unifrac distances and Bray-Curtis distance using R, then use the "follow-path" package in Matlab to learn model and make predictions.

In the package, "follow_entire_path.m" is the main function to run the regularization path algorithm, while others are supporting functions.

----Example usage for throat microbiome data analysis:

cal_dist.R: run before "main.m" to calculate the distance matrices and save them as ".mat" file

main.m: run after "cal_dist.R" to learn model and make predictions












