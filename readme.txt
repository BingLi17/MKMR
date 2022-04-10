MKMR: Multi-kernel machine regression

----Description

MKMR is adapted from the "follow-path" package at https://www.di.ens.fr/~fbach/path/. 
Please vivst https://www.di.ens.fr/~fbach/path/ for more details about the package.

In MKMR, we first calculate Unifrac distances and Bray-Curtis distance using R,
 then use the "follow-path" package in Matlab to learn model and make predictions.

In the package, "follow_entire_path.m" is the main function to run the regularization path algorithm,
and others are supporting functions.

----Example usage for throat microbiome data:

cal_dist.R: run before "main.m" to calculate the distance matrices and save them as ".mat" file

main.m: run after "cal_dist.R" to learn model and make predictions












