constructing_unnorm_Laplacian.m, constructing_RW_Laplacian.m, and constructing_SYM_Laplacian.m are driver functions for each respective type of Laplacian matrix.
Each requires the presence of Kmeans_March8_methodForGC_edit.m and distfuncentre.m to run clustering.
The dataset is contained in newset.txt and is in the format required by the driver functions.
Accuracy data for timeseries (clusters(a)=clusters(b)).xlsx contains the results (with each 10x10 grid labeled, showing the clusters per class on the vertical and the class pair on the horizontal).