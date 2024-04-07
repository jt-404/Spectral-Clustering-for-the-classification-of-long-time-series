constructing_unnorm_Laplacian.m, constructing_RW_Laplacian.m, and constructing_SYM_Laplacian.m are driver functions for each respective type of Laplacian matrix.
Each requires the presence of Kmeans_March8_methodForGC_edit.m and distfuncentre.m to run clustering.


The dataset will be in the file newset.txt.
The dataset used can be found at https://ebrary.net/59044/education/details_public_databases and https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/. The contents of the 500 .txt files should be copied (in order) into a new .txt file newset.txt which will be used for clustering.


Accuracy data for timeseries (clusters(a)=clusters(b)).xlsx contains the results (with each 10x10 grid labeled, showing the clusters per class on the vertical and the class pair on the horizontal).

