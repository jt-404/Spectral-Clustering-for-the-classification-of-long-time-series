constructing_unnorm_Laplacian.m, constructing_RW_Laplacian.m, and constructing_SYM_Laplacian.m are driver functions for each respective type of Laplacian matrix.
Each requires the presence of Kmeans_March8_methodForGC_edit.m and distfuncentre.m to run clustering.


The dataset used can be found at https://ebrary.net/59044/education/details_public_databases and https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/. The code requires that all the data be in one txt file (The dataset will be in the file newset.txt), with each row being one timeseries with a class indicator (0,1,2,3,4) as the 4097th element.

Accuracy data for timeseries (clusters(a)=clusters(b)).xlsx contains the results (with each 10x10 grid labeled, showing the clusters per class on the vertical and the class pair on the horizontal).


MATLAB version: R2022b MATLAB 9.13
