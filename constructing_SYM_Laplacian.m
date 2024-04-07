% Constructs the normalized symetric Laplacian matrix and takes the "K_1"
% smallest eigenvectors and gives the matrix U to Kmeans script and gets 
% back clusters
% Calculates the centres of the clusters and then classifies test data
% based on nearest centre and measures accuracy
% edit lines 22-29 "true" to select similarity function and centres
% dimensions
% (JT)

clear all;

%INPUT HERE
Data = dlmread('newset.txt'); %original dataset (500x4098)

for a_b = 1:10
for K_1 = 1:10 % number of clusters
n_class = 5; % number of classes
Percent_Training = 75;
Centres_ts = [];

% Similarity Functions
SF1_EuclideanSqrdSim = false; % SF1: SQUARED EUCLIDEAN DISTANCE SIMILARITY FUNCTION WITH SIM THRESHOLD
SF2_AveDiffSim = true; % SF2: Average of maximum sampled differences similarity function
SF3_maxminsim = false; % SF3: Average sampled max and min similarity function
SF4_diffmaxminsim = false; % SF4: Difference max, Difference Min similarity function
% Centre calculations
C1_4097 = true; % dimension = 4097
C2_1_AveDiffCentres = false; % dimension = 1
C3_2_maxmincentres = false; % dimension = 2

Cluster_pairs = [1 2;
                 1 3;
                 1 4;
                 1 5;
                 2 3;
                 2 4;
                 2 5;
                 3 4;
                 3 5;
                 4 5];
% Training
for i_class = [Cluster_pairs(a_b,1) Cluster_pairs(a_b,2)]
    Training_data = [Data((i_class-1)*100+1:(i_class-1)*100+Percent_Training,1:4097)]; % remove class index in n=4098

    % Sim-threshold is highest distance between points in training data
    if (SF1_EuclideanSqrdSim == true)
        sim_threshold_E = 0;
        for i = 1:size(Training_data,1)
            for j = 1:size(Training_data,1)
                d_ij = 0; % dissimilarity
                for n = 1:size(Training_data,2)
                    d_ij = (Training_data(i,n)-Training_data(j,n))^2;
                    Dis_Vector_1(n) = d_ij;
                end
                D_1 = sum(Dis_Vector_1);
                if  D_1 > sim_threshold_E
                    sim_threshold_E = D_1;
                end
            end
        end
    end

    % constructing similarity (weight) matrix
    SimMatrix = []; % Similarity matrix
    d_ij = 0; % difference between n-th entries of vectors


    % SF1: SQUARED EUCLIDEAN DISTANCE SIMILARITY FUNCTION WITH SIM THRESHOLD
    if (SF1_EuclideanSqrdSim == true)
        for i = 1:size(Training_data,1)
            for j = 1:size(Training_data,1)
                d_ij = 0; % clear from last loop
                for n = 1:size(Training_data,2)
                    d_ij = (Training_data(i,n)-Training_data(j,n))^2;
                    Dis_Vector(n) = d_ij;
                end
                Dis = sum(Dis_Vector); % squared Euclidean distance
                if Dis > sim_threshold_E
                    SimMatrix(i,j) = 0;
                end
                if Dis <= sim_threshold_E
                    SimMatrix(i,j) = sim_threshold_E - Dis;
                end
            end
        end
    end

    % SF2: Average of maximum sampled differences similarity function
    % SF3: Average sampled max and min similarity function
    if (SF2_AveDiffSim == true)||(C2_1_AveDiffCentres == true)||(SF3_maxminsim == true)||(C3_2_maxmincentres == true)
        maximum_matrix = [];
        minimum_matrix = [];
        for i = 1:size(Training_data,1)
            for a = 0:39
                maxi = Training_data(i,a*100+1);
                mini = Training_data(i,a*100+1);
                for j = (a*100+2):(a*100+100)
                    if Training_data(i,j) > maxi
                        maxi = Training_data(i,j);
                    end
                    if Training_data(i,j) < mini
                       mini = Training_data(i,j);
                    end
                end
                maximum_matrix(i,a+1) = maxi;
                minimum_matrix(i,a+1) = mini;
            end
            maximum_vect(i,1) = sum(maximum_matrix(i,:))/40;
            minimum_vect(i,1) = sum(minimum_matrix(i,:))/40;
        end

        % SF2: Average of maximum sampled differences similarity function
        if (SF2_AveDiffSim == true)||(C2_1_AveDiffCentres == true)
            Ave_Diff = maximum_vect - minimum_vect;
            sim_threshold = max(Ave_Diff) - min(Ave_Diff);    
            for i = 1:size(Ave_Diff,1)
                for j = 1:size(Ave_Diff,1)
                    Dis = abs(Ave_Diff(i,:) - Ave_Diff(j,:));
                    if Dis <= sim_threshold
                        SimMatrix(i,j) = sim_threshold - Dis;
                    end
                end
            end
        end

        % SF3: Average sampled max and min similarity function
        if (SF3_maxminsim == true)||(C3_2_maxmincentres == true)
            for i = 1:size(maximum_vect,1)
                MaxMinMatrix(i,:) = [maximum_vect(i,:) minimum_vect(i,:)];
            end
            MaxMinMatrix;
            DisMatrix = dist(MaxMinMatrix');
            sim_threshold = max(max(DisMatrix));
            for i = 1:size(DisMatrix,1)
                for j = 1:size(DisMatrix,2)
                    SimMatrix(i,j) = sim_threshold - DisMatrix(i,j);
                end
            end
        end
    end

    % SF4: Difference max, Difference Min similarity function
    if (SF4_diffmaxminsim == true)
        for i = 1:size(Training_data,1)
            for a = 0:39
                maxi = Training_data(i,a*100+1);
                mini = Training_data(i,a*100+1);
                for j = (a*100+2):(a*100+100)
                    if Training_data(i,j) > maxi
                        maxi = Training_data(i,j);
                    end
                    if Training_data(i,j) < mini
                       mini = Training_data(i,j);
                    end
                end
                maximum_matrix(i,a+1) = maxi;
                minimum_matrix(i,a+1) = mini;
            end
            maximum_vect(i,1) = sum(maximum_matrix(i,:))/40;
            minimum_vect(i,1) = sum(minimum_matrix(i,:))/40;
        end     
        for i = 1:size(Training_data,1)
            for j = 1:size(Training_data,1)
                DisMatrix(i,j) = abs(maximum_vect(i,:)-maximum_vect(j,:)) + abs(minimum_vect(i,:)-minimum_vect(j,:));
            end
        end
        sim_threshold = max(max(DisMatrix));
        for i = 1:size(Training_data,1)
            for j = 1:size(Training_data,1)
                SimMatrix(i,j) = sim_threshold - DisMatrix(i,j);
            end
        end
    end

    W = SimMatrix; % weight matrix

    % constructing diagonal matrix
    D = zeros(size(Training_data,1));
    for i = 1:size(Training_data,1)
        D(i,i) = sum(W(i,:));
    end
    D; % diagonal matrix

    L = D - W; % Unnormalized Laplacian


    % MODIFICATIONS TO UNNORM CODE BEGIN HERE (JWT)

    Dinv = inv(D); % Inverse of diagonal matrix
    Dinvsqrt = sqrtm(Dinv); % Inverse sqrt of diagonal matrix "D^(-1/2)"
    Lsym = Dinvsqrt*L*Dinvsqrt; % Normalised symetric Laplacian

    [Eigen_vectors, Eigen_values] = eigs(Lsym,K_1,'smallestabs');
    T1 = Eigen_vectors; % T1 is the matrix BEFORE row normalization

    U = [];

    % Row normalisation
    for i = 1:size(T1,1)
        T_row = [];
        for k = 1:size(T1,2)
            t_ik = T1(i,k)^2;
            T_row(k) = t_ik;
        end
        RowSum = sum(T_row);
        for j = 1:size(T1,2)
            U(i,j) = T1(i,j)/(RowSum)^(1/2);
        end
    end

    Udim = size(U);

    % MODIFICATIONS END HERE (JWT)


    % Use Kmeans on rows of U
    Kmeans_March8_methodForGC_edit;
    % clusters from Kmeans_March8_methodForGC_edit
    ClustCentres;
    % Only need the one additional column
    Signals(:,1:(size(U,2)+1));


    % Finding centres of clusters

    % C1: d=4097, No dimension reduction
    if C1_4097 == true
        for a = 1:K_1
            Clust_a_neg = [];
            Clust_a_pos = [];
            Centre_a_pos = zeros(1,4097);
            Centre_a_neg = zeros(1,4097);
            for i = 1:size(Signals,1)
                if Signals(i,size(U,2)+1) == a
                    if Training_data(i,1) < 0
                        Clust_a_neg = [Clust_a_neg; Training_data(i,:)];
                    else
                        Clust_a_pos = [Clust_a_pos; Training_data(i,:)];
                    end
                end
            end
            for j = 1:size(Clust_a_pos,2)
                Centre_a_pos(1,j) = mean(Clust_a_pos(:,j));
            end
            for j = 1:size(Clust_a_neg,2)
                Centre_a_neg(1,j) = mean(Clust_a_neg(:,j));
            end
            if Centre_a_pos(1,1) ~= 0
                Centres_ts = [Centres_ts; Centre_a_pos i_class-1];
            end
            if Centre_a_neg(1,1) ~= 0
                Centres_ts = [Centres_ts; Centre_a_neg i_class-1];
            end
        end
    end

    % C2: d=1, Average of maximum sampled differences centre
    % C3: d=2, Average sampled max and min centre
    if (C2_1_AveDiffCentres == true)||(C3_2_maxmincentres == true)
        for a = 1:K_1
            Clust_a = [];
            for i = 1:size(Signals,1)
                if Signals(i,size(U,2)+1) == a
                    if C2_1_AveDiffCentres == true
                        Clust_a = [Clust_a; Ave_Diff(i,:)]; % C2: d=1
                    end
                    if C3_2_maxmincentres == true
                        Clust_a = [Clust_a; MaxMinMatrix(i,:)]; % C3: d=2
                    end
                end
            end
            for j = 1:size(Clust_a,2)
                Centre_a(1,j) = mean(Clust_a(:,j));
            end
            Centres_ts = [Centres_ts; Centre_a i_class-1];
        end
    end

    i_class;
end


% Classification code
Centres_ts;
Test_data = [];
for i_class = [Cluster_pairs(a_b,1) Cluster_pairs(a_b,2)]
    Test_data = [Test_data; Data((i_class-1)*100+Percent_Training+1:(i_class)*100,:)];
end

% Test_data_dimension = [q-dimensional-points Test_data(:,4098)]
if C1_4097 == true % No dimension reduction, d=4097
    Test_data_dimension = Test_data;
end

if (C2_1_AveDiffCentres == true)||(C3_2_maxmincentres == true) % dimension = 1 or 2
    maximum_matrix = [];
    minimum_matrix = [];
    Ave_Diff = [];
    maximum_vect = [];
    minimum_vect = [];
    for i = 1:size(Test_data,1)
        for a = 0:39
            maxi = Test_data(i,a*100+1);
            mini = Test_data(i,a*100+1);
            for j = (a*100+2):(a*100+100)
                if Test_data(i,j) > maxi
                    maxi = Test_data(i,j);
                end
                if Test_data(i,j) < mini
                   mini = Test_data(i,j);
                end
            end
            maximum_matrix(i,a+1) = maxi;
            minimum_matrix(i,a+1) = mini;
        end
        maximum_vect(i,1) = sum(maximum_matrix(i,:))/40;
        minimum_vect(i,1) = sum(minimum_matrix(i,:))/40;
    end
    if C2_1_AveDiffCentres == true % dimension = 1
        Ave_Diff = maximum_vect - minimum_vect;
        Test_data_dimension = [Ave_Diff Test_data(:,4098)];
    end
    if C3_2_maxmincentres == true % dimension = 2
        MaxMinMatrix = [];
        for i = 1:size(maximum_vect,1)
            MaxMinMatrix(i,:) = [maximum_vect(i,:) minimum_vect(i,:)];
        end
        Test_data_dimension = [MaxMinMatrix Test_data(:,4098)];
    end
end
    
Test_dim = size(Test_data_dimension);

% cluster assignment
Cluster_dist = distfuncentre(Test_data_dimension(:,1:Test_dim(2)-1),Centres_ts(:,1:Test_dim(2)-1));
for i = 1:Test_dim(1)
    [M,Cluster_Index] = min(Cluster_dist(i,:));
    Test_data_dimension(i,Test_dim(2)+1) = Cluster_Index;
end

% accuracy check
for i = 1:size(Test_data,1)
    if Test_data_dimension(i,Test_dim(2)) == Centres_ts(Test_data_dimension(i,(Test_dim(2)+1)),Test_dim(2))
        Test_data_dimension(i,Test_dim(2)+2) = 0;    
    else
        Test_data_dimension(i,Test_dim(2)+2) = 1;
    end
end

Test_data_dimension(:,Test_dim(2):Test_dim(2)+2);
correct = size(Test_data,1) - sum(Test_data_dimension(:,Test_dim(2)+2));
Accuracy_ts = (correct*100)/size(Test_data,1);
K_1;
Accuracy_vect(K_1,:) = Accuracy_ts;
end
Accuracy_matrix(:,a_b) = Accuracy_vect;
end
