% EDITED TO WORK WITH "constructing_unnorm_Laplacian.m", 
% "constructing_RW_Laplacian.m" and "constructing_SYM_Laplacian.m" (JWT)

%clear all;
%start clock
tic;

%reading the points from the dataset
%Signalsbig=dlmread('newset.txt');
Signalsbig=U;
Signals=Signalsbig(1:Udim(1),:);%working with just a portion of data
%nclass is the number of classes
nclass=1;% NS assign nclass=1, if there is just one class
%nclust is the number of clusters in each class
nclust(1)=Udim(2);% THIS IS WHERE YOU CHANGE THE NUMBER OF CLUSTERS K

%Percentage for training
PercentTraining=Udim(1);% NS assign 100 if you don't have training

costs=[];
Plan=[];
ClustCentres=[];
% % run for iclass=class (just 1 class if nclass=1)
for iclass=0:nclass-1
Signalsclass=Signals(iclass*100+1:iclass*100+PercentTraining,1:Udim(2));
%DistMatrixTotal=distfuncentre(Signalsclass,Signalsclass);
%k is the number of clusters in the current class
k=nclust(iclass+1);
% n is the number of points (signals) in the current class
n=size(Signalsclass,1);
%m is the number of recordings in each signal
m=size(Signalsclass,2);


%K-means initialisation
%%%%%%%%%%%%%%%%%%%%%%%%
% Take the and first nclust(iclass) points (that is number of clusters) of the dataset as the initial centres
% If you want the initial centres to be generated randomly, include it here

%Centres=Ytotal(1:nclust,:);
Centres=Signalsclass(1:nclust(1+iclass),:);

%Calculate the distance between the points and all centres, assign each
%point to the nearest centre and put the cluster number as an additional
%coordinate

D=distfuncentre(Signalsclass,Centres);
for i=1:n
    [d,I]=min(D(i,:));
    PointAllocation(i,1)=I;
end;
Check=false;
step=0;
while not(Check)
    step=step+1;
%Recalculate the centres
Centres1=zeros(nclust(1+iclass),m);

for i=1:k
    j=1;
    for kk=1:n
        if PointAllocation(kk,1)==i;
            Centres1(i,:)=Centres1(i,:)+Signalsclass(kk,:);
            j=j+1;
        end;
    end;
    Centres1(i,:)=(1/j)*Centres1(i,:);
end;

%Reallocate the points
D1=distfuncentre(Signalsclass,Centres1);
for i=1:n
    [d,I]=min(D1(i,:));
    PointAllocation1(i,1)=I;
end;
%Check if some points are moving from one cluster to another
Check=max(abs(PointAllocation-PointAllocation1))==0;

%Reallocate the centres and the corresponding points befor moving to the
%next step
PointAllocation=PointAllocation1;
Centres=Centres1;
end;
%Append all the centres

ClustCentres=[ClustCentres;Centres];

end;

%Find the distance from the points to the centres
DistMatrixCentres=distfuncentre(Signals(:,1:Udim(2)),ClustCentres);

%Assign the clusters to the classes
sumnclust(1)=0;
for kk=2:nclass+1
sumnclust(kk)=sumnclust(kk-1)+nclust(kk-1);
end;

%Find which centre is the nearest
Signals(:,(Udim(2)+2))=0;
for i=1:size(Signals,1)
    [d,nearestcentre]=min(DistMatrixCentres(i,:));
    Signals(i,(Udim(2)+1))=nearestcentre; 
end;

%stop clock
toc;
