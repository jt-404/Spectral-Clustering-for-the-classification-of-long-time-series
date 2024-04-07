function [ DistMatrix ] = distfuncentre( M,C )
%Computes distances between clusters and centres and put them into an array
%   

Mdim=size(M);
Cdim=size(C);

DistMatrix=[];

for i=1:Mdim(1)
    for j=1:Cdim(1)
       DistMatrix(i,j)=sqrt(sum(abs(M(i,:)-C(j,:)).^2));
      % DistMatrix(i,j)=max(abs(M(i,:)-C(j,:)));
end;
end;

