%The attributes are (dontated by Riccardo Leardi, 
	%riclea@anchem.unige.it )
 	%1) Alcohol
 	%2) Malic acid
 	%3) Ash
	%4) Alcalinity of ash  
 	%5) Magnesium
	%6) Total phenols
 	%7) Flavanoids
 	%8) Nonflavanoid phenols
 	%9) Proanthocyanins
	%10)Color intensity
 	%11)Hue
 	%12)OD280/OD315 of diluted wines
 	%13)Proline  
    close all
    clear all
load wine.csv
 wineinfo(1:59,1)= (wine(1:59,2)-mean(wine(1:59,2)))/(max(wine(1:59,2))-min(wine(1:59,2)));%x1
 wineinfo(60:108,1)=(wine(130:178,2)-mean(wine(130:178,2)))/(max(wine(130:178,2))-min(wine(130:178,2)));%x1  (wine(130:178,2)-mean(wine(130:178,2)))/(max(wine(130:178,2))-min(wine(130:178,2)))
 wineinfo(1:59,2)= (wine(1:59,3)-mean(wine(1:59,3)))/(max(wine(1:59,3))-min(wine(1:59,3)));%x2
 wineinfo(60:108,2)=(wine(130:178,3)-mean(wine(130:178,3)))/(max(wine(130:178,3))-min(wine(130:178,3)));%x2
 wineinfo(1:59,3)= 1;%class1
 wineinfo(60:108,3)=-1;%class2

z=zeros(108,1);
nk=0.1;
threshold=0.001;
flearn = zeros(10000,1);
%initialize weights
w1=randperm(3)/2;
w2=randperm(3)/2;
wz=randperm(3)/2;


m = size(wineinfo,1);

    
    wtotal(1,:)=w1(1,:);
    wtotal(2,:)=w2(1,:);
    wtotal(3,:)=wz(1,:);
   
   
    op=1;
    k=1;
    
   
%for loop here
while (op>threshold)

for j=1:1:m
    
x0=[1;wineinfo(j,1:2)'];

% step 2
net1=w1*x0;
net2=w2*x0;

fnet1=sigmoid(net1);
fnet2=sigmoid(net2);

netz=wz(1,1)+wz(1,2)*fnet1+wz(1,3)*fnet2;
if(j>59)
z(j,1)=-sigmoid(netz);
else
    z(j,1)=sigmoid(netz);
end;

%step 3 BP compute errors
deltaz= (wineinfo(j,3)-z(j,1))*dsigmoid(netz);
delta1=dsigmoid(net1)*wz(1,2)*deltaz;
delta2=dsigmoid(net2)*wz(1,3)*deltaz;

%Update weight vectors

w1(1,1)=w1(1,1)-nk*delta1*x0(1,1);
w1(1,2)=w1(1,2)-nk*delta1*x0(2,1);
w1(1,3)=w1(1,3)-nk*delta1*x0(3,1); 

w2(1,1)=w2(1,1)-nk*delta2*x0(1,1);
w2(1,2)=w2(1,2)-nk*delta2*x0(2,1);
w2(1,3)=w2(1,3)-nk*delta2*x0(3,1);

wz(1,1)=wz(1,1)-nk*deltaz*x0(1,1);
wz(1,2)=wz(1,2)-nk*fnet1*deltaz;
wz(1,3)=wz(1,3)-nk*fnet2*deltaz;

end;
   
%update overal w matrix
    wtotal(1,:)=w1(1,:);
    wtotal(2,:)=w2(1,:);
    wtotal(3,:)=wz(1,:);
    
% test conditions of gradient norm

%% THIS PART NOT SURE

    f=gradw(wineinfo(:,3),z);% function in class J(w)=(1/2)*[summation from(1 to C)](tk-zk)^2= (1/2)*||t-z||^2
    %default matlab gradient function
    flearn(k:k+3,1) = flearn(k:k+3,1) + f(1:4,1);
    f1=gradient(f);
    op=(f1(1,1)^2+f1(2,1)^2+f1(3,1)^2+f1(4,1)^2)^0.5;
    k=k+1
    
    
end;

figure(1);

syms x y;
f=wtotal(1,:)*[1;x;y] ==0;
ysol=isolate(f,y);

ezplot(ysol);
hold on;



f=wtotal(2,:)*[1;x;y] ==0;
ysol=isolate(f,y);

ezplot(ysol);
hold on;



scatter(wineinfo(1:59,1),wineinfo(1:59,2),'o')
hold on;
scatter(wineinfo(60:108,1),wineinfo(60:108,2),'x')
title('Wine Classifier');
figure(2);
plot(flearn);
axis([1 200 0 inf]);
title('Learning Rate');



 