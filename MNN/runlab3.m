close all;
clear all;

x1=[-1 -1;1 -1;-1 1;1 1];
y=[-1;1;1;-1];
z=zeros(4,1);
nk=0.1;
threshold=0.001;
flearn = zeros(800,1);
%initialize weights
w1=randperm(3)/2;
w2=randperm(3)/2;
wz=randperm(3)/2;

m = size(x1,1);

    
    wtotal(1,:)=w1(1,:);
    wtotal(2,:)=w2(1,:);
    wtotal(3,:)=wz(1,:);
   
   
    op=1;
    k=1;
%for loop here
while (op>threshold)

for j=1:1:m
    
x0=[1;x1(j,:)'];

% step 2
net1=w1*x0;
net2=w2*x0;

fnet1=sigmoid(net1);
fnet2=sigmoid(net2);

netz=wz(1,1)+wz(1,2)*fnet1+wz(1,3)*fnet2;
if(j==1 || j==4)
z(j,1)=-sigmoid(netz)
else
    z(j,1)=-sigmoid(netz)
end;
disp(j);
%step 3 BP compute errors
deltaz= (y(j,1)-z(j,1))*dsigmoid(netz)
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

    f=grad(y,z);% function in class J(w)=(1/2)*[summation from(1 to C)](tk-zk)^2= (1/2)*||t-z||^2
    %default matlab gradient function
     flearn(k:k+3,1) = flearn(k:k+3,1) + f;
    f1=gradient(f);
    op=(f1(1,1)^2+f1(2,1)^2+f1(3,1)^2+f1(4,1)^2)^0.5;
    k=k+1;
    disp(k)
    
end;


syms x y;
f=wtotal(1,:)*[1;x;y] ==0;
ysol=isolate(f,y);

ezplot(ysol);
hold on;



f=wtotal(2,:)*[1;x;y] ==0;
ysol=isolate(f,y);

ezplot(ysol);
hold on;
title('XOR Classifier');


scatter(x1(2:3,1),x1(2:3,2),'x')
hold on;
scatter(x1(1,1),x1(1,2),'o')
hold on;
scatter(x1(4,1),x1(4,2),'o')
hold on;

figure(2);
plot(flearn);
axis([1 200 0 inf]);
title('XOR Learning Rate')



 
