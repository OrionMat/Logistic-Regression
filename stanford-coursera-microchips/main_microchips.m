clear;
close all;
clc;

data = load('microchips_approval.txt');
X = data(:, 1:end-1); 
y = data(:, end);
[m, n] = size(X);
% add bias
X = [ones(m, 1) X];

% train linear logistic classifier
[theta, cost] = trainLogisticClassifier(X, y, m, n);

% train qudratic logistic classifier
degreeQuad = 2;
Xquad = polynomialExpand_2feature(X(:,2), X(:,3), degreeQuad);
nQuad = size(Xquad, 2)-1;
[thetaQuad, costQuad] = trainLogisticClassifier(Xquad, y, m, nQuad);

% train Nth order logistic classifier
degreeN = 10;
XN = polynomialExpand_2feature(X(:,2), X(:,3), degreeN);
nN = size(XN, 2)-1;
[thetaN, costN] = trainLogisticClassifier(XN, y, m, nN);





% regularized logistic regression

lambda0 = 1
[thetaReg0, costReg0] = trainLogisticClassifier_Reg(XN, y, m, nN, lambda0);

lambda1 = 10
[thetaReg1, costReg1] = trainLogisticClassifier_Reg(XN, y, m, nN, lambda1);

lambda2 = 100
[thetaReg2, costReg2] = trainLogisticClassifier_Reg(XN, y, m, nN, lambda2);





% plotting 

figure 1;
% figure 1 subplot 1: raw data
subplot(2,2,1); 
hold on
plotData(X(:, 2:end), y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
%legend('Admitted', 'Not admitted', 'location', 'southwest')

% figure 1 subplot 2: linear descision boundry
subplot(2,2,2);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_2feature(theta, X, y);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% figure 1 subplot 3: quadratic descision boundry
subplot(2,2,3);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaQuad, degreeQuad);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% figure 1 subplot 4: Nth descision boundry
subplot(2,2,4);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaN, degreeN);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')





figure 2;
% figure 2 subplot 1: 10th order descision boundry
subplot(2,2,1);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaN, degreeN);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% figure 2 subplot 2: 10th order descision boundry - regularized lambda=1
subplot(2,2,2);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaReg0, degreeN);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% figure 2 subplot 3: 10th order descision boundry - regularized lambda=10
subplot(2,2,3);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaReg1, degreeN);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% figure 2 subplot 4: 10th order descision boundry - regularized lambda=100
subplot(2,2,4);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_Nfeature(thetaReg2, degreeN);  
%legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')