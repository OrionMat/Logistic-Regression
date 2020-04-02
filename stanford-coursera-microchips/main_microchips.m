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