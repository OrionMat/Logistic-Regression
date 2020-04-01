clear;
close all;
clc;

data = load('student_admissions.txt');
X = data(:, 1:end-1); 
y = data(:, end);
[m, n] = size(X);

% add bias and initialise theta
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);

% minimize cost function
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y, m)), initial_theta, options);





% single prediction
x = [1 ; 45 ; 85];
prob = sigmoid(theta'*x)

% training accuracy
probTrain = sigmoid(X*theta);
predictionsTrain = round(probTrain);
fprintf('Train Accuracy: %f%%\n', mean(double(predictionsTrain == y)) * 100);





% plotting 

figure 1;
% figure 1 subplot 1: raw data
subplot(1,2,1); 
hold on
plotData(X(:, 2:end), y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted', 'location', 'southwest')

% figure 1 subplot 2: linear descision boundry
subplot(1,2,2);
hold on
plotData(X(:, 2:end), y);
descisionBoundry_2feature(theta, X, y);  
legend('Admitted', 'Not admitted', 'Decision Boundary', 'location', 'southwest')
xlabel('Exam 1 score')
ylabel('Exam 2 score')