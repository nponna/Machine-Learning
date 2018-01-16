function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_trials = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_trials = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
values = zeros(length(c_trials)*length(sigma_trials),2);
prediction_error_all = zeros(length(c_trials)*length(sigma_trials),1);
p = 1;
for c = 1:length(c_trials)
  for s = 1:length(sigma_trials)

    model= svmTrain(X, y, c_trials(c), @(x1, x2) gaussianKernel(x1, x2, sigma_trials(s)));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    prediction_error_all(p,:) =  prediction_error;
    values(p,:) = [c_trials(c) sigma_trials(s)];
    p = p +1;
  end
end

[M,I] = min(prediction_error_all);

right_values = values(I,:);
C = right_values(1);
sigma = right_values(2);







% =========================================================================

end
