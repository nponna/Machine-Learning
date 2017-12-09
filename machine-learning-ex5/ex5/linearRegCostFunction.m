function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
summation = sum((h - y).^2);
half = 1/(2*m);
J = half*summation;


div2 = lambda/(2*m);

theta_square = sum(theta(2:end,:).^2);

J = J + (div2 * theta_square);

hx = X * theta;
diff = h - y;
sum = diff' * X;
mul_constants = (1/m);
grad = mul_constants * sum';

div3 = lambda/m;

grad(2:end,:) = grad(2:end,:) + (div3 .* theta(2:end,:)); 











% =========================================================================

grad = grad(:);

end
