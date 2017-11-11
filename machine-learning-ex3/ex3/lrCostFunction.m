function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h = X * theta;
hx = sigmoid(h);
hx_minus = 1 - hx;
log_hx = log(hx);
log_hx_minus = log(hx_minus);
minus_y = - y;
y_minus = - (1 - y);
div = 1/m;

mul = minus_y' * log_hx;
mul2 = y_minus' * log_hx_minus;
J_before = div * (mul + mul2);

theta_square = sum(theta(2:end,:).^2);
div2 = lambda/(2*m);

J = J_before + (div2 * theta_square);


diff = hx - y;
sum = X' * diff;
grad = div .* sum;

div3 = lambda/m;

grad(2:end,:) = grad(2:end,:) + (div3 .* theta(2:end,:));






% =============================================================

grad = grad(:);

end
