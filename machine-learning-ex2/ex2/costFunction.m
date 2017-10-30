function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
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
J = div * (mul + mul2);


diff = hx - y;
sum = diff' * X;
grad = div .* sum;




% =============================================================

end
