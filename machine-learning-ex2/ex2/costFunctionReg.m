function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
theta_square = 0;

for j = 2:size (theta)
  theta_square = theta_square + (theta(j))^2;
end

div2 = lambda/(2*m);

J = J_before + (div2 * theta_square);

diff = hx - y;
sum = diff' * X;
grad = div .* sum;

div3 = lambda/m;

for j = 2:size (theta)
  grad(j) = grad(j) + div3*theta(j);
end



% =============================================================

end
