function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1) X];
y = eye(num_labels)(y,:);

a1 = X;
z2 = a1*(Theta1');

a2 = sigmoid(z2);
m2 = size(a2, 1);
a2 = [ones(m2, 1) a2];

z3 = a2*(Theta2');
a3 = sigmoid(z3);

hx = a3;
hx_minus = 1 - hx;

log_hx = log(hx);
log_hx_minus = log(hx_minus);

minus_y = - y;
y_minus = - (1 - y);

div = 1/m;


mul = minus_y .* log_hx;
mul2 = y_minus .* log_hx_minus;

plus = (mul + mul2);

double_summation = sum(sum(plus,2));
J_before = div * double_summation;


div2 = lambda/(2*m);

theta1_square = sum(sum(Theta1(:,2:end).^2));
theta2_square = sum(sum(Theta2(:,2:end).^2));

theta_square = theta1_square + theta2_square;

J = J_before + (div2 * theta_square);

%{
size(a3)
size(y)
%}
delta_3 = (a3 - y);
%size(delta_3)
%Theta2_unbiased = Theta2(:,2:end);
%size(Theta2)
m2_z = size(z2, 1);
z2 = [ones(m2_z, 1) z2];

grad_z2 = sigmoidGradient(z2);

%size(grad_z2)

delta_2 = (delta_3 * Theta2) .* grad_z2;

delta_2 = delta_2(:,2:end);

%size(delta_2)

theta_delta_2 = 0;
theta_delta_1 = 0;

%size(a2)
%size(delta_3)

theta_delta_2 = theta_delta_2 + (delta_3' * a2);

%size(theta_delta_2) 

%size(a1')
%size(delta_2)


theta_delta_1 = theta_delta_1 + (delta_2' * a1);

%size(theta_delta_1) 

Theta1_grad = div * theta_delta_1;
Theta2_grad = div * theta_delta_2;

div3 = lambda/m;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (div3 .* Theta1(:,2:end)); 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (div3 .* Theta2(:,2:end)); 




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
