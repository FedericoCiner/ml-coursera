function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X * theta);
loss = h - y;

% calculate regularized cost
regTerm = (lambda / 2) * theta .^ 2;
cost = -(1 / m) * (log(h') * y + log(1 - h') * (1 - y));
J = cost + sum(cat(1, eye(1), (regTerm(2:end) / m)) - eye(size(theta)));

% calculate regularized gradient
g = (1 / m) * (X' * loss);
grad = g + cat(1, eye(1), (lambda * (theta(2:end) / m))) - eye(size(theta));

end
