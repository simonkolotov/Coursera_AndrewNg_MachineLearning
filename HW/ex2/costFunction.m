function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = -(log(X*theta)'*y + log(1-X*theta)'*(1-y))/m;
grad = (((theta'*X') - y')*X/m)';

% ====================== YOUR COD
end
