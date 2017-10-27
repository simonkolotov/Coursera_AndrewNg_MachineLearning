function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly 
  errorsVector = (X*theta - y);
  J = 1/(2*m)*(errorsVector'*errorsVector) + lambda*(theta(2:end)'*theta(2:end))/(2*m);
  grad = (errorsVector'*X/m)' + [0; lambda/m*theta(2:end)];

  grad = grad(:);

end
