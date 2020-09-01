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

hx = zeros(m);
for i = 1:1:m
    hx(i) = sigmoid(theta'*X(i,:)');
end

sum1 = 0;
sum2 = 0;
sum3 = zeros(size(theta));
for i = 1:1:m
    sum1 = sum1 - y(i)*log(hx(i));
    sum2 = sum2 - (1-y(i))*log(1-hx(i));
    sum3(1) = sum3(1) + (hx(i)-y(i))*X(i,1);
    sum3(2) = sum3(2) + (hx(i)-y(i))*X(i,2);
    sum3(3) = sum3(3) + (hx(i)-y(i))*X(i,3);
end

J = (1/m)*(sum1 + sum2);

for i = 1:1:size(theta)
    grad(i) = (1/m)*sum3(i);
end




% =============================================================

end
