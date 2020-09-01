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

hx = zeros(m,1);
loghx1 = zeros(m,1);
loghx2 = zeros(m,1);
oneintery = ones(m,1);
sum1 = 0;
sum2 = 0;
sum3 = 0;

for i = 1:1:m
    hx(i) = sigmoid(theta'*X(i,:)');
    loghx1(i) = log(hx(i));
    loghx2(i) = log(1-hx(i));
end

sum1 = -y'*loghx1;
sum2 = -(oneintery - y)'*loghx2;

for i = 2:1:size(theta)
    sum3 = sum3 + theta(i)^2;
end

J = (1/m)*(sum1+sum2)+(lambda/(2*m))*sum3;
grad(1) = (1/m)*(hx-y)'*X(:,1);


for i = 2:1:size(theta)
    grad(i) = (1/m)*(hx-y)'*X(:,i)+(lambda/m)*theta(i);
end

    






% =============================================================

end
