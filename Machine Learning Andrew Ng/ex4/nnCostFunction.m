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
XMat = [ones(m,1) X];
yMat = zeros(m,num_labels);

for i = 1:1:m
    yMat(i,y(i)) = 1;
end

z2 = Theta1 * XMat';
a2 = [ones(m,1) sigmoid(z2')];

z3 = Theta2 * a2';
hx = sigmoid(z3');

oneCalc = ones(m,1);

for i = 1:1:size(hx,2)
    sum1 = -yMat(:,i)' * log(hx(:,i));
    sum2 = -(oneCalc-yMat(:,i))'*log(oneCalc-hx(:,i));
    J = J + (1/m)*(sum1+sum2);
end

theta1Reg = Theta1(:,2:end);
theta2Reg = Theta2(:,2:end);

theta1Reg = theta1Reg.^2;
theta2Reg = theta2Reg.^2;

J = J + (lambda/(2*m))*(sum(theta1Reg,'all') + sum(theta2Reg,'all'));

Delta_1 = zeros(hidden_layer_size,input_layer_size+1);
Delta_2 = zeros(num_labels,hidden_layer_size+1);

% for i = 1:1:m
%     a1 = XMat(i,:)';
%     z2 = Theta1*a1;
%     a2 = [1;sigmoid(z2)];
%     z3 = Theta2*a2;
%     hx = sigmoid(z3);
%     
%     delta_3 = hx - yMat(i,:)';
%     delta_2 = Theta2'*delta_3.*[0;sigmoidGradient(z2)];
%     
%     Delta_1 = Delta_1+delta_2(2:end)*a1';
%     Delta_2 = Delta_2+delta_3*a2';
% end

for i = 1:1:m
    a1 = XMat(i,:)';
    z2backprop = z2(:,i);
    a2backprop = a2(i,:)';
    z3backprop = z3(:,i);
    hxbackprop = hx(i,:)';
    
    delta_3 = hxbackprop - yMat(i,:)';
    delta_2 = Theta2'*delta_3.*[0;sigmoidGradient(z2backprop)];
    
    Delta_1 = Delta_1+delta_2(2:end)*a1';
    Delta_2 = Delta_2+delta_3*a2backprop';
end


Theta1Reg = [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2Reg = [zeros(num_labels,1) Theta2(:,2:end)];
Theta1_grad = (1/m)*Delta_1+(lambda/m)*Theta1Reg;
Theta2_grad = (1/m)*Delta_2+(lambda/m)*Theta2Reg;
    
    















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end