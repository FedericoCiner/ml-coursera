function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta = theta;

for iter = 1:num_iters
    pred = X * theta; % get predictions
    loss = pred - y; % calculate loss
    gradient = (X' * loss) / m;
    theta = theta - (alpha * gradient);

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));

end

end
