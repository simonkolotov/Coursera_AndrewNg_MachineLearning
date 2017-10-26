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
  
  #Calculate the NN components (they would be needed in separate form for Back Propagation)
  a1 = a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2];
  z3 = a2 * Theta2';
  
  #Feedforward the NN
  h = sigmoid(z3);

  # Is there a faster way to do that?
  for k = 1:num_labels
    yk = (y == k);
    hthetak = h(:, k);
    J -= 1 / m * (log(hthetak)'*yk + log(1 - hthetak)'*(1 - yk) );
  end

  #Funnily enough, the following code ran slower than a direct sum... probably because we don't need all the combinations
  #binarize y from a {1,..,num_labels}^input_layer_size to {0,1}^(input_layer_size,num_labels)
  #Y = [];
  #for sampleId = 1:num_labels
    #Y = [Y; [y==sampleId]'];
  #end
  
  #J1 = trace(-1/m*(Y*log(h) + (1-Y)*log(1-h)));

  #regularization - this is the same or slower than a direct sum for smaller matrices, but gets more efficient for larger ones
  J += lambda / (2 * m) * (ones(1,size(Theta1,1))*(Theta1.^2)(:,2:end)*ones(size(Theta1,2)-1,1) + ones(1,size(Theta2,1))*(Theta2.^2)(:,2:end)*ones(size(Theta2,2)-1,1));

  #Back Propagation
  for t = 1:m
    for k = 1:num_labels
      yk = y(t) == k;
      delta_3(k) = h(t, k) - yk;
    end
    
    delta_2 = Theta2' * delta_3' .* sigmoidGradient([1, z2(t, :)])';
    delta_2 = delta_2(2:end);
  
    Theta1_grad = Theta1_grad + delta_2 * a1(t, :);
    Theta2_grad = Theta2_grad + delta_3' * a2(t, :);
  end

  Theta1_grad = Theta1_grad / m;
  Theta2_grad = Theta2_grad / m;
  % -------------------------------------------------------------
  
  Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
  Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

  
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
  
end
