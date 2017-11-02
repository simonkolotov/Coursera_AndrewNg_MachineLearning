function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
#Those are the correct values for the submission
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cValues = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaValues = [0.01 0.03 0.1 0.3 1 3 10 30];

cValuesNum = length(cValues);
sigmaValuesNum = length(sigmaValues);
minPredictionError = inf;

#commented out to speed up submission
%for i=1:cValuesNum
%    for j=1:sigmaValuesNum
%      
%        printf('iteration %d/%d\n', (i-1)*cValuesNum + j, cValuesNum*sigmaValuesNum);
%      
%        cCurVal = cValues(i);
%        sigmaCurVal = sigmaValues(j);
%        
%        model = svmTrain(X, y, cCurVal, @(x1, x2) gaussianKernel(x1, x2, sigmaCurVal)); 
%        predictions = svmPredict(model, Xval);
%        
%        predictions_error = mean(double(predictions ~= yval));
%        
%        printf ("Prediction Error: %f\n", predictions_error);
%        
%        if predictions_error < minPredictionError
%          minPredictionError = predictions_error;
%          C = cCurVal;
%          sigma = sigmaCurVal;
%          printf ("Prediction Error improved, C=%.03f sigma = %.03f\n", predictions_error, C, sigma);
%        end
%        
%        printf("\n\n");
%    end
%end
%printf("final values: C = %f, sigma = %f", C, sigma);





% =========================================================================

end
