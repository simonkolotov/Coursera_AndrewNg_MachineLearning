function [X_norm] = normalize(X, mu, sigma, doNormalizeCoords = false)
  %FEATURENORMALIZE Normalizes the features in X 
  %   NORMALIZE(X) returns a normalized version of X around the given mu and sigma
  %If doNormalizeCoords is true, an additional column of 1s is given

  X_norm = (X-mu)./sigma;
  
  if doNormalizeCoords,
    X_norm = [1 X_norm];
  end
    

end