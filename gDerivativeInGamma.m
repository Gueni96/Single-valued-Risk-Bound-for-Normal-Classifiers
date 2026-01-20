function derivativeValue = ...
    gDerivativeInGamma(gammaC,etaC,N,barMu,barSigma,varargin)
%functionValue Evaluation of the partial derivative of g in gamma
%   First parameter is the confidence level with respect to mean, the
%   second parameter is the confidence level with respect to the standard
%   deviation, the third parameter is the number of test samples, the
%   fourth parameter is the mean estimate and the fifth parameter is the
%   standard deviation estimate
    
    p = inputParser;
    addOptional(p,'SuppressWarning',false,@islogical);
    parse(p,varargin{:});
    supressWarning = p.Results.SuppressWarning;

    GammaFraction = (gamma((N-1)/2)/gamma(N/2));
    if isnan(GammaFraction)
        GammaFraction = sqrt(2/(N-1));
        if ~supressWarning
            disp('Warning: Gamma Fraction Approximated.');
        end
    end
    nu = sqrt(chi2inv(etaC,(N-1))./(N-1)) .* ( -(barMu/barSigma) + ...
        (tinv(1-gammaC,N-1)./sqrt(N)) );
    derivativeValue = ( (-1/sqrt(2)) .* exp((nu.^2)./(-2)) .* ...
        sqrt(chi2inv(etaC,(N-1))./N) .* ...
        GammaFraction .* ...
        ((1+(tinv(1-gammaC,N-1)./(N-1))).^(N/2)) ) + 1;
end