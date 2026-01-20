function derivativeValue = gDerivativeInEta(gammaC,etaC,N,barMu,barSigma,varargin)
%gDerivativeInEta Evaluation of the partial derivative of g in eta
%   First parameter is the confidence level with respect to mean, the
%   second parameter is the confidence level with respect to the standard
%   deviation, the third parameter is the number of test samples, the
%   fourth parameter is the mean estimate and the fifth parameter is the
%   standard deviation estimate
    p = inputParser;
    addOptional(p,'SuppressWarning',false,@islogical);
    parse(p,varargin{:});
    supressWarning = p.Results.SuppressWarning;

    CriticalFactor = ((2^((N-3)/2)*gamma((N-1)/2))./sqrt(N-1)) .* ...
        (chi2inv(etaC,N-1).^(1-(N/2)));
    if CriticalFactor ~= Inf && ~isnan(CriticalFactor)
        nu = sqrt(chi2inv(etaC,(N-1))./(N-1)) .* ( -(barMu/barSigma) + ...
            (tinv(1-gammaC,N-1)./sqrt(N)) );
        derivativeValue = ( (-1/sqrt(2*pi)) .* exp((nu.^2)./(-2)) .* ...
             CriticalFactor.* ...
            ((-barMu/barSigma) + (tinv(1-gammaC,N-1)./sqrt(N))) .* ...
            exp(chi2inv(etaC, N-1)./2) ) + 1;
    else
        derivativeValue = NaN;
        if ~supressWarning
            disp('Failure: Numerical Problems for selected Inputs.');
        end
    end
end