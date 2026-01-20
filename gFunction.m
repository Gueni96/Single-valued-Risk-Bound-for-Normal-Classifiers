function functionValue = gFunction(gammaC,etaC,N,barMu,barSigma)
%functionValue Evaluation of the function g
%   First parameter is the confidence level with respect to mean, the
%   second parameter is the confidence level with respect to the standard
%   deviation, the third parameter is the number of test samples, the
%   fourth parameter is the mean estimate and the fifth parameter is the
%   standard deviation estimate
arguments (Input)
    gammaC
    etaC
    N
    barMu
    barSigma
end

arguments (Output)
    functionValue
end
    nu = sqrt(chi2inv(etaC,(N-1))./(N-1)) .* ( -(barMu/barSigma) ...
        + (tinv(1-gammaC,N-1)./sqrt(N)) );
    functionValue = normcdf(nu) + gammaC + etaC;
end