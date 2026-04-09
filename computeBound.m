function [gammaMin,etaMin,bound] = computeBound(barMu,barSigma,N,varargin)
%computeBound minimizes the upper bound function g
%   The first parameter is the estimated mean, the second parameter is the
%   estimated standard deviation, the third parameter is the number of
%   test samples and the fourth input contains the hyperparameter of start
%   value for gamma, eta, the change threshold and stepsize

%% Set Hyperparameters
gammaStartDefault = 0.25;
etaStartDefault = 0.25;
changeThresholdDefault = 10^(-6);
rateDefault = 10^(-1);
stepSizeDefault = 10^(-4);
rateModifierDefault = 2;

checkConfidenceLevel = @(v) 0<v && v < 0.5;
checkStepSize = @(s) 0 < s && s < 0.25;
p = inputParser;
addOptional(p,'gammaStartValue',gammaStartDefault,checkConfidenceLevel);
addOptional(p,'etaStartValue',etaStartDefault,checkConfidenceLevel);
addOptional(p,'rate',rateDefault,@isnumeric);
addOptional(p,'rateModifier',rateModifierDefault,@isnumeric);
addOptional(p,'changeThreshold',changeThresholdDefault,@isnumeric);
addOptional(p,'stepSize',stepSizeDefault,checkStepSize);
parse(p,varargin{:})

gammaC = p.Results.gammaStartValue;
etaC = p.Results.etaStartValue;
changeThreshold = p.Results.changeThreshold;
s = p.Results.stepSize;
rate = p.Results.rate;
rateModifier = p.Results.rateModifier;
change = Inf;

%% Perform Minimization via Gradient Descent or line Search iteratively
while changeThreshold < change
    gOld = gFunction(gammaC,etaC,N,barMu,barSigma);
    gCurrent = gFunction(gammaC,etaC,N,barMu,barSigma);
    % Minimize in gamma
    gammaChange = Inf;
    while gammaChange > changeThreshold
        gammaNew = gammaC - rate*gDerivativeInGamma(gammaC,etaC,N,barMu,...
            barSigma,'SuppressWarning',true);
        if gammaNew>0 && gammaNew < 1
            gNew = gFunction(gammaNew,etaC,N,barMu,barSigma);
            if gNew <= gCurrent
                gammaC = gammaNew;
                gammaChange = gCurrent - gNew;
                gCurrent = gNew;
            else
                rate = rate/rateModifier;
            end
        else
            rate = rate/rateModifier;
        end
    end
    % Reset rate
    rate = p.Results.rate;
    % Minimize in eta
    etaChange = Inf;
    etaDerivative = gDerivativeInEta(gammaC,etaC,N,barMu,barSigma,'SuppressWarning',true);
    if isnan(etaDerivative)
        improving = true;
        while 0<etaC+s && etaC+s < 1 && improving
            etaNew = etaC + s;
            gNew = gFunction(gammaC,etaNew,N,barMu,barSigma);
            if (gNew<gCurrent)
                etaC = etaNew;
                gCurrent = gNew;
            else
                improving = false;
            end
        end
        improving = true;
        while 0<etaC-s && etaC-s < 1 && improving
            etaNew = etaC - s;
            gNew = gFunction(gammaC,etaNew,N,barMu,barSigma);
            if (gNew<gCurrent)
                etaC = etaNew;
                gCurrent = gNew;
            else
                improving = false;
            end
        end
        gCurrent = gFunction(gammaC,etaC,N,barMu,barSigma);
    else
        while etaChange > changeThreshold
            etaNew = etaC - ...
                rate*gDerivativeInEta(gammaC,etaC,N,barMu,barSigma);
            if etaNew > 0 && etaNew < 1
                gNew = gFunction(gammaC,etaNew,N,barMu,barSigma);
                if gNew <= gCurrent
                    etaC = etaNew;
                    etaChange = gCurrent - gNew;
                    gCurrent = gNew;
                else
                    rate = rate/rateModifier;
                end
            else
                rate = rate/rateModifier;
            end
        end
    end
    % Combined Update
    change = gOld - gCurrent;
end
%% Set output
if ((-barMu/barSigma)+(tinv(1-gammaC,N-1)/sqrt(N))) < 0
    gammaMin = gammaC;
    etaMin = etaC;
    bound = gFunction(gammaMin,etaMin,N,barMu,barSigma);
else
    gammaMin = NaN;
    etaMin = NaN;
    bound = 1;
end
end