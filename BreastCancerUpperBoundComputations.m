%% Load and Preprocess Data
% Read Data
Datatab = readtable('..\data\breast-cancer-wisconsin_txtConversion.txt');
X = Datatab{:,2:10};
Y = Datatab{:,11};
% Remove missing values
keep = true(1,length(Y));
for j=1:size(X,1)
    for k=1:size(X,2)
        if isnan(X(j,k))
            keep(j) = false;
        end
    end
end
X = X(keep,:);
Y = Y(keep);
% Split randomly into splitPoint=300 test samples and remaining as training
rng('default');
perm = randperm(length(Y));
X = X(perm,:);
Y = Y(perm,:);
splitPoint=300;
X_test = X(1:splitPoint,:);
X_train = X(splitPoint+1:end,:);
Y_test = Y(1:splitPoint);
Y_train = Y(splitPoint+1:end);
%% Train and evaluate SVM on test set
% Train Model
SVMModel = fitcsvm(X_train,Y_train,"KernelFunction","linear");
% Predict with Model
[label, score] = predict(SVMModel,X_test);
% Copmute Z variables
UndesiredPredictionIndex = Y_test~=label;
Z = ((Y_test==2)-(Y_test==4)) .* (score(:,1) - score(:,2));
Pmu = mean(Z);
Pstd = std(Z);
Pmu2 = mean(Z(Y_test==2));
Pmu4 = mean(Z(Y_test==4));
Ps2 = std(Z(Y_test==2));
Ps4 = std(Z(Y_test==4));
% Evaluate Model
overallErrorRate = 1-nnz(label==Y_test)/length(Y_test);
BenignErrorRate = 1-nnz(label(Y_test==2)==2)/nnz(Y_test==2);
MalignantErrorRate = 1-nnz(label(Y_test==4)==4)/nnz(Y_test==4);
disp(['Overall Error Rate: ',num2str(overallErrorRate)]);
disp(['Error Rate for Class 2/Benign: ', num2str(BenignErrorRate)]);
disp(['Error Rate for Class 4/Malignant: ', num2str(MalignantErrorRate)]);

%% Check Z Distribution
Z2 = Z(Y_test==2);
Z4 = Z(Y_test==4);
% histogram plot
figure('Name','Score Histograms');
subplot(3,1,1);
histogram(Z,16);
title('Score Histogram');
subplot(3,1,2);
histogram(Z2,16);
title('Class 2/Benign, Score Histogram');
%fontsize(gcf, 48, "points");
subplot(3,1,3);
histogram(Z4,16);
title('Class 4/Malignant, Score Histogram');
%fontsize(gcf, 48, "points");
hold off;
% Normality test
disp(['Indicators if Score is normal distributed at 5% Significance,' ...
    ' 0=yes, 1=no'])
disp(['     JB-Test Z: ', num2str(jbtest(Z)),...
    ', AD-Test Z: ', num2str(adtest(Z))]);
disp(['     JB-Test Class 2/Benign: ', num2str(jbtest(Z2)),...
    ', AD-Test Class 2/Benign: ', num2str(adtest(Z2))]);
disp(['     AD-Test Class 4/Malignant: ', num2str(jbtest(Z4)),...
    ', AD-Test Class 4/Malignant: ', num2str(adtest(Z4))]);
%% Compare Bounds
[~,~,singleValueBoundOverall] = computeBound(Pmu,Pstd,length(Z));
disp(['Bound for Overall Risk: ', num2str(singleValueBoundOverall),...
    ' vs. Estimator only: ', num2str(normcdf(-Pmu/Pstd)),...
    ' vs. measured: ', num2str(overallErrorRate)]);
[~,~,singleValueBoundBenign] = computeBound(Pmu2,Ps2,length(Z2));
disp(['Bound for Class 2/ Bengin: ', num2str(singleValueBoundBenign),...
    ' vs. Estimator only: ', num2str(normcdf(-Pmu2/Ps2)),...
    ' vs. measured: ', num2str(BenignErrorRate)]);
[gammaMi,etaMi,singleValueBoundMalignant] = ...
    computeBound(Pmu4,Ps4,length(Z4));
disp(['Bound for Class 4/ Malignant: ', ...
    num2str(singleValueBoundMalignant),...
    ' vs. Estimator only: ', num2str(normcdf(-Pmu4/Ps4)),...
    ' vs. measured: ', num2str(MalignantErrorRate)]);

%% Create Plotting Data for Malignant
nbrSamplePoints = 50;
line = linspace(min(gammaMi,etaMi)/4,3*max(gammaMi,etaMi),nbrSamplePoints);
Bound = zeros(nbrSamplePoints,nbrSamplePoints);
Points = zeros(nbrSamplePoints^2,3);
for gammaInd = 1:nbrSamplePoints
    for etaInd = 1:nbrSamplePoints
        gammaValue = line(gammaInd);
        etaValue = line(etaInd);
        Bound(gammaInd,etaInd) = ...
            gFunction(gammaValue,etaValue,length(Z4),Pmu4,Ps4);
        Points((gammaInd-1)*nbrSamplePoints + etaInd,:) = ...
            [gammaValue,etaValue,Bound(gammaInd,etaInd)]; 
    end
end
[X,Y] = meshgrid(line);
figure('Name','Malignant Bound Plot');
surf(X,Y,Bound);

%% Store in .csv
writematrix(Points, '..\data\boundFunctionOfMalignantSurfacePoints.csv');
writematrix(Z, '..\data\Zscores.csv');
writematrix(Z2, '..\data\ZBenignScores.csv');
writematrix(Z4, '..\data\ZMalignantScores.csv');
Measure = {'Failure Rate';...
    '$\Phi\left( -\frac{\bar{\mu}_{N}}{\bar{\sigma}_{N}}\right)$';...
    ['$g_{N}\left( \gamma_{\text{mi}}\text{\char044}\hspace{0.1667em}' ...
    '\eta_{\text{mi}}\right)$']};
Overall = [overallErrorRate;normcdf(-Pmu/Pstd);singleValueBoundOverall];
ClassBenign = [BenignErrorRate;normcdf(-Pmu2/Ps2);singleValueBoundBenign];
ClassMalignant = [MalignantErrorRate;normcdf(-Pmu4/Ps4);...
    singleValueBoundMalignant];
T = table(Measure, Overall, ClassBenign, ClassMalignant);
writetable(T,'..\data\ResultsBreastCancerPrediction.csv');