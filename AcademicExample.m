%% Set Parameter
muOne = [1;0];
sigmaOne = [0.5,0.25;0.25,0.5];
muTwo = [2;2];
sigmaTwo = [0.35,0.1;0.1,0.25];
N_train = 1000;
Ns_test = [150,200,250];

%% Simulate and train model
rng('default');
X_train = [mvnrnd(muOne,sigmaOne,fix(N_train/2));...
    mvnrnd(muTwo,sigmaTwo,fix(N_train/2))];
Y_train = [ones(fix(N_train/2),1);2*ones(fix(N_train/2),1)];
rp = randperm(N_train);
X_train = X_train(rp,:);
Y_train = Y_train(rp);
% Train Model
SVMModel = fitcsvm(X_train,Y_train,"KernelFunction","linear");
%% Simulate and evaluate Test data
PrintTab = zeros(length(Ns_test),7);
for Nind = 1:length(Ns_test)
    % Generate test data
    N_test = Ns_test(Nind);
    X_test = [mvnrnd(muOne,sigmaOne,fix(N_test/2));...
        mvnrnd(muTwo,sigmaTwo,fix(N_test/2))];
    Y_test = [ones(fix(N_test/2),1);2*ones(fix(N_test/2),1)];
    rp = randperm(N_test);
    X_test = X_test(rp,:);
    Y_test = Y_test(rp);
    % Predict with Model
    [label, score] = predict(SVMModel,X_test);
    Z = ((Y_test==1)-(Y_test==2)) .* (score(:,1) - score(:,2));
    % Sanity Checks
    figure('Name',['Score Histograms N=', num2str(N_test)]);
    subplot(2,1,1);
    histogram(Z(Y_test==1),16);
    title(['Score Histogram Class 1 for N=', num2str(N_test)]);
    subplot(2,1,2);
    histogram(Z(Y_test==2),16);
    title(['Score Histogram Class 2 for N=', num2str(N_test)]);
    % Normality test
    disp(['Indicators, for N=',num2str(N_test),...
        ' if Score is normal distributed at 5% Significance, 0=yes, 1=no'])
    disp(['     JB-Test Class 1: ', num2str(jbtest(Z(Y_test==1))),...
        ', JB-Test Class 1: ', num2str(jbtest(Z(Y_test==1))),...
        ' for N=', num2str(N_test)]);
    disp(['     JB-Test Class 2: ', num2str(jbtest(Z(Y_test==2))),...
        ', JB-Test Class 2: ', num2str(jbtest(Z(Y_test==2))),...
        ' for N=', num2str(N_test)]);
    % Scatter Plot
    figure('Name',['Scatter Plot N=', num2str(N_test)]);
    scatter(X_test(Y_test==1,1),X_test(Y_test==1,2),'blue');
    hold on;
    scatter(X_test(Y_test==2,1),X_test(Y_test==2,2),'red');
    hold off;
    T = array2table([X_test,Y_test]);
    T.Properties.VariableNames(1:3) = {'x','y','label'};
    writetable(T,['..\data\academicExampleTestDataN=',...
        num2str(N_test),'.csv'])
    %% Compare Bounds
    [gammaMiOne,etaMiOne,boundOne] = ...
        computeBound(mean(Z(Y_test==1)),...
        std(Z(Y_test==1)),nnz(Y_test==1));
    estimatorOnlyOne = ...
        normcdf(-mean(Z(Y_test==1))/std(Z(Y_test==1)));
    measuredErrorOne = nnz(label(Y_test==1)==2)/nnz(Y_test==1);
    disp(['Bound for Class 1: ', num2str(boundOne),...
        ' vs. Estimator only: ', num2str(estimatorOnlyOne),...
        ' vs. measured: ', num2str(measuredErrorOne),...
        ' for N=', num2str(N_test)]);
    disp(['Minimum for Class 1 attained at gammaMiOne=',...
        num2str(gammaMiOne),', etaMiOne=', num2str(etaMiOne),'.']);
    [gammaMiTwo,etaMiTwo,boundTwo] = ...
        computeBound(mean(Z(Y_test==2)),...
        std(Z(Y_test==2)),nnz(Y_test==2));
    estimatorOnlyTwo = ...
        normcdf(-mean(Z(Y_test==2))/std(Z(Y_test==2)));
    measuredErrorTwo = nnz(label(Y_test==2)==1)/nnz(Y_test==2);
    disp(['Bound for Class 2: ', num2str(boundTwo),...
        ' vs. Estimator only: ', num2str(estimatorOnlyTwo),...
        ' vs. measured: ', num2str(measuredErrorTwo),...
        ' for N=', num2str(N_test)]);
    disp(['Minimum for Class 2 attained at gammaMiTwo=', ...
        num2str(gammaMiTwo),', etaMiTwo=', num2str(etaMiTwo),'.']);
    % Write to variable
    PrintTab(Nind,:) = [N_test,measuredErrorOne,estimatorOnlyOne,...
        boundOne,measuredErrorTwo,estimatorOnlyTwo,boundTwo];
end
%% Print Bound to file
writematrix(PrintTab,'..\data\academicExampleResults.csv');
