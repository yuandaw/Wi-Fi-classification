%load your data file
features=data(:,1:100);%features
classlabel=data(:,101);%labels
n = randperm(size(features,1));%generating training&testing set randomly

%% training set--400 samples
train_features=features(n(1:400),:);
train_label=classlabel(n(1:400),:);
%% testing set--(total- 400)samples
test_features=features(n(401:end),:);
test_label=classlabel(n(401:end),:);

%% Data normalization
 [Train_features,PS] = mapminmax(train_features');
 Train_features = Train_features'; 
 Test_features = mapminmax('apply',test_features',PS); 
 Test_features = Test_features';
 
 %% Generating and training the SVM model
model = svmtrain(train_label,Train_features);

%% SVM simulation test
[predict_train_label] = svmpredict(train_label,Train_features,model);
[predict_test_label] = svmpredict(test_label,Test_features,model);

%% printing accuracy
compare_train = (train_label == predict_train_label);
accuracy_train = sum(compare_train)/size(train_label,1)*100; 
fprintf('training set accuracy：%f\n',accuracy_train)
compare_test = (test_label == predict_test_label);
accuracy_test = sum(compare_test)/size(test_label,1)*100;
fprintf('testing set accuracy：%f\n',accuracy_test)
