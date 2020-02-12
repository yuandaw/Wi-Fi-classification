class = 7 %number of classes
sample = 200 %sample number for each class
proportion = 0.7 %proportion of training set for each class
features=mobile(:,1:200);%features
classlabel=mobile(:,201);%labels
m(1,1:sample) = randperm(sample);%generating training&testing set randomly

n=[]

for i = 1:class
    n(1,sample*proportion*(i-1)+1:sample*proportion*i) = m(1,1:sample*proportion)+sample*(i-1);
end

l=[]

for i = 1:class
    l(1,sample*(1-proportion)*(i-1)+1:sample*(1-proportion)*i) = m(1,sample*proportion+1:sample)+sample*(i-1);
end

%% training set
train_features=features(n(1:end),:);
train_label=classlabel(n(1:end),:);
%% testing set
test_features=features(l(1:end),:);
test_label=classlabel(l(1:end),:);

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

