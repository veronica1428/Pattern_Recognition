% @Author : Parul Gupta and Veronica Sharma
%
% BACK PROPAGATION ALGORITHM

clc;
clear all;
%load('TrainedData.mat');

%% INITIALIZATION

%Training data details
digits = 10;
samples = 7;
datasize = 16*16;
rows = digits*samples;cols = datasize;
trainingdata = zeros(rows,cols);

% Number of neurons
input = datasize;
middle = (datasize+digits)/2;
output = digits;

% Initialization of neurons
inputlayer = zeros(1,input);
middlelayer = zeros(1,middle);
outputlayer = zeros(1,output);
d = zeros(1,digits);                %actual output
W1 = zeros(middle,input,'double');  %input to middle layer weights
W2 = zeros(output,middle,'double'); %middle to output layer weights

% Initialization of weights
n = 0.2;
a = -1;b=1;
sumerror = 0;
W1 = (b-a).*rand(middle,input) + a;
W2 = (b-a).*rand(output,middle) + a;

%% TRAINING

% Creating training data
count = 0;
for i=0:digits-1
    for j=1:samples
        name = ['Image db/Training db/' int2str(i) '_' int2str(j) '.jpg'];
        image = imread(name);
        grayimage = rgb2gray(image);
        doubleimage = im2double(grayimage);
        count = count + 1;
        trainingdata(count,:) = reshape(doubleimage.',1,[]);
    end
end
        
% Train the network
arrCount = 1;
errorArr = zeros(1,100);
while(true)
    error = zeros(1,rows);
    for i=1:rows
        inputlayer = trainingdata(i,:);
        d = zeros(1,digits);
        d(ceil(i/samples))=1;   %desired output
        [W1 W2 error(i)] = AdjustWeight(d, inputlayer, input, output, middle, W1, W2);
    end
    sumerror = sum(error)/70;
    errorArr(arrCount) = sumerror;
    arrCount = arrCount + 1;
    sortErrorArr = sort(errorArr, 'descend');
    %sortErrorArr
    if sumerror < 0.05          %loop untill error is negligible
        break;
    end
end

%% TESTING

% Testing the test samples
test_samples = 3;
test_count = 0;
test_rows = digits*test_samples;
test_cols = datasize;
testdata = zeros(test_rows,test_cols);
matrix = zeros(test_rows,2);
% Creating the test data
for i=0:digits-1
    for j=1:test_samples
        test_name = ['Image db/Testing db/' int2str(i) '_' int2str(j) '.jpg'];
        test_image = imread(test_name);
        test_grayimage = rgb2gray(test_image);
        test_doubleimage = im2double(test_grayimage);
        test_count = test_count + 1;
        testdata(test_count,:) = reshape(test_doubleimage.',1,[]);
    end
end
% Calculating output layer
for k=1:test_rows
    test_inputlayer = testdata(k,:);
    [test_middlelayer test_outputlayer] = ForwardPass(input, output, middle, test_inputlayer, W1, W2);
    matrix(k,1) = ceil(k/test_samples)-1;         %desired output
    [ele matrix(k,2)] = max(test_outputlayer);    %actual output
    matrix(k,2) = matrix(k,2) - 1;
end

%Dictionary to contain accuracy of each digit
digitAccuracy = containers.Map ; 
accuracyMat = zeros(test_rows, 2);
%printmat(matrix, 'Prediction Matrix', 'Test0_1 Test0_2 Test0_3 Test1_1 Test1_2 Test1_3 Test2_1 Test2_2 Test2_3 Test3_1 Test3_2 Test3_3 Test4_1 Test4_2 Test4_3 Test5_1 Test5_2 Test5_3 Test6_1 Test6_2 Test6_3 Test7_1 Test7_2 Test7_3 Test8_1 Test8_2 Test8_3 Test9_1 Test9_2 Test9_3', 'Actual Predicted' )
matrix
count=0;
for i=1:test_rows
    
   desiredVal = matrix(i,1);
   predictedVal = matrix(i,2);
   
   if desiredVal ~= predictedVal
       count = count + 1;
   else
       if digitAccuracy.isKey(int2str(desiredVal))
          digitAccuracy(int2str(desiredVal)) = digitAccuracy(int2str(desiredVal)) + 1;
       else
          digitAccuracy(int2str(desiredVal)) = 1;
      end
   end
end

% display('Values in Dictionary are');
% [dictRow, dictCol] = size(digitAccuracy);
% 
% for i=0 : (dictRow-1)
%    display(i);
%    fprintf('value: %d', digitAccuracy(int2str(i)));
% end

%get Keys and Values of dictionary
keySet = cell2mat(keys(digitAccuracy));
accuracySet = cell2mat(values(digitAccuracy));

Accuracy_total = ( (test_rows - count)/test_rows )*100;
fprintf('Total Accuracy is  %2.2f %% \n', Accuracy_total);

