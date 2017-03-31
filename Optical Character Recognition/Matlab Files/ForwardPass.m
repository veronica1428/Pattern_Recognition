% @Author : Parul Gupta and Veronica Sharma
%
% ForwardPass function


function [middlelayer outputlayer] = ForwardPass(input, output, middle, inputlayer, W1, W2)

%Initialization
middlelayer = zeros(1,middle);
outputlayer = zeros(1,output);

% Compute middle layer
for j=1:middle
    sum = 0;
    for i=1:input
        sum = sum + W1(j,i)*inputlayer(i);
    end
    middlelayer(j) = 1 / (1 + exp(-1*sum));
end

% Compute output layer and error
for k=1:output
    sum = 0;
    for j=1:middle
        sum = sum + W2(k,j)*middlelayer(j);
    end
    outputlayer(k) = 1 / (1 + exp(-1*sum));
end

end