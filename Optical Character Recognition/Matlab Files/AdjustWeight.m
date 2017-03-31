% @Author : Parul Gupta and Veronica Sharma
%
% AdjustWeight function



function [W1 W2 error] = AdjustWeight(d, inputlayer, input, output, middle, W1, W2)

% Initializations
middlelayer = zeros(1,middle);
outputlayer = zeros(1,output);
error = 0;

% FORWARD PASS
[middlelayer outputlayer] = ForwardPass(input, output, middle, inputlayer, W1, W2);

% BACKWARD PASS
[W1 W2 error] = BackwardPass(d, input, inputlayer, output,outputlayer, middle, middlelayer, W1, W2);

end