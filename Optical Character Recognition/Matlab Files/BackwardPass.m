% @Author : Parul Gupta and Veronica Sharma
%
% BackwardPass function



function [W1 W2 error] = BackwardPass(d, input, inputlayer, output,outputlayer, middle, middlelayer, W1, W2)

%Initialization
deltak = zeros(1,output);
deltaj = zeros(1,middle);
error = 0;
n=0.2;

% Re-calculate weights from middle to output layer
for k=1:output
    deltak(k) = (d(k)-outputlayer(k))*outputlayer(k)*(1-outputlayer(k));
    for j=1:middle
        W2(k,j) = W2(k,j) + n*deltak(k)*middlelayer(j);
    end
end

% Re-calculate weights from input to middle layer
sum = 0;
for j=1:middle
    for k=1:output
        sum = sum + deltak(k)*W2(k,j);
    end
    deltaj(j) = (middlelayer(j)*(1-middlelayer(j))*sum);
    for i=1:input
        W1(j,i) = W1(j,i) + n*deltaj(j)*inputlayer(i);
    end
end

% Calculate error
for k=1:output
    error = error + (d(k)-outputlayer(k))*(d(k)-outputlayer(k));
end

end