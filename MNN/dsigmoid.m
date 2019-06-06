function Y = dsigmoid( x )
%derivative of sigmoid function (tanh(x))= o(x)(1-o(x))
    
    Y=1-tanh(x)^2;

end

