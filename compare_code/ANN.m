function [test_lab] = ANN(Fet_ANNtran,Lab_ANNtran,Fet_ANNtest)

%   Fet_ANNtran - input data.
%   Lab_ANNtran - target data.

x = Fet_ANNtran;
t = Lab_ANNtran;


trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

hiddenLayerSize = 28;
net = patternnet(hiddenLayerSize, trainFcn);

net.input.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'crossentropy';  % Cross-Entropy

net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false; 
% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

genFunction(net,'ANNNeuralNetworkFunction');
test_lab = ANNNeuralNetworkFunction(Fet_ANNtest);
end

