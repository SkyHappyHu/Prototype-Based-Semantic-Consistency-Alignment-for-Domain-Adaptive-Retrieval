function dataset = construct_dataset(dataname, test_num, param)
addpath('./data/');
switch dataname
    case 'MNIST-USPS'        
        load ./data/MNIST_vs_USPS X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'USPS-MNIST'        
        load USPS_vs_MNIST X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
        
        
    case 'C1-C2'        
        load ./data/COIL_1 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
  
end

c = length(unique(Y_tar));  % The number of classes;
dataset.c = c;



%% Select the test set
randIdx = randperm(length(Y_tar));
if test_num < 1
    sele_num = round(test_num * size(X_tar, 1));    % Ten percent as the test set
else
    sele_num = 500;
end
Xt_test = X_tar(randIdx(1: sele_num), :);       
Yt_test = Y_tar(randIdx(1: sele_num));
Xt = X_tar(randIdx(sele_num + 1: length(Y_tar)), :);        
Yt = Y_tar(randIdx(sele_num + 1: length(Y_tar)));
nt = length(Y_tar) - sele_num;          

% Test set
dataset.Xt_test = Xt_test;
dataset.Yt_test = Yt_test;

% The remaining part of the target domain and all the source domain samples are used as the training set
dataset.Xs = X_src;
dataset.Ys = Y_src;
dataset.Xt = Xt;
dataset.Yt = Yt;

dataset.nt = nt;
dataset.ns = size(X_src, 1);

    
X=[dataset.Xs;dataset.Xt];
samplemean = mean(X,1);
dataset.Xs = dataset.Xs-repmat(samplemean,size(dataset.Xs,1),1);
dataset.Xt = dataset.Xt-repmat(samplemean,size(dataset.Xt,1),1);
dataset.Xt_test = dataset.Xt_test-repmat(samplemean,size(dataset.Xt_test,1),1);


dataset.d = size(dataset.Xs, 2);



%% For the PR curve
YS = repmat(Y_src, 1, length(Yt_test));     
YT = repmat(Yt_test, 1, length(Y_src));
WTT = (YT==YS');
dataset.WTT = WTT;

YT = repmat(Yt, 1, length(Yt_test));
YTest = repmat(Yt_test,1,length(Yt));
WTT_single = (YTest==YT');
dataset.WTT_single = WTT_single;

end
