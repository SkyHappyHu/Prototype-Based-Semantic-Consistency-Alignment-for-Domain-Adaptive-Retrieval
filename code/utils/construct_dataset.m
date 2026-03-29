function dataset = construct_dataset(dataname, test_num, param)
addpath('./data/');
addpath('./data/cross-dataset');
addpath('./data/VLSC');
addpath('./data/Office31');
addpath('./data/Office-Home(vgg)');
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
        
    case 'VOC2007-Caltech101'        
        load VOC2007 data;
        X_src = double(normalize1(data(:, 1:4096)));       
        Y_src = double(data(:, 4097));
        clear data;
        
        load Caltech101 data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data
                
    case 'Caltech256-ImageNet'        
        load dense_caltech256_decaf7_subsampled fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load dense_imagenet_decaf7_subsampled fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
                
    case 'Pr-Rw'
        load ./data/Office-Home(vgg)/Product_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Pr'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Product_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Cl-Rw'        
        load ./data/Office-Home(vgg)/Clipart_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Cl'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Clipart_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Ar-Rw'        
        load ./data/Office-Home(vgg)/Art_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Ar'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Art_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'P27-P05'        
        load PIE27 fea gnd;
        X_src = double(normalize1(fea));
        Y_src = double(gnd);
        
        load PIE05 fea gnd;
        X_tar = double(normalize1(fea));
        Y_tar = double(gnd);
        
    case 'A-W'
        load amazon_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'A-D'
        load ./data/Office31/amazon_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load ./data/Office31/dslr_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'W-D'
        load webcam_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load dslr_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'D-A'
        load dslr_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'W-A'
        load webcam_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'D-W'
        load dslr_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'C1-C2'        
        load ./data/COIL_1 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'C2-C1'        
        load COIL_2 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);    
  
end

c = length(unique(Y_tar));  % The number of classes;
dataset.c = c;



%% 挑选测试集
randIdx = randperm(length(Y_tar));
if test_num < 1
    sele_num = round(test_num * size(X_tar, 1));    % 百分之十作为测试集
else
    sele_num = 500;
end
Xt_test = X_tar(randIdx(1: sele_num), :);       % 测试集
Yt_test = Y_tar(randIdx(1: sele_num));
Xt = X_tar(randIdx(sele_num + 1: length(Y_tar)), :);        % 目标域训练集
Yt = Y_tar(randIdx(sele_num + 1: length(Y_tar)));
nt = length(Y_tar) - sele_num;          % 目标域训练集数量

% 测试集
dataset.Xt_test = Xt_test;
dataset.Yt_test = Yt_test;
% 剩下一部分目标域和全部源域样本作为训练集
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



%% 用于PR曲线
YS = repmat(Y_src, 1, length(Yt_test));     
YT = repmat(Yt_test, 1, length(Y_src));
WTT = (YT==YS');
dataset.WTT = WTT;

YT = repmat(Yt, 1, length(Yt_test));
YTest = repmat(Yt_test,1,length(Yt));
WTT_single = (YTest==YT');
dataset.WTT_single = WTT_single;

end
