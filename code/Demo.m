close all;
warning off;
rng(3407); % Seed setting
addpath('./utils/');
param.maxIter = 15; % Iterations
param.plot_loss_acc = 0;      % Plot convergence?

    % C1-C2
    % lambda1s = [1]; % MMD
    % lambda2s = [1]; %21 norm
    % lambda3s = [100]; % Classifier approximation
    % hashs = [0.1]; % Subspace dimension
    % k = [32]; 
    
    % MNIST-USPS
    lambda1 = [1];
    lambda2 = [1];
    lambda3 = [100];
    hashs = [0.1];
    k = [64];


    test_num = 0.1;         % Number of test set: 10% / 500
    DB_name = 'MNIST-USPS';
%% ---------------------------------------------------------------
    nbits = [64]; % Hash code length
    mAP = [];
    times = 1;
    cross_MAP = zeros(1,times);
    single_MAP = zeros(1,times);
        max_cross_MAP = zeros(1,times);
  for t = 1:times
                            dataset = construct_dataset(DB_name, test_num, param);
                            param.nbits = nbits;
                            param.lambda1 = lambda1;           
                            param.lambda2 = lambda2;
                            param.lambda3 = lambda3;
                            param.hash = hashs;  
                            param.k = k;                        
                            [cross_MAP(t), single_MAP(t)] = PSCA(dataset, param);
                            fprintf('bit=%f,lambda1=%.2f, lambda2=%.2f, lambda3=%.2f,hash=%.2f,k=%.2f, cross_MAP=%.2f, single_MAP=%.2f \n'...
                                 ,param.nbits,param.lambda1,param.lambda2,param.lambda3,param.hash,param.k,cross_MAP(t),single_MAP(t));
  end




            