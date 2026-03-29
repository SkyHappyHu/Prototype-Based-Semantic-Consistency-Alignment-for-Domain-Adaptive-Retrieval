function [cross_MAP, single_MAP] = PSCA(dataset, param)
    addpath('./utils/');
    Xs = dataset.Xs';        % source data
    Ys = dataset.Ys';
    ns = dataset.ns;        
    Xt = dataset.Xt';        % target data
    nt = dataset.nt;        
    n = ns + nt;
    X_test = dataset.Xt_test';    
    c = dataset.c;          % classes
    d = dataset.d;          % dimension

    %% parameters
    nbit = param.nbits;
    maxIter = param.maxIter;
    lambda1 = param.lambda1;
    lambda2 = param.lambda2;
    lambda3 = param.lambda3;
    hash = param.hash;
    k = param.k;


    %% Initialization
    X = [Xs, Xt];
    [vec,val]  =   eig(X*X');
    [~,Idx]      =   sort(diag(val),'descend');
    W          =   vec(:,Idx(1:k));


    Ls = dummyvar(Ys);
    [Lt] = SPL(W, Xs, Xt, Ys, c);
    Lt_onehot = dummyvar(Lt);


    L = [Ls;Lt_onehot];
    O = O_init(W'*X,L'); % Initialize the prototype

    % Construct MMD matrix
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    M = e * e' * length(unique(Ys));

    dimension = k*2; % The dimension after concatenation
    R = Lt_onehot;    
    WB = rand(nbit,dimension); % Ws, Wt
    [WBL,~,WBR] = svd(WB, 'econ');
    Wt = WBL*WBR';
    Ws = Wt;
    
    obji = 1;
for t = 1:maxIter % step one
    [Lt,probMatrix] = SPL(W, Xs, Xt, Ys, c); % Pseudo labeling
    Lt_onehot = dummyvar(Lt);
 
    Y = [Ls;R];
    D = diag(sum(Y,2)); 
    E = diag(sum(Y));

    % update W  
    temp = sqrt(sum(W.^2, 2) + eps);
    Select = diag(1./(2*temp)); % 21 norm
    Wl = X*D*X'+lambda1*X*M*X'+lambda2*Select;
    Wr = X*Y*O';
    W = pinv(Wl)*Wr;
    
    % update R
    MID = L2_distance_1(W'*Xt,O);
    alpha = compute_geometric_confidence_alpha(MID,probMatrix); % Adaptive factor alpha
    grad_R = 2*R .*MID-alpha.*Lt_onehot./(R+eps);
    temp_R = R - 0.01*grad_R;

    for i=1:1:nt
        R(i,:) = EProjSimplex(temp_R(i,:));
    end

    % update O
    Ot = (W'*X*(Y))* pinv(E);
    [OL,~,OR] = svd(Ot, 'econ');
    O = OL*OR';

    % If the loss value is too small, then break
    obj(t)= norm(Y.*pdist2((W'*X)', O'), 'fro')^2+ lambda1*trace(W'*X*(M)*X'*W)+lambda2*sum(sqrt(sum(W.^2, 2)+eps));
    cver = abs((obj(t) - obji)/obji);
    obji = obj(t);
    t1 = t + 1;
    if (cver < 10^-5 && t1 > 2) , break, end  
end

    % Feature Reconstruction
    Xe = [Ls*O';R*O'];
    Xp = W'*X;
    XX = [Xe,Xp']';
    Ds = XX(:, 1:ns);
    Dt = XX(:, ns+1:n);

    obji =1;
for t = 1:maxIter % Step two 

    % update Bs
    for n= 1:nbit
        Bss(n,:)=Ws(n,:)*Ds;
        Bs(n,:) = sign(Bss(n,:));
    end

    
    % update Bt
    for n= 1:nbit
        Btt(n,:)=Wt(n,:)*Dt;
        Bt(n,:) = sign(Btt(n,:));
    end

    % update Ws
    Ws_l = Ds*Ds'+lambda3*eye(dimension);
    Ws_r = Bs*Ds'+lambda3*Wt;
    Wst = Ws_r*pinv(Ws_l);
    [WsL,~,WsR] = svd(Wst, 'econ');
    Ws = WsL*WsR';

    % update Wt
    Wt_l = Dt*Dt'+lambda3*eye(dimension);
    Wt_r = Bt*Dt'+lambda3*Ws;
    Wtt = Wt_r*pinv(Wt_l);
    [WtL,~,WtR] = svd(Wtt, 'econ');
    Wt = WtL*WtR';
    

    %% Hash learning
    B = [Bs,Bt];
    H = B*X'*pinv(X*X'+hash*eye(d));
   
    % If the loss value is too small, then break
    obj(t)= norm((Ws*Ds-Bs),'fro')^2+norm((Wt*Dt-Bt),'fro')^2+lambda3*norm((Wt-Ws),'fro')^2;
    cver = abs((obj(t) - obji)/obji);
    obji = obj(t);
    t2 = t + 1;
    if (cver < 10^-5 && t2 > 2) , break, end 
end



% B_train1           =    (Xs'*H'>0);
B_train2           =    (Xt'*H'>0);
B_train1           =    (Bs'>0);
% B_train2           =    (Bt'>0); % Out-of-sample extension
B_train = [B_train1;B_train2]';
B_test            =    (X_test'*H'>0);
B_te_comp = compactbit(B_test);
B_train_cross =  B_train(:,1:ns); 
B_tr_comp = compactbit(B_train_cross');
Dhamm = hammingDist(B_te_comp, B_tr_comp);
[recall, precision, ~] = recall_precision(dataset.WTT, Dhamm);
cross_MAP = area_RP(recall, precision) * 100;

B_train_single = B_train(:,ns+1:end); 
B_tr_comp_single = compactbit(B_train_single');
Dhamm = hammingDist(B_te_comp, B_tr_comp_single);
[recall, precision, ~] = recall_precision(dataset.WTT_single, Dhamm);
single_MAP = area_RP(recall, precision) * 100;



function [L,probMatrix] = SPL(W, Xs, Xt, Ys, c)
[k,~] = size(W'*Xs);
domainS_proj = (W'*Xs)';
domainT_proj = (W'*Xt)';
domainS_proj = L2Norm(domainS_proj);        % ||zs||  
domainT_proj = L2Norm(domainT_proj);        % ||zt||

% The distance to the cluster center
classMeans = zeros(c,k);
for i = 1:c
    classMeans(i,:) = mean(domainS_proj(Ys==i,:));
end
classMeans = L2Norm(classMeans);    % Source domain class center

targetClusterMeans = vgg_kmeans(double(domainT_proj'), c, classMeans')';        % Target domain class center
targetClusterMeans = L2Norm(targetClusterMeans);

distClassMeans = EuDist2(domainT_proj,classMeans);      % Distance from target domain samples to the class centers of the source domain
distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);        % Distance from target domain samples to the center of target domain clusters

expMatrix = exp(-distClassMeans);
expMatrix2 = exp(-distClusterMeans);

probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 c]);     % NCP
probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 c]);  % SP


probMatrix = max(probMatrix,probMatrix2); % SPL

    [prob,predLabels] = max(probMatrix');
    L = predLabels;

end
end




function alpha = compute_geometric_confidence_alpha(MID, probMatrix) %Semantic Consistency Alignment
    [n_target, ~] = size(MID);
    alpha = zeros(n_target, 1);
    
    for i = 1:n_target
        % Find the two prototypes that are geographically closest 
        [sorted_dists, geo_order] = sort(MID(i, :));
        closest_proto = geo_order(1);      

        
        % Semantically preferred class
        [sorted_probs, sem_order] = sort(probMatrix(i, :), 'descend');
        preferred_class = sem_order(1);    
        
        % Whether the most recent prototype of geometry matches the most preferred category of semantics
        if closest_proto == preferred_class
            % Consistent situation
            geo_advantage = sorted_dists(2) - sorted_dists(1);  
            sem_advantage = sorted_probs(1) - sorted_probs(2);  
            alpha(i) = sem_advantage / (geo_advantage + eps);   % Semantic advantage / Geometric advantage
        else
            % Conflict: Geometric proximity ¡Ù Semantic preference
            % Comparing the degree of conflict
            conflict_intensity = abs(probMatrix(i, closest_proto) - probMatrix(i, preferred_class));
            alpha(i) = eps + sorted_probs(1).*(1 - conflict_intensity);  % The greater the conflict, the smaller the alpha value
        end
    end
end
