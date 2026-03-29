function H = O_init(X,Y)
SUM = sum(Y,2);
D = diag(SUM);
H = X*(Y');
H = H*pinv(D);
end
