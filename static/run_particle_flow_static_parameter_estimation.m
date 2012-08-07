clup

dbstop if error
dbstop if warning

rand_seed = 0;

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Parameters
K = 200;
R = 100*eye(2);
% m0 = [5 5 0 0];
% P0 = 10*eye(4);
m0 = [-0.2 0.3];
P0 = 10*eye(2);

Nf = 100;

% Create some nonlinear data depending on a single parameter vector
% a = [0.1 1 0 0]';
a = [-0.2 0.3]';
x0 = [0 0 10, 10]';

x = zeros(4, 100);
y = zeros(2, 100);
for kk = 1:K
    x(:,kk) = next_state(x0, a, kk, 0.1);
    y(:,kk) = mvnrnd(x(1:2,kk), R);
end

% % Plot it
% figure, hold on;
% plot(x(1,:), x(2,:), 'b');
% plot(y(1,:), y(2,:), 'xr');

%% Grid search

figure(1), clf
list = -10:0.1:10;
prob = zeros(length(list));
x_GS_mode = zeros(2,K);
for kk = 1:K
    kk
    t = kk;
    for ii = 1:length(list)
        for jj = 1:length(list)
            a = [list(ii); list(jj)];
            xkk = next_state(x0, a, t, -inf);
            prob(ii, jj) = prob(ii, jj) + log( mvnpdf( y(:,kk)', xkk(1:2)', R ) );
        end
    end
    prob = prob - max(max(prob));
    [ii, jj] = find(prob==max(prob(:)));
    x_GS_mode(:,kk) = [list(ii); list(jj)];
end

%% Estimate with particle flow

x_PF_pts = staticPEwithPF(x0, m0, P0, y, R, Nf);
x_PF_mn = cell2mat(cellfun( @(x) {mean(x,2)}, x_PF_pts));

%% Estimate with a UKF
x_UT_mn = zeros(2,K);

dr = 2; do = 2; ds = 4;
[WM,WC,c] = ut_weights(dr+do, 0.5, 2, dr+do+1);
N_sigs = length(WM);
w_mn = zeros(dr, 1);
w_var = P0;

for kk = 1:K
    t = kk;
    % Calculate sigma points
    W = ut_sigmas([w_mn; zeros(do,1)],[w_var zeros(dr,do); zeros(do,dr) R],c);
    
    % Propagate SPs through transition and observation functions
    x_sigmas = zeros(ds,N_sigs);
    Y = zeros(do,N_sigs);
    for ii = 1:N_sigs
        x_sigmas(:,ii) = next_state(x0,W(1:dr,ii),t,-inf);
        Y(:,ii) = x_sigmas(1:2,ii);
    end
    
    % Add observation noise
    Y = Y + W(dr+1:end,:);
    
    % Calculate observation mean, covariance and cross-covariance
    mu = Y*WM;
    Y_diff = bsxfun(@minus, Y, mu);
    W_diff = bsxfun(@minus, W(1:dr,:), w_mn);
    S = Y_diff*diag(WC)*Y_diff';
    C = W_diff*diag(WC)*Y_diff';
    
    % Update recursions
    gain = C / S;
    w_mn = w_mn + gain * (bsxfun(@minus, y(:,kk), mu));
    w_var = w_var - gain * S * gain';
    
    x_UT_mn(:,kk) = w_mn;
    
end

%% Estimate with a MCMC

%% Plotting
for dd = 1:2
    figure, hold on,
    plot([0, K], [a(dd), a(dd)], 'k', 'linewidth', 2);
    plot(1:K, x_PF_mn(dd,:), 'b');
end