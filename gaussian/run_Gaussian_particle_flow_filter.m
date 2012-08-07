% Clear up
clup
dbstop if error

rand_seed = 0;

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Set parameters
set_ar1_parameters;

% Generate some data
[ t, x, y ] = generate_ar1_data(params);

% Run the filter
init_pts = mvnrnd(hyper_params.start_prior_mn, hyper_params.start_prior_var, params.Nx);
[ x_pts_array ] = particle_flow_filter( init_pts, t, y, params, @ar1_trans, @ar1_obs );

% % Run the parameter learning filter
% init_x_pts = mvnrnd(hyper_params.start_prior_mn, hyper_params.start_prior_var, params.Nx);
% init_tau_pts = 1./gamrnd(hyper_params.procprec_prior_shape, hyper_params.procprec_prior_scale, [params.Nx,1]);
% norm_vars = init_tau_pts/hyper_params.decay_prior_df;
% init_alpha_pts = normrnd(repmat(hyper_params.decay_prior_mn, params.Nx, 1), sqrt(norm_vars));

% [ x_pts_array, tau_pts_array, alpha_pts_array ] = particle_flow_PE_filter( init_x_pts, init_tau_pts, init_alpha_pts, t, y, params, @ar1_trans, @ar1_obs );

%% Kalman filter
[kf_m, kf_P] = kf_loop(hyper_params.start_prior_mn, hyper_params.start_prior_var, 1, params.obs_var, y, params.decay, params.proc_var);
kf_P = squeeze(kf_P)';

% procprec_mn = 1; procprec_vr = 1;
% decay_vr = 1;
% [kf_m, kf_P] = kf_loop([hyper_params.start_prior_mn; procprec_mn; hyper_params.decay_prior_mn], diag([hyper_params.start_prior_var, procprec_vr, decay_vr]), [1 0 0], params.obs_var, y, diag([params.decay 1 1]), diag([params.proc_var 0 0]));

%% Analyse

pf_mse = sqrt(sum((x - mean(x_pts_array, 1)).^2))
kf_mse = sqrt(sum((x - kf_m).^2))


%% Output

figure(1), hold on
plot(t, x_pts_array');
plot(t, x, 'b', 'linewidth', 2);
plot(t, y, 'r', 'linewidth', 2);

figure(2), hold on
plot(t, x, 'b', 'linewidth', 2);
plot(t, y, 'r', 'linewidth', 2);
plot(t, mean(x_pts_array,1), 'k')
plot(t, mean(x_pts_array,1)+2*std(x_pts_array), ':k')
plot(t, mean(x_pts_array,1)-2*std(x_pts_array), ':k')
plot(t, kf_m, 'g')
plot(t, kf_m+2*sqrt(kf_P), ':g')
plot(t, kf_m-2*sqrt(kf_P), ':g')
