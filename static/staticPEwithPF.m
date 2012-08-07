function [ pts_array ] = staticPEwithPF( x0, m0, P0, y, R, Nf )
%STATICPEWITHPF Estimate static parameters with particle flow (no hidden
%state)

K = size(y,2);
pts_array = cell(1,K);

H = [1 0 0 0; 0 1 0 0];
dl = 0.1;
min_speed = 0.01;
s0 = sqrt(x0(3)^2+x0(4)^2);
p0 = atan2(x0(4), x0(3));
ds = length(m0);

% Sample prior
init_pts = zeros(ds,Nf);
init_prob = zeros(Nf,1);
for ii = 1:Nf
    init_pts(:,ii) = mvnrnd(m0, P0);
    init_prob(ii) = log(mvnpdf( init_pts(:,ii)', m0, P0 ));
end

pts = init_pts;
prob = init_prob;

% Loop through time
for kk = 1:K
    
    fprintf(1, 'Processing time step %u.\n', kk);
    
%     fprintf(1, ' start ');
%     figure(1), plot(pts(1,:), pts(2,:), 'x')
%     pause
    
    t = kk;
    
    m = mean(pts, 2);
    P = cov(pts');

    new_pts = pts;
    new_prob = prob;
    
    for ll = 0:dl:1-dl
        
        % Loop through particles
        for ii = 1:Nf
            
            a = pts(:,ii);
            
            %%%%%%%%%%
            
            % Gaussian update
%             aT = a(1); aN = a(2);
%             dx_daT = ((s0 + aT*t)^2*(2*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t))) - aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t)))))/(aN^2 + 4*aT^2) - (2*cos(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) - (2*t*(s0 + aT*t)*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2) + (2*t*(s0 + aT*t)*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2) + (8*aT*(s0 + aT*t)^2*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2)^2 - (8*aT*(s0 + aT*t)^2*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
%             dx_daN = ((s0 + aT*t)^2*(sin(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0) + (aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0))/aT))/(aN^2 + 4*aT^2) - (sin(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) + (2*aN*(s0 + aT*t)^2*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2)^2 - (2*aN*(s0 + aT*t)^2*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
%             dy_daT = (2*t*(s0 + aT*t)*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2) - ((s0 + aT*t)^2*(aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t))) - 2*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT) + 2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t)))))/(aN^2 + 4*aT^2) - (2*sin(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) - (2*t*(s0 + aT*t)*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2) - (8*aT*(s0 + aT*t)^2*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2)^2 + (8*aT*(s0 + aT*t)^2*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
%             dy_daN = (cos(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) + ((s0 + aT*t)^2*(2*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0) - cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + (aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0))/aT))/(aN^2 + 4*aT^2) - (2*aN*(s0 + aT*t)^2*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2)^2 + (2*aN*(s0 + aT*t)^2*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
%             
%             H = [dx_daT dx_daN t 0;
%                 dy_daT dy_daN 0 t];
            
            H = [0.5*t^2 0; 0 0.5*t^2];

            assert(all(isreal(H)));
            
            A = -0.5*P*H'*((R+ll*H*P*H')\H);
            b = (eye(ds)+2*ll*A)*((eye(ds)+ll*A)*P*H'*(R\y(:,kk))+A*m);
            a = a + dl*(A*a+b);
            
            
            %%%%%%%%%%
            
%             %%%%%%%%%%
%             
%             x = next_state(x0,a,t,min_speed);
%             
%             % Approximate prior gradient
%             deltaS = zeros(Nf,1);
%             deltaV = zeros(Nf,ds);
%             for jj = 1:Nf
%                 v = pts(:,jj) - pts(:,ii);
%                 deltaV(jj,:) = v/magn(v);
%                 deltaS(jj,1) = (prob(jj) - prob(ii))/magn(v);
%             end
%             deltaS(ii) = [];
%             deltaV(ii,:) = [];
%             
%             prior_grad = (deltaV'*deltaV)\deltaV'*deltaS;
% %             prior_grad = zeros(size(prior_grad));
%             
%             % Calculate likelihood gradient
% %             aT = a(1); aN = a(2);
% %             dx_daT = ((s0 + aT*t)^2*(2*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t))) - aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t)))))/(aN^2 + 4*aT^2) - (2*cos(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) - (2*t*(s0 + aT*t)*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2) + (2*t*(s0 + aT*t)*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2) + (8*aT*(s0 + aT*t)^2*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2)^2 - (8*aT*(s0 + aT*t)^2*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
% %             dx_daN = ((s0 + aT*t)^2*(sin(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0) + (aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0))/aT))/(aN^2 + 4*aT^2) - (sin(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) + (2*aN*(s0 + aT*t)^2*(2*aT*cos(p0) + aN*sin(p0)))/(aN^2 + 4*aT^2)^2 - (2*aN*(s0 + aT*t)^2*(2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
% %             dy_daT = (2*t*(s0 + aT*t)*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2) - ((s0 + aT*t)^2*(aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t))) - 2*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT) + 2*aT*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*((aN*log((s0 + aT*t)/s0))/aT^2 - (aN*t)/(aT*(s0 + aT*t)))))/(aN^2 + 4*aT^2) - (2*sin(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) - (2*t*(s0 + aT*t)*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2) - (8*aT*(s0 + aT*t)^2*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2)^2 + (8*aT*(s0 + aT*t)^2*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
% %             dy_daN = (cos(p0)*(s0 + aT*t)^2)/(aN^2 + 4*aT^2) + ((s0 + aT*t)^2*(2*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0) - cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) + (aN*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)*log((s0 + aT*t)/s0))/aT))/(aN^2 + 4*aT^2) - (2*aN*(s0 + aT*t)^2*(aN*cos(p0) - 2*aT*sin(p0)))/(aN^2 + 4*aT^2)^2 + (2*aN*(s0 + aT*t)^2*(aN*cos(p0 + (aN*log((s0 + aT*t)/s0))/aT) - 2*aT*sin(p0 + (aN*log((s0 + aT*t)/s0))/aT)))/(aN^2 + 4*aT^2)^2;
% %             lhood_grad = ( ( (y(:,kk)-x(1:2))'*(R\H) ) * [dx_daT dx_daN t 0; dy_daT dy_daN 0 t; 0 0 0 0; 0 0 0 0] )'; 
%             
%             lhood_grad = ( ( (y(:,kk)-x(1:2))'*(R\H) ) * [0.5*t^2 0; 0 0.5*t^2; t 0; 0 t] )';
% 
%             % Calculate likelihood
%             lhood = log( mvnpdf( y(:,kk)', x(1:2)', R ) );
%             
%             % Calculate flow
%             homot_grad = prior_grad+ll*lhood_grad;
%             flow = -lhood*homot_grad/sum(homot_grad.^2);
%             flow(isnan(flow)) = 0;
%             
%             a = a + dl*flow;
%             new_prob(ii) = prob(ii) + dl^2*flow'*prior_grad;
%             
%             %%%%%%%%%%
            
%             if (s0+a(1)*(t+1)) < min_speed
%                 a(1) = (min_speed - s0)/(t+1);
%             end
            
            new_pts(:,ii) = a;
            
        end
        
        pts = new_pts;
        prob = new_prob;
        
%         if rem(ll, 1*dl) == 0
%             fprintf(1, ' %2.1f ', ll+dl);
%             figure(1), plot(pts(1,:), pts(2,:), 'x')
% %             pause
%         end
        
    end
    
    for ii = 1:Nf
        x = next_state(x0,pts(:,ii),t,min_speed);
        lhood = log( mvnpdf( y(:,kk)', x(1:2)', R ) );
        prob(ii) = prob(ii) + lhood;
    end
    
    prob = prob - max(prob);
    
    fprintf(1, '\n');
    pts_array{kk} = pts;
    
end

end

