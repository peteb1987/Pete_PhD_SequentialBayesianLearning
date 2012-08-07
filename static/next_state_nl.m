function [ new_x ] = next_state( old_x, w, dt, min_speed )
%NEXT_STATE Deterministically calculate the next state given the previous
%state, the random variable, and the time difference using the various
%dynamic models

assert(numel(dt)==1);
assert(size(w,2)==1);

% If no time has passed (i.e. we're looking at an observation exactly
% after a jump) then return straight away. (This happens at t=0)
if (dt==0)
    new_x=old_x;
    return
end

% Intrinsic case

% Get accelerations
aT = w(1,:);
aN = w(2,:);
aX = w(3:end,:);

% Get old state
old_r = old_x(1:2,:);
old_v = old_x(3:4,:);

% Transform to planar intrinisics
old_sdot = norm(old_v);

% Calculate 2D bearing
old_psi = atan2(old_v(2), old_v(1));
aNc = aN(1,:);


%%% Solve planar differential equation %%%

% speed
aT = max(aT, (min_speed-old_sdot)/dt(end));
new_sdot = old_sdot + aT*dt;

% bearing
if abs(aT)>1E-10
    new_psi = old_psi + (aNc./aT).*log(new_sdot./old_sdot);
else
    new_psi = old_psi + (aNc.*dt)./old_sdot;
end

% displacement
SF = 4*aT.^2 + aNc.^2;

new_u = zeros(2,1);
if (aT~=0)&&(aNc~=0)
    new_u(1,:) = ((new_sdot.^2)./SF).*( aNc.*sin(new_psi)+2*aT.*cos(new_psi)) - ((old_sdot^2)./SF)*( aNc.*sin(old_psi)+2*aT.*cos(old_psi));
    new_u(2,:) = ((new_sdot.^2)./SF).*(-aNc.*cos(new_psi)+2*aT.*sin(new_psi)) - ((old_sdot^2)./SF)*(-aNc.*cos(old_psi)+2*aT.*sin(old_psi));
elseif (aT==0)&&(aNc~=0)
    new_u(1,:) = ((new_sdot.^2)./aNc).*( sin(new_psi) - sin(old_psi) );
    new_u(2,:) = ((new_sdot.^2)./aNc).*(-cos(new_psi) + cos(old_psi) );
elseif (aT~=0)&&(aNc==0)
    new_u(1,:) = 0.5*dt.*cos(old_psi).*new_sdot;
    new_u(2,:) = 0.5*dt.*sin(old_psi).*new_sdot;
else
    new_u(1,:) = ( old_sdot*dt.*cos(old_psi) );
    new_u(2,:) = ( old_sdot*dt.*sin(old_psi) );
end

% Calculate cartesian in-plane velocity
new_udot = zeros(2,1);
[new_udot(1,:), new_udot(2,:)] = pol2cart(new_psi, new_sdot);

% Translate back to original coordinate system
new_r = bsxfun(@plus, new_u+aX*dt, old_r);
new_v = new_udot;

% Stack them up
new_x = [new_r; new_v];

% Check
assert(all(isreal(new_x)));

end

