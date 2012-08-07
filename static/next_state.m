function [ new_x ] = next_state( old_x, w, dt, min_speed )
%NEXT_STATE Deterministically calculate the next state given the previous
%state, the random variable, and the time difference using a simplified
%dynamic model

assert(numel(dt)==1);
assert(size(w,2)==1);

% If no time has passed (i.e. we're looking at an observation exactly
% after a jump) then return straight away. (This happens at t=0)
if (dt==0)
    new_x=old_x;
    return
end

% Get old state
old_r = old_x(1:2,:);
old_v = old_x(3:4,:);

%Update
new_v = old_v + w*dt;
new_r = old_r + old_v*dt + 0.5*w*dt^2;

% Stack them up
new_x = [new_r; new_v];

% Check
assert(all(isreal(new_x)));

end

