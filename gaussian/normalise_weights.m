function [ w ] = normalise_weights( w )
%NORMALISE_WEIGHTS Normalise a set of log-weight

% Normalise
w = w - logsumexp(w);

% Assert no error (usually caused by 0s or infs)
assert(~any(isnan(w)));

end

