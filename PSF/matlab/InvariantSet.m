function [outputArg1,outputArg2] = InvariantSet(A_set,B_set, Hx, Hu, hx, hu)



nx = size(A_set{1},2);
nu = size(B_set{1},2);

E = sdpvar(nx);
Y = sdpvar(nx,nu);


% --------- Start Modifying Code Here -----------
% Please use the provided variables

% Objective: Maximize ellipsoidal volume
objective = -geomean(E);

constraints = [];
% Constraints
% Positive Definite and Lyapunov Decrease
for k = 1:numel(A)
    A = A_set{k};
    B = B_set{k};
    constraints= [constraints, [E, (A*E+B*Y)'; A*E+B*Y, E]>=0];
end

% State constraints
for i=1:size(Hx,1)
    constraints=[constraints, [hx(i)^2, Hx(i,:)*E; E*Hx(i,:)', E]>=0];
end
% Input constraints
for j=1:size(Hu,1)
    constraints=[constraints, [hu(j)^2, Hu(j,:)*Y;Y'*Hu(j,:)', E]>=0];
end
% --------- End Modifying Code Here -----------

% Solve
opts = sdpsettings('verbose',2,'solver','mosek');
optimize(constraints, objective,opts);   

% --------- Start Modifying Code Here -----------
obj.P = inv(value(E));
obj.K = value(Y)*obj.P;
end

