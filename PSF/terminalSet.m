function [P, K, x0, u0] = terminalSet(A_set,B_set, Hx, Hu, hx, hu, Ts)
eps=1e-6;

nx = size(Hx,2);
nu = size(Hu,2);

A_set = reshape(A_set,nx,nx,numel(A_set)/nx^2);
B_set = reshape(B_set,nx,nu,numel(B_set)/(nx*nu));

opts=optimoptions('fmincon','Algorithm','interior-point', 'Display','off');
% Finds centroid, x0 u0, of polyhedral constrainst
fx  = @(x)sum((Hx*x-hx).^2);
fu  = @(u)sum((Hu*u-hu).^2);

x0 = fmincon(fx,zeros(nx,1),Hx,hx,[],[],[],[],[], opts);
u0 =  fmincon(fu,zeros(nu,1),Hu,hu,[],[],[],[],[],  opts);




% Scale B for better performance
B_scale=max(B_set,[],[1,3])-min(B_set,[],[1,3]);
B_set = B_set./B_scale;

Hu = Hu./B_scale;
Pu_scale =max([Hu,hu],[],2)-min([Hu,hu],[],2);
Hu = Hu./Pu_scale;
hu = hu./Pu_scale;


% Shifts the constrainst to x0,u0
hx0 = hx-Hx*x0;

Hx0= Hx;
Hx0(end,end+1)=0;

hu0 = hu-Hu*u0;

Hu0= Hu;

E = sdpvar(nx+1);
Y = sdpvar(nu,nx+1);


% Objective: Maximize ellipsoidal volume
objective = -logdet(E);

constraints = [];
%%% BEGIN Constraining the problem %%%
size(B_set)
for k = 1:size(A_set,3)    
    
    Ac = A_set(:,:,k);
    Bc = B_set(:,:,k);
    if Ts>0
        sys = c2d(ss(Ac,Bc,[],[]),Ts);
        A = sys.A;
        B = sys.B;
         % Assert controllabilty of the system dyn
        assert(rank(ctrb(A,B))==3)

% 
%         % Shiftes the dynamics 
         A0 = [A -A*x0-B*u0];
         A0(end+1,end)=1-eps;
% 
         B0 = B ;
         B0(end+1,end)=0;

        % Positive Definite and Lyapunov Decrease
        constraints= [constraints, [E, (A0*E+B0*Y)'; A0*E+B0*Y, E]>=0];
    else
        A0 = [Ac -Ac*x0];
        A0(end+1,end)=0;

        B0 = [Bc -Bc*u0];
        B0(end+1,end)=0;

        constraints= [constraints, [ E*A0' + A0*E+Y'*B0'+B0*Y]<=0];
    end
end

%State constraints
for i=1:size(Hx,1)
    constraints=[constraints, [hx0(i)^2, Hx0(i,:)*E; E*Hx0(i,:)', E]>=0];
end
%Input constraints
for j=1:size(Hu,1)
    constraints=[constraints, [hu0(j)^2, Hu0(j,:)*Y;Y'*Hu0(j,:)', E]>=0];
end


%%% END Constraining the problem %%% 

% Solve
opts = sdpsettings('verbose',2,'solver','mosek');
a=optimize(constraints, objective,opts);
error_flag = false;
if a.problem >0
    error_flag = true;
    disp("***")
     disp("WARNING")
     disp("The problem migth be invalid. Solver did not converge.")
     disp(a)
     disp("***")
end

% Obtain P and K from E and Y
P = inv(value(E));
K = value(Y)*P;

K = K(1:nu,1:nx);
P = P(1:nx,1:nx);

% 
% 
%Verify integrity of solution
xplot = sdpvar(nx,1);
Pproj1 = YSet(xplot,(xplot-x0)'*P*(xplot-x0) <= 1);

Px = Polyhedron(Hx,hx);
u = PolyUnion([Pproj1.outerApprox,Px]);
[primal_feas, dual_feas]=check(constraints);
if any(primal_feas<-eps)
     disp("***")
     disp("WARNING")
     disp("The problem migth be infeasible. Primal feasiblity threshold violated")
     disp("***")
     disp(primal_feas)
     error_flag = true;

end
if not(u.isConnected())
     disp("***")
     disp("WARNING")
     disp("The Ellipse Outer Approx is not connected to constrain polyhedron.")
     disp("***")
     error_flag = true;

end
if error_flag
    disp("Debug files are created for inspection")
    figure()

    plot(Polyhedron(Hx,hx),'alpha',0.1);
    hold on
    plot(Pproj1,'alpha',0.1);
    savefig(gcf,'debug_terminal_set.fig');
    saveas(gcf,'debug_terminal_set', 'pdf');
    save("debug_PK",'P','K',"Px");
end

end

