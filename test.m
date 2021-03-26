A_set = A_ill;
B_set = B_ill./mean(B_ill,[1,3]);

E = sdpvar(nx+1);
Y = sdpvar(nx+1,nu+1);


% Objective: Maximize ellipsoidal volume
objective = -logdet(E);

constraints = [];
%%% BEGIN Constraining the problem %%%

for k = 1:size(A_set,3)    
    
    Ac = A_set(:,:,k);
    Bc = B_set(:,:,k);
    if Ts>0
        sys = c2d(ss(Ac,Bc,[],[]),Ts);
        A = sys.A;
        B = sys.B;
         % Assert controllabilty of the system dyn
        assert(rank(ctrb(A,B))==3)


        % Shiftes the dynamics 
        A0 = [A -A*x0];
        A0(end+1,end)=1;

        B0 = [B -B*u0];
        B0(end+1,end)=1;


        % Positive Definite and Lyapunov Decrease
        constraints= [constraints, [E, (A0*E+B0*Y)'; A0*E+B0*Y, E]>=0];
    else
        A0 = [A -A*x0];
        A0(end+1,end)=0;

        B0 = [B -B*u0];
        B0(end+1,end)=0;
        constraints= [constraints, [ E*A0' + A0*E+Y'*B0'+B0*Y]<=-0.00001];
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
optimize(constraints, objective,opts);   

% Obtain P and K from E and Y
P = inv(value(E));
K = value(Y)*P;


% Project constraints
P =P(1:nx, 1:nx);
K = K(1:nx,1:nu);



%Verify integrity of solution
xplot = sdpvar(nx,1);
Pproj1 = YSet(xplot,(xplot-x0)'*P*(xplot-x0) <= 1);

Px = Polyhedron(Hx,hx);
u = PolyUnion([Pproj1.outerApprox,Px]);
if not(u.isConnected())
     disp()
     disp("WARNING")
     disp("The Ellipse Outer Approx is not connected to constrain polyhedron")
     disp()
end

figure()
hold off
plot(Polyhedron(Hx,hx),'alpha',0.1);
hold on
plot(Pproj1,'alpha',0.1);
savefig(gcf,'Ellips.fig');
save("LastPK",'P','K',"Px");
