classdef MPSF < handle
    properties
        sys         % LinearSystem instance defining dynamics & constraints
        optimizer   % Yalmip optimizer
        N           % Prediction horizon
        P           % Shape matrix defining terminal set x_N^T P x_N < alpha
        alpha       % Terminal set level for level set
        Uc          % Candidate solution
        Xc          % Candidate solution
        K           % local controller for candidate solutions
    end
    
    methods
        function obj = MPSF(sys,N,P,alpha,K)
            obj.sys = sys;
            obj.N = N;
            obj.P = P;
            obj.alpha = alpha;
            obj.Uc = zeros(sys.m,N);
            obj.Xc = zeros(sys.n,N+1);
            obj.K = K;
            
            x_i=sdpvar(obj.sys.n, obj.N+1); %state
            u_i=sdpvar(obj.sys.m, obj.N); %input
            x_0=sdpvar(obj.sys.n,1); %initial state
            u_L=sdpvar(obj.sys.m,1); %learning input
            eps=sdpvar(1,1); %slack
            
            % --------- Start Modifying Code Here -----------
            % Define Cost Function & Constraints
            %objective_MPC = norm(u_L - u_i(:,1),2);
            objective_MPC = (u_L - u_i(:,1))'*(u_L - u_i(:,1));
            % --------- Stop Modifying Code Here -----------
            
            % Initial State Constraints
            constraints_MPC=[x_i(:,1)==x_0];
                        
            for i=1:obj.N
                % State Propagation Constraints
                constraints_MPC=[constraints_MPC, x_i(:,i+1) == obj.sys.A*x_i(:,i)+obj.sys.B*u_i(:,i)];

                % State & Input Constraints
                constraints_MPC=[constraints_MPC, obj.sys.Px.A*x_i(:,i+1)<=(1+eps)*obj.sys.Px.b];
                constraints_MPC=[constraints_MPC, obj.sys.Pu.A*u_i(:,i)<=obj.sys.Pu.b];
            end

            % Terminal constraint
            %constraints_MPC=[constraints_MPC, x_i(:,obj.N+1)'*obj.P*x_i(:,obj.N+1)<=obj.alpha];
            constraints_MPC=[constraints_MPC, norm(chol(obj.P)*x_i(:,obj.N+1),2)<=obj.alpha];

            % soft constraints / slack
            objective_MPC = objective_MPC + 1e6*eps;
            constraints_MPC=[constraints_MPC, eps >= 0];    
            

            % Optimizer allows to solve optimisation problem repeatedly in a fast
            % manner. Inputs are x_0 and outputs are u_i, x_i
            % To speed up the solve time, MOSEK can be used with an academic license.
            % Replace [] with sdpsettings('solver','mosek') if installed. Other solvers
            % can be used as well (OSQP,...)
            ops = sdpsettings('verbose',0,'solver','mosek');
            %ops = sdpsettings('verbose',1,'solver','sedumi');
            obj.optimizer=optimizer(constraints_MPC, objective_MPC, ops, {x_0,u_L}, {u_i,x_i});
        end
        
        function [u, U, X] = solve(obj, x, u_L)
            % Call optimizer and check solve status
            
            [sol, flag] = obj.optimizer(x, u_L);
            U = sol{1};
            X = sol{2};
            
            if flag~=0 && flag~=4 % Sedumi keeps having numerical problems
            %if flag~=0
                warning(yalmiperror(flag))
                if any(isnan(U))
                    U = [obj.Uc(:,2:end), obj.K*obj.Xc(:,end)];
                    X = [obj.Xc(:,2:end), (obj.sys.A+obj.sys.B*obj.K)*obj.Xc(:,end)];
                    warning('MPSF returned NaN, overwriting with candidate solution')
                end
            end
            obj.Uc = U;
            obj.Xc = X;
            u = U(:,1);
            
        end
    end
end

