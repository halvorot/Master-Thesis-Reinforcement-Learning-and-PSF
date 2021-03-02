classdef IBSF
    properties
        sys
        K
        P
    end
    
    methods
        function obj = IBSF(sys)
        obj.sys = sys;
        
        % SDP Variables
        E = sdpvar(sys.n);
        Y = sdpvar(sys.m,sys.n);
        

% --------- Start Modifying Code Here -----------
        % Please use the provided variables
        
        % Objective: Maximize ellipsoidal volume
        objective = -logdet(E);

        % Constraints
        % Positive Definite and Lyapunov Decrease
        constraints = [[E, (obj.sys.A*E+obj.sys.B*Y)'; obj.sys.A*E+obj.sys.B*Y, E]>=0];

        % State constraints
        for i=1:size(obj.sys.Px.A,1)
            constraints=[constraints, [obj.sys.Px.b(i)^2, obj.sys.Px.A(i,:)*E; E*obj.sys.Px.A(i,:)', E]>=0];
        end
        % Input constraints
        for j=1:size(obj.sys.Pu.A,1)
            constraints=[constraints, [obj.sys.Pu.b(j)^2, obj.sys.Pu.A(j,:)*Y;Y'*obj.sys.Pu.A(j,:)', E]>=0];
        end
% --------- End Modifying Code Here -----------

        % Solve
        opts = sdpsettings('verbose',0,'solver','sedumi');
        optimize(constraints, objective,opts);   

% --------- Start Modifying Code Here -----------
        obj.P = inv(value(E));
        obj.K = value(Y)*obj.P;
% --------- End Modifying Code Here -----------
        end
        
        function u_s = filter(obj, x, uL)
            
            % --------- Start Modifying Code Here -----------
            % Predicted state at the next time step
            x_next = obj.sys.A*x + obj.sys.B*uL;
        
            % Check if predicted state is in the safe set and if input constraints are verified
            if x_next'*obj.P*x_next <= 1 && obj.sys.Pu.contains(uL)
                % If so, simulate state using learning-controller
                u_s = uL;
                return
            else
                % Else use safety input
                u_s = obj.K*x;
                return
            end
            % --------- End Modifying Code Here -----------
        end
    end
end

