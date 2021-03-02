classdef PSF
    % PSF with  horizon N of a discrete  linear model
    %   Model: x[t+1]=Ax + Bu s.t. Hx<=hx , Hu<=hu
    properties
        mpsf
    end
    
    methods
        function obj = PSF(A, B, Ax, bx, Au, bu,N)
            addpath(genpath("matlab"))
            %PSF Construct an instance of this class

            sys = LinearSystem(A, B, Ax, bx, Au, bu);
            ibsf = IBSF(sys);
            obj.mpsf = MPSF(sys,N,ibsf.P,1,ibsf.K);
            
            
        end
        
        function u = calc(obj,x,u_L)
            % Performs the argmin ||u-u_L|| for the given model.
            
            u = obj.mpsf.solve(x,u_L);
           
        end
    end
end

