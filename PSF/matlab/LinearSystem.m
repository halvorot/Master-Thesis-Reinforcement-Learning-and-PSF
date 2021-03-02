classdef LinearSystem
    %SYSTEM System Class for programming exercise
    %       
    %     Args:
    %         A : Dynamic matrix A; x^+ = A*x + B*u
    %         B : Dynamic matrix B; x^+ = A*x + B*u
    %         Hx : State constraint matrix; Ax*x <= bx
    %         bx : State constraint vector; Ax*x <= bx
    %         Hu : State constraint matrix; Au*u <= bu
    %         bu : State constraint vector; Au*u <= bu
    % 
    %     Properties:
    %         A : Dynamic matrix A; x^+ = A*x + B*u
    %         B : Dynamic matrix B; x^+ = A*x + B*u
    %         Px : Polyhedron; Ax*x <= bx
    %         Pu : Polyhedron Au*u <= bu
    %         umax : bounding box around Au*u <= bu implies u <= umax
    %         umin : bounding box around Au*u <= bu implies u >= umin
    %         n : Number of states
    %         m : Number of inputs
    %         nx : Number of state constraints/half-spaces
    %         nu : Number of input constraints/half-spaces

    properties
        A
        B
        Px
        Pu
        umax
        umin
        xmax
        xmin
        n
        m
        nx
        nu
    end
    
    methods
        function obj = LinearSystem(A, B, Ax, bx, Au, bu)
            obj.A = A;
            obj.B = B;
            obj.Px = Polyhedron(Ax,bx);
            obj.Pu = Polyhedron(Au,bu);
            obj.n = size(A,2);
            obj.m = size(B,2);
            obj.nx = size(bx,1);
            obj.nu = size(bu,1);
            
            %compute bounding boxes
            aux = obj.Pu.outerApprox;
            obj.umax = aux.b(1:obj.m);
            obj.umin = -aux.b(obj.m+1:end);
            
            aux = obj.Px.outerApprox;
            obj.xmax = aux.b(1:obj.n);
            obj.xmin = -aux.b(obj.n+1:end);
        end
        
        function xp = step(obj,x,u,w)
            if nargin < 3       % autonomous
                xp = obj.A*x;   
            elseif nargin < 4   % controlled
                xp = obj.A*x + obj.B*u;
            else                % with disturbance
                xp = obj.A*x + obj.B*u + w;
            end
            
        end
    end
end

