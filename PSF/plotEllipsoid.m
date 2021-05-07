function plot_flag = plotEllipsoid(P, Hx,hx,x0,filestr)
nx=size(Hx,2);
x = sdpvar(nx,1);
Pproj1 = YSet(x,(x-x0)'*P*(x-x0) <= 1);

Px = Polyhedron(Hx,hx);
figure()

plot(Px,'alpha',0.1);
hold on
plot(Pproj1,'alpha',0.1);
savefig(gcf,filestr);
saveas(gcf,filestr, 'epsc');

plot_flag = true; 
end

