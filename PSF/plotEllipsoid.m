function plot_flag = plotEllipsoid(P, Hx,hx)
nx=size(Hx,2);
x = sdpvar(nx,1);
Pproj1 = YSet(x,x'*P*x <= 1);

Px = Polyhedron(Hx,hx);
figure()

plot(Px,'alpha',0.1);
hold on
plot(Pproj1,'alpha',0.1);
savefig(gcf,'debug_terminal_set.fig');
saveas(gcf,'debug_terminal_set', 'pdf');

plot_flag = true; 
end

