function [x,u] = SimulateSF(sys,cntr, x_init, T)
if nargin < 5
    w = zeros(size(x_init,1),T);
end

Ns = size(x_init,2);
% Set up variables
x = zeros(sys.n,Ns,T+1);
u = zeros(sys.m,Ns,T);
x(:,:,1)=x_init;


for j=1:Ns
    for i=1:T
        u(:,j,i) = cntr(x(:,j,i));
        x(:,j,i+1) = sys.step(x(:,j,i),u(:,j,i));
    end
end
% ==============
% Plot System Trajectory within Constraints
% ==============

figure()
% Plot trajectory 
plot(squeeze(x(1,:,:))',squeeze(x(3,:,:))','r')
hold on
% Plot state constraints
plot(sys.Px.projection([1,3]),'alpha',0.1)
hold off
xlabel('X-Position')
ylabel('Y-Position')

figure()
% Plot trajectory 
plot(squeeze(x(2,:,:))',squeeze(x(4,:,:))','r')
hold on
% Plot state constraints
plot(sys.Px.projection([2,4]),'alpha',0.1)
hold off
xlabel('X-Velocity')
ylabel('Y-Velocity')

figure()
subplot(2,1,1)
plot(squeeze(u(1,:,:))','b')
hold on
plot([1,T],[sys.umax(1), sys.umax(1)],'k--')
plot([1,T],[sys.umin(1), sys.umin(1)],'k--')
hold off
xlim([1,T])
ylabel('Input 1')
xlabel('Time steps')

subplot(2,1,2)
plot(squeeze(u(2,:,:))','b')
hold on
plot([1,T],[sys.umax(2), sys.umax(2)],'k--')
plot([1,T],[sys.umin(2), sys.umin(2)],'k--')
hold off
xlim([1,T])
ylabel('Input 2')
xlabel('Time steps')

end

