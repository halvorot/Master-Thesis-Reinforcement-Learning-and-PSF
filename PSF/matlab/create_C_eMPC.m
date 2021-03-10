function create_C_eMPC(A, B, Rx,Ru,N)

model = LTISystem('A',A,'B',B);

model.setDomain('x',Rx);
model.setDomain('u',Ru);
model.u.with('reference');
model.u.reference = 'free';

model.u.penalty = OneNormFunction(eye(Ru.Dim));
ctrl = MPCController(model,N);

ectrl=ctrl.toExplicit();
simp=ectrl.simplify();

ectrl.exportToC('output','c_mpc')
end

