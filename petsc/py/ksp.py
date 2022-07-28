# ksp/tutorial/ex1.c

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Parameters 
OptDB = PETSc.Options()
n = OptDB.getInt('n',100)

size = PETSc.COMM_WORLD.getSize()
rank = PETSc.COMM_WORLD.getRank()

# Matrix setting
A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes([n, n])
A.setType('aij')
A.setPreallocationNNZ(5) # What's this?
A.setFromOptions()

Istart, Iend = A.getOwnershipRange()
val0 = -1.0; val1 = 2.0; val2 = -1.0
for i in range(Istart, Iend):
    A[i,i] = val1
    if  i>0 : A[i-1,i] = val0
    if  i<n-1 : A[i+1,i] = val2

A.assemblyBegin()
A.assemblyEnd()

A.setOption(A.Option.SYMMETRIC,True)

# Vector setting
x = PETSc.Vec().create(PETSc.COMM_WORLD)
x.setSizes(n)
x.setFromOptions()

b = x.duplicate()
u = x.duplicate()

u.set(1.0)
b = A(u)

## View exact solution if desired
flg = OptDB.getBool('view_exact_sol', False)
if flg:
    u.view()

# KSP setting
ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
ksp.setType('cg')
ksp.setOperators(A,A)

## KSP options
ksp.max_it = 10000
ksp.rtol = 1e-7
ksp.atol = 1e-50
ksp.setFromOptions()

# Solver
ksp.solve(b,x)
print("Iterations: %d. Residual norm: %.6g" % (ksp.its, ksp.norm))

x = x - u # x.axpy(-1.0, u)
norm = x.norm(PETSc.NormType.NORM_2)
its = ksp.getIterationNumber()

# Printing routine
if norm > 1.0e-2:
    PETSc.Sys.Print("Norm of error {}, Iterations {}".format(norm,its),comm=comm)
else:
    if size==1:
        PETSc.Sys.Print("- Serial OK",comm=PETSc.COMM_WORLD)
    else:
        PETSc.Sys.Print("- Parallel OK",comm=PETSc.COMM_WORLD)
