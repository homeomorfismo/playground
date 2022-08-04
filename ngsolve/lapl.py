# Illustration of ngsolve.ngs2petsc as n2p submodule shipped with ngsolve
# (without any add-on), assuming petsc & petsc4py is installed.
# Run this file using eg,   mpirun -np 4 python3 lapl.py


from mpi4py import MPI
import petsc4py.PETSc as psc
import ngsolve as ng
import netgen
import ngsolve.ngs2petsc as n2p
from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, grad, dx


# Generate Netgen mesh and distribute:

comm = MPI.COMM_WORLD
if comm.rank == 0:
    ngmesh = unit_cube.GenerateMesh(maxh=0.1).Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)

for i in range(1):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

# Do standard NGSolve stuff, but it's parallel now:

V = ng.H1(mesh, order=3, dirichlet=[1, 2, 3, 4])
print('Dofs distributed: rank '+str(comm.rank)+' has '+str(V.ndof) +
      ' of '+str(V.ndofglobal)+' dofs!')
u, v = V.TnT()
a = ng.BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = ng.LinearForm(V)
f += 32 * (y*(1-y)+x*(1-x)) * v * dx
f.Assemble()
uh = ng.GridFunction(V)

if comm.rank == 0:
    print('Parallel ngsolve assembly complete')
    print('Converting matrix system to Petsc ...')

# Set up things to move to Petsc.
# n2p = ngsolve.ngs2petsc can transfer vec/mat between NGSolve and Petsc:

psc_mat = n2p.CreatePETScMatrix(a.mat, V.FreeDofs())
vecmap = n2p.VectorMapping(a.mat.row_pardofs, V.FreeDofs())

# Get some Petsc vectors

psc_f, psc_u = psc_mat.createVecs()

psc_f.setFromOptions()
psc_u.setFromOptions()
psc_mat.setFromOptions()

# Set up KSP from psc = petsc4py.PETSc

ksp = psc.KSP()
ksp.create()
ksp.setOperators(psc_mat)
ksp.setType(psc.KSP.Type.CG)
ksp.setNormType(psc.KSP.NormType.NORM_NATURAL)
ksp.getPC().setType('gamg')
ksp.setTolerances(rtol=1e-14, atol=0, divtol=1e16, max_it=500)

ksp.setFromOptions()


def monitor(ksp, its, rnorm):
    """ I think this is the way petsc4py wants us to set up
        iteration monitoring, not sure though. """

    if comm.rank == 0:
        print('Iteration #', its, ' residual = %2.4e' % rnorm)


ksp.setMonitor(monitor)
if comm.rank == 0:
    print('Prepating to solve with gamg preconditioned CG')


# Convert NGSolve assembled rhs vector to Petsc vector

vecmap.N2P(f.vec, psc_f)

# Solve using Petsc:

ksp.solve(psc_f, psc_u)

# Convert from Petsc to NGSolve:

vecmap.P2N(psc_u, uh.vec)


# Compute error (in NGSolve)  since  exact solution known:

exact = 16*x*(1-x)*y*(1-y)

error = ng.Integrate((uh-exact)*(uh-exact), mesh)
if comm.rank == 0:
    print('L2-error', error)
