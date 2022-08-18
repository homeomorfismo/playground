# Illustration of ngsolve.ngs2petsc as n2p submodule shipped with ngsolve
# (without any add-on), assuming petsc & petsc4py is installed.
# Run this file using eg,   mpirun -np 4 python3 lapl.py

from mpi4py import MPI
import ngsolve as ng
import netgen
import ngsolve.ngs2petsc as n2p
import petsc4py
import petsc4py.PETSc as psc
from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, grad, dx
from ngsolve.ngstd import Timer
from tabulate import tabulate

import sys

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print('Initializing PETSc...')

# PETSc Options
opt = psc.Options()

nref = opt.getInt('nref',1)
hcoarse = opt.getReal('h',0.5)
if comm.rank == 0:
    print('User-defined values')
    print('nref', nref)
    print('hcoarse', hcoarse)
# sys.exit()

# Define PETSC stages

stage_msh = psc.Log.Stage('Meshing')
stage_ngs = psc.Log.Stage('Setting in NGS')
stage_trf = psc.Log.Stage('Transfer ngs2petsc')
stage_ksp = psc.Log.Stage('PETSc solver')

# Define ngstd-Timers

timer_msh = Timer('Meshing')
timer_ngs = Timer('Setting in NGS')
timer_trf = Timer('Transfer ngs2petsc')
timer_ksp = Timer('PETSc solver')

# Generate Netgen mesh and distribute:

timer_msh.Start()
stage_msh.push()

if comm.rank == 0:
    print('Generating Mesh ...')
    ngmesh = unit_cube.GenerateMesh(maxh=hcoarse).Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)

for i in range(nref):
    ngmesh.Refine()
mesh = Mesh(ngmesh)

stage_msh.pop()
timer_msh.Stop()

# sys.exit()

# Do standard NGSolve stuff, but it's parallel now:

timer_ngs.Start()
stage_ngs.push()

V = ng.H1(mesh, order=3, dirichlet=[1, 2, 3, 4])
print('Dofs distributed: rank '+str(comm.rank)+' has '+str(V.ndof) +
      ' of '+str(V.ndofglobal)+' dofs!')
u, v = V.TnT()
a = ng.BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = ng.LinearForm(V)
f += 32 * (y*(1-y)+x*(1-x)) * v * dx
f.Assemble()
uh = ng.GridFunction(V)

stage_ngs.pop()
timer_ngs.Stop()

if comm.rank == 0:
    print('Parallel ngsolve assembly complete')
    print('Converting matrix system to Petsc ...')

# Set up things to move to Petsc.
# n2p = ngsolve.ngs2petsc can transfer vec/mat between NGSolve and Petsc:

timer_trf.Start()
stage_trf.push()

psc_mat = n2p.CreatePETScMatrix(a.mat, V.FreeDofs())
vecmap = n2p.VectorMapping(a.mat.row_pardofs, V.FreeDofs())

# Get some Petsc vectors
psc_f, psc_u = psc_mat.createVecs()

psc_mat.setType('aijcusparse')
psc_f.setType('cuda')
psc_u.setType('cuda')

psc_f.setFromOptions()
psc_u.setFromOptions()
psc_mat.setFromOptions()

stage_trf.pop()
timer_trf.Stop()

# Set up KSP from psc = petsc4py.PETSc

timer_ksp.Start()
stage_ksp.push()

ksp = psc.KSP()
ksp.create()
ksp.setOperators(psc_mat)
ksp.setType(psc.KSP.Type.CG)
ksp.setNormType(psc.KSP.NormType.NORM_NATURAL)
ksp.getPC().setType('gamg')
ksp.setTolerances(rtol=1e-14, atol=0, divtol=1e16, max_it=500)

ksp.setFromOptions()

stage_ksp.pop()
timer_ksp.Stop()

def monitor(ksp, its, rnorm):
    """ I think this is the way petsc4py wants us to set up
        iteration monitoring, not sure though. """

    if comm.rank == 0:
        print('Iteration #', its, ' residual = %2.4e' % rnorm)


ksp.setMonitor(monitor)
if comm.rank == 0:
    print('Prepating to solve with gamg preconditioned CG')


# Convert NGSolve assembled rhs vector to Petsc vector

timer_trf.Start()
stage_trf.push()

vecmap.N2P(f.vec, psc_f)

stage_trf.pop()
timer_trf.Stop()

# Solve using Petsc:

timer_ksp.Start()
stage_ksp.push()

ksp.solve(psc_f, psc_u)

stage_ksp.pop()
timer_ksp.Stop()

# Convert from Petsc to NGSolve:

timer_trf.Start()
stage_trf.push()

vecmap.P2N(psc_u, uh.vec)

stage_trf.pop()
timer_trf.Stop()

# Compute error (in NGSolve)  since  exact solution known:

timer_ngs.Start()
stage_ngs.push()

exact = 16*x*(1-x)*y*(1-y)

error = ng.Integrate((uh-exact)*(uh-exact), mesh)
if comm.rank == 0:
    print('L2-error', error)

stage_ngs.pop()
timer_ngs.Stop()

# Make table - profiling?

prof = {'Rank '+str(comm.rank):['Meshing','NGS settings','Transfer with ngs2petsc','PETSc Solver'],
        'Times':[timer_msh.time,timer_ngs.time,timer_trf.time,timer_ksp.time]}

comm.Barrier()
print(' ')
print(tabulate(prof,headers='keys'))
print(' ')
