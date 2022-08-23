# Illustration of ngsolve.ngs2petsc as n2p submodule shipped with ngsolve
# (without any add-on), assuming petsc & petsc4py is installed.
# Run this file using, eg., mpiexec -n $NCORES python3 stokes-refine.py

from mpi4py import MPI
import ngsolve as ng
import netgen
import ngsolve.ngs2petsc as n2p
import petsc4py
import petsc4py.PETSc as psc
from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, z, grad, dx
from ngsolve.ngstd import Timer
from tabulate import tabulate

import sys

comm = MPI.COMM_WORLD

if comm.rank == 0:
    print('Initializing PETSc...')

# PETSc Options
## Add high order Taylor-Hood
opt = psc.Options()

nref = opt.getInt('nref',1)
hcoarse = opt.getReal('h',0.5)
order_th = opt.getInt('ord',2)
#dim = opt.getInt('dim',3)

if comm.rank == 0:
    print('--- User-defined values ---')
    print('nref ', nref)
    print('hcoarse ', hcoarse)
    print('Order Taylor-Hood (Velocity) ', order_th)
    #print('Dimension',dim)
    print('--- ---')
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
    ngmesh = unit_cube.GenerateMesh(maxh=hcoarse)
    for i in range(nref):
        ngmesh.Refine()
    ngmesh = ngmesh.Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)
mesh=Mesh(ngmesh)

comm.Barrier()

stage_msh.pop()
timer_msh.Stop()

# Do standard NGSolve stuff, but it's parallel now:

timer_ngs.Start()
stage_ngs.push()

## Define Stokes problem setting
### Taylor-Hood
V = ng.H1(mesh, order=order_th, dirichlet=[1, 2, 3, 4])
# V = ng.VectorH1(mesh, order=order_th, dirichlet=[1, 2, 3, 4])
# V.SetOrder(TRIG,dim)
# V.Update()
Q = ng.H1(mesh, order=order_th-1)
### 3d prob
X = V*V*V*Q

if comm.rank == 0:
    print('--- DoFs distribution ---')
comm.Barrier()
print('Rank '+str(comm.rank)+' has '+str(V.ndof) +
      ' of '+str(V.ndofglobal)+' dofs!')

### Trial and test vars
ux,uy,uz,p = X.TrialFunction()
vx,vy,vz,q = X.TestFunction()

div_u = grad(ux)[0]+grad(uy)[1]+grad(uz)[2]
div_v = grad(vx)[0]+grad(vy)[1]+grad(vz)[2]

### Linear forms
a = ng.BilinearForm(X)
a += (grad(ux)*grad(vx)+grad(uy)*grad(vy)+grad(uz)*grad(vz))*dx
a += (div_u*q)*dx
a += (p*div_v)*dx
a.Assemble()

f = ng.LinearForm(X)
comp1 = -24*x**4*y**4*z + 12*x**4*y**4 + 48*x**4*y**3*z**2 - 16*x**4*y**3 - 48*x**4*y**2*z**3 + 24*x**4*y**2*z + 24*x**4*y*z**4 - 24*x**4*y*z**2 + 4*x**4*y - 12*x**4*z**4 + 16*x**4*z**3 - 4*x**4*z + 5*x**4 + 48*x**3*y**4*z - 24*x**3*y**4 - 96*x**3*y**3*z**2 + 32*x**3*y**3 + 96*x**3*y**2*z**3 - 48*x**3*y**2*z - 48*x**3*y*z**4 + 48*x**3*y*z**2 - 8*x**3*y + 24*x**3*z**4 - 32*x**3*z**3 + 8*x**3*z - 48*x**2*y**4*z**3 + 72*x**2*y**4*z**2 - 48*x**2*y**4*z + 12*x**2*y**4 + 48*x**2*y**3*z**4 - 48*x**2*y**3*z**2 + 48*x**2*y**3*z - 16*x**2*y**3 - 72*x**2*y**2*z**4 + 48*x**2*y**2*z**3 + 48*x**2*y*z**4 - 48*x**2*y*z**3 + 4*x**2*y - 12*x**2*z**4 + 16*x**2*z**3 - 4*x**2*z + 48*x*y**4*z**3 - 72*x*y**4*z**2 + 24*x*y**4*z - 48*x*y**3*z**4 + 96*x*y**3*z**2 - 48*x*y**3*z + 72*x*y**2*z**4 - 96*x*y**2*z**3 + 24*x*y**2*z - 24*x*y*z**4 + 48*x*y*z**3 - 24*x*y*z**2 - 8*y**4*z**3 + 12*y**4*z**2 - 4*y**4*z + 8*y**3*z**4 - 16*y**3*z**2 + 8*y**3*z - 12*y**2*z**4 + 16*y**2*z**3 - 4*y**2*z + 4*y*z**4 - 8*y*z**3 + 4*y*z**2
comp2 = 24*x**4*y**4*z - 12*x**4*y**4 - 48*x**4*y**3*z + 24*x**4*y**3 + 48*x**4*y**2*z**3 - 72*x**4*y**2*z**2 + 48*x**4*y**2*z - 12*x**4*y**2 - 48*x**4*y*z**3 + 72*x**4*y*z**2 - 24*x**4*y*z + 8*x**4*z**3 - 12*x**4*z**2 + 4*x**4*z - 48*x**3*y**4*z**2 + 16*x**3*y**4 + 96*x**3*y**3*z**2 - 32*x**3*y**3 - 48*x**3*y**2*z**4 + 48*x**3*y**2*z**2 - 48*x**3*y**2*z + 16*x**3*y**2 + 48*x**3*y*z**4 - 96*x**3*y*z**2 + 48*x**3*y*z - 8*x**3*z**4 + 16*x**3*z**2 - 8*x**3*z + 48*x**2*y**4*z**3 - 24*x**2*y**4*z - 96*x**2*y**3*z**3 + 48*x**2*y**3*z + 72*x**2*y**2*z**4 - 48*x**2*y**2*z**3 - 72*x**2*y*z**4 + 96*x**2*y*z**3 - 24*x**2*y*z + 12*x**2*z**4 - 16*x**2*z**3 + 4*x**2*z - 24*x*y**4*z**4 + 24*x*y**4*z**2 - 4*x*y**4 + 48*x*y**3*z**4 - 48*x*y**3*z**2 + 8*x*y**3 - 48*x*y**2*z**4 + 48*x*y**2*z**3 - 4*x*y**2 + 24*x*y*z**4 - 48*x*y*z**3 + 24*x*y*z**2 - 4*x*z**4 + 8*x*z**3 - 4*x*z**2 + 12*y**4*z**4 - 16*y**4*z**3 + 4*y**4*z + 5*y**4 - 24*y**3*z**4 + 32*y**3*z**3 - 8*y**3*z + 12*y**2*z**4 - 16*y**2*z**3 + 4*y**2*z
comp3 = -48*x**4*y**3*z**2 + 48*x**4*y**3*z - 8*x**4*y**3 + 72*x**4*y**2*z**2 - 72*x**4*y**2*z + 12*x**4*y**2 - 24*x**4*y*z**4 + 48*x**4*y*z**3 - 48*x**4*y*z**2 + 24*x**4*y*z - 4*x**4*y + 12*x**4*z**4 - 24*x**4*z**3 + 12*x**4*z**2 + 48*x**3*y**4*z**2 - 48*x**3*y**4*z + 8*x**3*y**4 + 48*x**3*y**2*z**4 - 96*x**3*y**2*z**3 - 48*x**3*y**2*z**2 + 96*x**3*y**2*z - 16*x**3*y**2 + 48*x**3*y*z**2 - 48*x**3*y*z + 8*x**3*y - 16*x**3*z**4 + 32*x**3*z**3 - 16*x**3*z**2 - 72*x**2*y**4*z**2 + 72*x**2*y**4*z - 12*x**2*y**4 - 48*x**2*y**3*z**4 + 96*x**2*y**3*z**3 + 48*x**2*y**3*z**2 - 96*x**2*y**3*z + 16*x**2*y**3 + 24*x**2*y*z**4 - 48*x**2*y*z**3 + 24*x**2*y*z - 4*x**2*y + 24*x*y**4*z**4 - 48*x*y**4*z**3 + 48*x*y**4*z**2 - 24*x*y**4*z + 4*x*y**4 - 48*x*y**3*z**2 + 48*x*y**3*z - 8*x*y**3 - 24*x*y**2*z**4 + 48*x*y**2*z**3 - 24*x*y**2*z + 4*x*y**2 + 4*x*z**4 - 8*x*z**3 + 4*x*z**2 - 12*y**4*z**4 + 24*y**4*z**3 - 12*y**4*z**2 + 16*y**3*z**4 - 32*y**3*z**3 + 16*y**3*z**2 - 4*y*z**4 + 8*y*z**3 - 4*y*z**2 + 5*z**4
f += (comp1*vx + comp2*vy + comp3*vz)*dx
f.Assemble()
uph = ng.GridFunction(X)

stage_ngs.pop()
timer_ngs.Stop()

if comm.rank == 0:
    print('Parallel ngsolve assembly complete')
    print('Converting matrix system to Petsc ...')

# Set up things to move to Petsc.
# n2p = ngsolve.ngs2petsc can transfer vec/mat between NGSolve and Petsc:

timer_trf.Start()
stage_trf.push()

psc_mat = n2p.CreatePETScMatrix(a.mat, X.FreeDofs())
vecmap = n2p.VectorMapping(a.mat.row_pardofs, X.FreeDofs())

## Get some Petsc vectors
psc_f, psc_up = psc_mat.createVecs()

psc_mat.setType('aijcusparse')
psc_f.setType('cuda')
psc_up.setType('cuda')

psc_f.setFromOptions()
psc_up.setFromOptions()
psc_mat.setFromOptions()

stage_trf.pop()
timer_trf.Stop()

# Set up KSP from psc = petsc4py.PETSc

timer_ksp.Start()
stage_ksp.push()

#TODO Check different solvers! Maybe in runtime options though
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

ksp.solve(psc_f, psc_up)

stage_ksp.pop()
timer_ksp.Stop()

# Convert from Petsc to NGSolve:

timer_trf.Start()
stage_trf.push()

vecmap.P2N(psc_up, uph.vec)

stage_trf.pop()
timer_trf.Stop()

# Compute error (in NGSolve)  since  exact solution known:
print('TEMP DONE!')
sys.exit()

timer_ngs.Start()
stage_ngs.push()

exact_p = x**5 + y**5 + z**5 - 0.5
#TODO exact_ux = x**5 + y**5 + z**5 - 0.5
#TODO exact_ux = x**5 + y**5 + z**5 - 0.5
#TODO exact_ux = x**5 + y**5 + z**5 - 0.5
#TODO decouple solution vector
#TODO CHECK FINAL STAGE

error = ng.Integrate((uph-exact)*(uph-exact), mesh)
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
