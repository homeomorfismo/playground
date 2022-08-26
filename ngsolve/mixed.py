import ngsolve as ng
import netgen
from mpi4py import MPI
import petsc4py
import petsc4py.PETSc as psc

from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, grad, dx
from ngsolve.ngstd import Timer

from tabulate import tabulate

# Not sure if we requiere ngs2petsc
import ngsolve.ngs2petsc as n2p

########## --- --- ##########

# Should be useful for runtime options without exporting PETSC_OPTIONS
import sys
comm = MPI.COMM_WORLD

# Initialize PETSc 
if comm.rank == 0:
    print('Initializing PETSc...')

## PETSc Options
opt = psc.Options()
nref = opt.getInt('nref',1)
hcoarse = opt.getReal('h',0.5)
order_fes = opt.getInt('ord',1)
if comm.rank == 0:
    print('--- Ini user-defined values ---')
    print('nref', nref)
    print('hcoarse', hcoarse)
    print('ord', order_fes)
    print('--- End user-defined values ---')
# sys.exit()

## Define PETSC stages (Profiling)
stage_msh = psc.Log.Stage('Meshing')
stage_ngs = psc.Log.Stage('Setting in NGS')
stage_trf = psc.Log.Stage('Transfer ngs2petsc')
stage_ksp = psc.Log.Stage('PETSc solver')

## Define ngstd-Timers (Profiling/timing)
timer_msh = Timer('Meshing')
timer_ngs = Timer('Setting in NGS')
timer_trf = Timer('Transfer ngs2petsc')
timer_ksp = Timer('PETSc solver')

# Generate Netgen mesh and distribute:
timer_msh.Start()
stage_msh.push()

comm.Barrier()
if comm.rank == 0:
    print('Generating Mesh ...')
    ngmesh = unit_square.GenerateMesh(maxh=hcoarse)
    for i in range(nref):
        ngmesh.Refine()
    ngmesh = ngmesh.Distribute(comm)
else:
    print('Rank [',comm.rank,'] receiving mesh...')
    ngmesh = netgen.meshing.Mesh.Receive(comm)
mesh=Mesh(ngmesh)
comm.Barrier()

stage_msh.pop()
timer_msh.Stop()

# Standard mixed set-up in NGSolve (parallel)
timer_ngs.Start()
stage_ngs.push()

#### TODO Check notes. Check DoFs here!

# FES: RT and Disc Pol
V = ng.HDiv(mesh,order=order_fes+1, RT=True,dirichlet="top|bottom|right|left") 
P = ng.L2(mesh,order=order_fes)
VP = V*P

#TODO TEMP PRINT DOFS
print('Global DoF Velocity space: ', V.ndofglobal,'. Local (shared) DoFs from rank [', comm.rank, ']: ', V.ndof,'.')
print('Global DoF Pressure space: ', P.ndofglobal,'. Local (shared) DoFs from rank [', comm.rank, ']: ', P.ndof,'.')
print('Global DoF Compound space: ', VP.ndofglobal,'. Local (shared) DoFs form rank [', comm.rank, ']: ', VP.ndof,'.')
comm.Barrier()

pardof_vel = V.ParallelDofs()
pardof_pre = P.ParallelDofs()
pardof_com = VP.ParallelDofs()

#TODO TEMP PRINT PAR_DOFS
#TODO OBS pardof is a NGSolve object
#TODO OBS FES.ParallelDofs().ndoflocal = FES.ndof
#TODO OBS Enumeration from the compound space is different
print('Local ParallelDofs V from rank [', comm.rank, ']:', pardof_vel.ndoflocal)
print('Local ParallelDofs P from rank [', comm.rank, ']:', pardof_pre.ndoflocal)
print('Local ParallelDofs VP from rank [', comm.rank, ']:', pardof_com.ndoflocal)

#TODO TEMP PRINT ENUMERATION FROM PARDOFS
globalnums_vel, nglob_vel = pardof_vel.EnumerateGlobally()
globalnums_pre, nglob_pre = pardof_pre.EnumerateGlobally()
globalnums_com, nglob_com = pardof_com.EnumerateGlobally()

print('Local Enumerations V from rank [', comm.rank, ']:', globalnums_vel, 'Global en V',nglob_vel)
print('Local Enumerations P from rank [', comm.rank, ']:', globalnums_pre, 'Global en P',nglob_pre)
print('Local Enumerations VP from rank [', comm.rank, ']:', globalnums_com, 'Global en VP',nglob_com)

#TODO TEMP EXIT
sys.exit()

#TODO Copy the right bilinear forms
u, v = V.TnT()
a = ng.BilinearForm(grad(u)*grad(v)*dx).Assemble()

f = ng.LinearForm(V)
f += 32 * (y*(1-y)+x*(1-x)) * v * dx
f.Assemble()
uh = ng.GridFunction(V)

stage_ngs.pop()
timer_ngs.Stop()
