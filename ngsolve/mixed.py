import ngsolve as ng
import netgen
from mpi4py import MPI
import petsc4py
import petsc4py.PETSc as psc
import numpy as np
from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, grad, dx, div
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
# timer_ngs.Start()
# stage_ngs.push()
# stage_ngs.pop()
# timer_ngs.Stop()
#### TODO Check notes. Check DoFs here!

# FES: RT and Disc Pol
V = ng.HDiv(mesh,order=order_fes+1, RT=True,dirichlet="top|bottom|right|left") 
P = ng.L2(mesh,order=order_fes)
VP = V*P

pardof_vel = V.ParallelDofs()
pardof_pre = P.ParallelDofs()
pardof_com = VP.ParallelDofs()

globalnums_vel, nglob_vel = pardof_vel.EnumerateGlobally()
globalnums_pre, nglob_pre = pardof_pre.EnumerateGlobally()
globalnums_com, nglob_com = pardof_com.EnumerateGlobally()

#TODO TEMP PRINT PAR_DOFS
#TODO OBS pardof is a NGSolve object
#TODO OBS FES.ParallelDofs().ndoflocal = FES.ndof
#TODO OBS Enumeration from the compound space is different; I don't know if that
#         can be made compatible.

# TEMP PRINT GLOBAL & LOCAL DOFS
# print('Global DoF Velocity space: ', V.ndofglobal,'. Local (shared) DoFs from rank [', comm.rank, ']: ', V.ndof,'.')
# print('Global DoF Pressure space: ', P.ndofglobal,'. Local (shared) DoFs from rank [', comm.rank, ']: ', P.ndof,'.')
# print('Global DoF Compound space: ', VP.ndofglobal,'. Local (shared) DoFs form rank [', comm.rank, ']: ', VP.ndof,'.')
# comm.Barrier()

# TEMP PRINT LOCAL DOFS WITH PAR_DOFS
# print('Local ParallelDofs V from rank [', comm.rank, ']:', pardof_vel.ndoflocal)
# print('Local ParallelDofs P from rank [', comm.rank, ']:', pardof_pre.ndoflocal)
# print('Local ParallelDofs VP from rank [', comm.rank, ']:', pardof_com.ndoflocal)

# TEMP PRINT ENUMERATION FROM PARDOFS
# print('Local Enumerations V from rank [', comm.rank, ']:', globalnums_vel, 'Global en V',nglob_vel)
# print('Local Enumerations P from rank [', comm.rank, ']:', globalnums_pre, 'Global en P',nglob_pre)
# print('Local Enumerations VP from rank [', comm.rank, ']:', globalnums_com, 'Global en VP',nglob_com)

# Compound form
u, p = VP.TrialFunction()
v, q = VP.TestFunction()
c = ng.BilinearForm(VP)
c += (u*v - p*div(v) - div(u)*q)*dx
c.Assemble()

#TODO Set RHS
# func = 32 * (y*(1-y)+x*(1-x)) 
f = ng.LinearForm(VP)
f += -32*(y*(1-y)+x*(1-x))*q*dx # + g*v*dx
f.Assemble()

uph = ng.GridFunction(VP)

# Create Mat
c_locmat = c.mat.local_mat
c_val, c_col, c_ind = c_locmat.CSR()
c_ind = np.array(c_ind, dtype='int32')
c_psc_loc = psc.Mat().createAIJ(size=(c_locmat.height, c_locmat.width),csr=(c_ind,c_col,c_val),comm=MPI.COMM_SELF)

# IS
iset = psc.IS().createGeneral(indices=globalnums_com, comm=comm) # nglob_com
lgmap = psc.LGMap().createIS(iset)

# Global mat
c_psc = psc.Mat().createPython(size=nglob_com, comm=comm)
c_psc.setType(psc.Mat.Type.IS)
c_psc.setLGMap(lgmap)
c_psc.setISLocalMat(c_psc_loc)
c_psc.assemble()
c_psc.convert("mpiaij")
c_psc.setFromOptions()

# RHS + vectors ?
f.vec.Cumulate()
v1, v2 = c_psc.createVecs()
v2_loc = v2.getSubVector(iset)
v2_loc.getArray()[:] = f.vec.FV()
v2.restoreSubVector(iset,v2_loc)

# KSP Solver
ksp = psc.KSP()
ksp.create()
ksp.setOperators(c_psc)
ksp.setType(psc.KSP.Type.CG)
ksp.setNormType(psc.KSP.NormType.NORM_NATURAL)
ksp.getPC().setType("gamg")
ksp.setTolerances(rtol=1e-6, atol=0, divtol=1e16, max_it=400)
ksp.solve(v2,v1)

# Recover sol
v1_loc = v1.getSubVector(iset)
for i in range(len(uph.vec)):
    uph.vec.FV()[i] = v1_loc.getArray()[i]

#TODO TEMP EXIT
sys.exit()

# Convert matrix (Compound)
psc_c = n2p.CreatePETScMatrix(c.mat, VP.FreeDofs())
vecmap = n2p.VectorMapping(c.mat.row_pardofs, VP.FreeDofs())
psc_f, psc_up = psc_c.createVecs()
# Define type
psc_c.setType('aijcusparse')
psc_f.setType('cuda')
psc_up.setType('cuda')
psc_f.setFromOptions()
psc_up.setFromOptions()
psc_c.setFromOptions()
# Define KSP
ksp = psc.KSP()
ksp.create()
ksp.setOperators(psc_c)
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
# Solve KSP
vecmap.N2P(f.vec, psc_f)
ksp.solve(psc_f, psc_up)

vecmap.P2N(psc_up, uph.vec)

exact = 16*x*(1-x)*y*(1-y)
error = ng.Integrate((uph.components[1]-exact)*(uph.components[1]-exact), mesh)
if comm.rank == 0:
    print('L2-error', error)

#TODO TEMP EXIT
sys.exit()

######
if comm.rank == 0:
    print('Parallel ngsolve assembly complete')
    print('Converting matrix system to Petsc ...')

# Serial
u, v = V.TnT()
uph = ng.GridFunction(V)
