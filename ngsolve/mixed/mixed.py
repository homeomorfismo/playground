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

import ngsolve.ngs2petsc as n2p

import sys
comm = MPI.COMM_WORLD

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

## Define PETSC stages (Profiling)
#OBS PETSc stages take avr time among all ranks! Even if there is no use of PETSc
#    thus, no need of netgen Timer

stage_msh = psc.Log.Stage('Meshing')
stage_ngs = psc.Log.Stage('Setting in NGS')
stage_trf = psc.Log.Stage('Transfer ngs2petsc')
stage_ksp = psc.Log.Stage('PETSc solver')

# Generate Netgen mesh and distribute:
#timer_msh.Start()
stage_msh.push()

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

V = ng.HDiv(mesh,order=order_fes+1,RT=True) 
Q = ng.L2(mesh,order=order_fes)

u,v = V.TnT()
p,q = Q.TnT()

a = ng.BilinearForm(V)
a += ng.InnerProduct(u,v)*dx
a.Assemble()

b = ng.BilinearForm(trialspace=V,testspace=Q)
b += div(u)*q*dx
b.Assemble()

f = ng.LinearForm(Q)
# f += 
f.Assemble()

g = ng.LinearForm(Q)
g += ng.CF((0,x-0.5))*v*dx
g.Assemble()

gu = ng.GridFunction(V, name="grad")
gq = ng.GridFunction(Q, name="solution")

# Transport to PETSc
eh, ew = b.mat.local_mat.entrysizes
## Recover Parallel DoFs
pardof_vel = V.ParallelDofs()
pardof_pre = Q.ParallelDofs()
globalnums_vel, nglob_vel = pardof_vel.EnumerateGlobally(V.FreeDofs())
globalnums_pre, nglob_pre = pardof_pre.EnumerateGlobally(Q.FreeDofs())
## Remove non-free DoFs
globalnums_vel = np.array(globalnums_vel, dtype=psc.IntType)[V.FreeDofs()]
globalnums_pre = np.array(globalnums_pre, dtype=psc.IntType)[Q.FreeDofs()]
## Local 2 Global maps
pre_set = psc.IS().createGeneral(indices=globalnums_pre, comm=comm)
vel_set = psc.IS().createGeneral(indices=globalnums_vel, comm=comm)
pre_lgmap = psc.LGMap().createIS(pre_set)
vel_lgmap = psc.LGMap().createIS(vel_set)
## Free dofs
vel_freedof = np.flatnonzero(V.FreeDofs()).astype(psc.IntType)
pre_freedof = np.flatnonzero(Q.FreeDofs()).astype(psc.IntType)
#IS Sets
vel_islocfree = psc.IS().createBlock(indices=vel_freedof, bsize=eh)
pre_islocfree = psc.IS().createBlock(indices=pre_freedof, bsize=ew)

## V, with N2P
a_psc = n2p.CreatePETScMatrix(a.mat, V.FreeDofs())
vecmap_V = n2p.VectorMapping(a.mat.row_pardofs, V.FreeDofs())

## handmade
# Local A mat NGSolve format
#a_locmat = a.mat.local_mat
#a_val, a_col, a_ind = a_locmat.CSR()
#a_ind = np.array(a_ind, dtype='int32')
# Local A mat PETSc
# a_locmat_psc = psc.Mat().createAIJ(size=(a_locmat.height, a_locmat.width), csr=(a_ind,a_col,a_val), comm=MPI.COMM_SELF)
# a_psc_2 = psc.Mat().createPython(size=nglob_vel, comm=comm)
# a_psc_2.setType(psc.Mat.Type.IS)
# a_psc_2.setLGMap(vel_lgmap)
# a_psc_2.setISLocalMat(a_locmat_psc)
# a_psc_2.assemble()
# a_psc_2.convert("mpiaij");

## Q, hand-made
# Local B mat NGSolve format
b_locmat = b.mat.local_mat

b_val, b_col, b_ind = b_locmat.CSR()
b_ind = np.array(b_ind).astype(psc.IntType)
b_col = np.array(b_col).astype(psc.IntType)

# Local B mat PETSc
b_locmat_psc = psc.Mat().createAIJ(size=(eh*b_locmat.height, ew*b_locmat.width),
        csr=(b_ind,b_col,b_val), comm=MPI.COMM_SELF)
b_locmat_psc = b_locmat_psc.createSubMatrices(pre_islocfree,iscols=vel_islocfree)[0]

vel_pardof = b.mat.row_pardofs
comm = vel_pardof.comm.mpi4py
vel_globnums, vel_nglob = vel_pardof.EnumerateGlobally(V.FreeDofs())
vel_globnums = np.array(vel_globnums,dtype=psc.IntType)[V.FreeDofs()]
vel_lgmap = psc.LGMap().create(indices=vel_globnums, bsize=eh, comm=comm)

#b_psc = psc.Mat().createPython(size=[nglob_pre,nglob_vel], comm=comm)
b_psc = psc.Mat().create(comm=comm)
b_psc.setSizes(size=[nglob_pre,nglob_vel],bsize=eh)

b_psc.setType(psc.Mat.Type.IS)
b_psc.setLGMap(pre_lgmap,cmap=vel_lgmap)
b_psc.setISLocalMat(b_locmat_psc)
b_psc.assemble()
b_psc.convert("mpiaij")

bT_psc = b_psc.duplicate(copy=True)
bT_psc.transpose()
bT_psc.assemble()

#vecmap_Q = n2p.VectorMapping(b.mat.row_pardofs, Q.FreeDofs())

####DEBUG

## PC for P
# mass_p_psc = n2p.CreatePETScMatrix(mass_p.mat, Q.FreeDofs())
# vecmap_mass = n2p.VectorMapping(mass_p.mat.row_pardofs, Q.FreeDofs())

# Mat-Nest
mats = [[a_psc,bT_psc],
        [b_psc,None]]

psc_mat = psc.Mat().create().createNest(mats)
psc_mat.assemble()

# print(a_psc.getSize())
# print(b_psc.getSize())
# print(psc_mat.getSize())
# sys.exit()

# Define fields
## Extract ISs from MatNest
ISs = psc_mat.getNestISs()
## Define KSP
ksp = psc.KSP().create()
ksp.setOperators(psc_mat)
## Get PC, setup PC
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitIS(("vel",ISs[0][0]), ("pre",ISs[0][1]))
pc.setFieldSplitType(psc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(psc.PC.SchurFactType.FULL)
pc.setFieldSplitSchurPreType(psc.PC.SchurPreType.SELF)
## OPTIONS ARE
#  ADDITIVE, MULTIPLICATIVE, SYMMETRIC_MULTIPLICATIVE, SPECIAL, SCHUR
#  DIAG, LOWER, UPPER, FULL
#  SELF, SELFP, A11, USER, FULL


# ISs[0][0].view()
# ISs[0][1].view()
# comm.Barrier()
# ISs[1][0].view()
# ISs[1][1].view()

##TODO Get SubKSP
# subksp = pc.getFieldSplitSubKSP()
# aa, pp = subksp[0].getOperators()
# subksp[0].setOperators(A=aa, P=mass_p_psc)

# Configure KSP & PC
ksp.setTolerances(rtol=1e-14, atol=0, divtol=1e16, max_it=500)

psc_mat.convert("mpiaij")

# Runtime options
psc_mat.setFromOptions()
a_psc.setFromOptions()
b_psc.setFromOptions()
bT_psc.setFromOptions()
# mass_p_psc.setFromOptions()
ksp.setFromOptions()
pc.setFromOptions()

psc_mat.setUp()
a_psc.setUp()
b_psc.setUp()
bT_psc.setUp()
# mass_p_psc.setUp()
# THIS LINE CUDA
pc.setUp()
ksp.setUp()

# TODO Create PETSc vectors

psc_up, psc_fg = psc_mat.createVecs()

psc_f = vecmap_V.N2P(f.vec)
psc_g = vecmap_mass.N2P(g.vec)

#psc_fg = psc.Vec().create().createNest([psc_f,psc_g])
#psc_fg.concatenate([psc_f, psc_g])
psc_fg, is_fg = psc_fg.concatenate([psc_f, psc_g])

psc_f.setFromOptions()
psc_g.setFromOptions()
psc_fg.setFromOptions()
psc_up.setFromOptions()

psc_f.setUp()
psc_g.setUp()
psc_fg.setUp()
psc_up.setUp()

ksp.solve(psc_fg, psc_up)

psc_u = psc_up.getSubVector(ISs[0][0])
psc_p = psc_up.getSubVector(ISs[0][1])

vecmap_V.P2N(psc_u,gu.vec) 
vecmap_mass.P2N(psc_p,gp.vec)

# exact = 16*x*(1-x)*y*(1-y)
# error = ng.Integrate((uh-exact)*(uh-exact), mesh)
# if comm.rank == 0:
    # print('L2-error', error)
