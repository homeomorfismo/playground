# CHECK https://link.springer.com/content/pdf/10.1007/s10915-015-0090-8.pdf

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
# Yes, we do. Maybe this is worth a contribution!
import ngsolve.ngs2petsc as n2p

# Should be useful for runtime options without exporting PETSC_OPTIONS
import sys
comm = MPI.COMM_WORLD

# Initialize PETSc 
if comm.rank == 0:
    print('Initializing PETSc...')

## PETSc Options
opt = psc.Options()
nref = opt.getInt('nref',0)
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
stage_ass = psc.Log.Stage('Assembling in PETSc')
stage_ISt = psc.Log.Stage('Re-indexing')
stage_ksp = psc.Log.Stage('PETSc solver')

# Generate Netgen mesh and distribute:
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

# Standard mixed set-up in NGSolve (parallel)
# FES: Taylor Hood P2-P1
stage_ngs.push()

V = ng.VectorH1(mesh,order=order_fes+1,dirichlet="top|bottom|right|left") 
Q = ng.H1(mesh,order=order_fes)

u,v = V.TnT()
p,q = Q.TnT()

a = ng.BilinearForm(V)
a += ng.InnerProduct(grad(u),grad(v))*dx
a.Assemble()

b = ng.BilinearForm(trialspace=V,testspace=Q)
b += (-1)*div(u)*q*dx
b.Assemble()

# Mass PC
mass_p = ng.BilinearForm(Q)
mass_p += p*q*dx
mass_p.Assemble()

#TODO Linear Forms
# fx = 4*ng.pi*ng.pi*ng.sin(2*ng.pi*y)
# fy = 4*ng.pi*ng.pi*ng.sin(2*ng.pi*x)*(-1 + 4*ng.cos(2*ng.pi*y))
f = ng.LinearForm(V)
f += ng.CF((4*ng.pi*ng.pi*ng.sin(2*ng.pi*y),4*ng.pi*ng.pi*ng.sin(2*ng.pi*x)*(-1 + 4*ng.cos(2*ng.pi*y))))*v*dx
f.Assemble()

g = ng.LinearForm(Q)
g.Assemble()

#TODO Grid Functions
gu = ng.GridFunction(V, name="vel")
gp = ng.GridFunction(Q, name="pre")

stage_ngs.pop()

# Transport to PETSc
stage_ISt.push()
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
### ngs2petsc.py - CreatePETScMatrix
## Free dofs
vel_freedof = np.flatnonzero(V.FreeDofs()).astype(psc.IntType)
pre_freedof = np.flatnonzero(Q.FreeDofs()).astype(psc.IntType)
#IS Sets
vel_islocfree = psc.IS().createBlock(indices=vel_freedof, bsize=eh)
pre_islocfree = psc.IS().createBlock(indices=pre_freedof, bsize=ew)

# print('V', nglob_vel,'. Q', nglob_pre)
# print('V', list(globalnums_vel), 'Q', list(globalnums_pre))

stage_ISt.pop()
## V, with N2P
stage_ass.push()
a_psc = n2p.CreatePETScMatrix(a.mat, V.FreeDofs())
stage_ass.pop()

stage_ISt.push()
vecmap_V = n2p.VectorMapping(a.mat.row_pardofs, V.FreeDofs())
stage_ISt.pop()

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
stage_ass.push()
b_locmat = b.mat.local_mat
stage_ass.pop()

stage_ISt.push()
eh, ew = b_locmat.entrysizes
b_val, b_col, b_ind = b_locmat.CSR()
b_ind = np.array(b_ind).astype(psc.IntType)
b_col = np.array(b_col).astype(psc.IntType)
stage_ISt.pop()

# Local B mat PETSc
stage_ass.push()
b_locmat_psc = psc.Mat().createAIJ(size=(eh*b_locmat.height, ew*b_locmat.width),
        csr=(b_ind,b_col,b_val), comm=MPI.COMM_SELF)
b_locmat_psc = b_locmat_psc.createSubMatrices(pre_islocfree,iscols=vel_islocfree)[0]

vel_pardof = b.mat.row_pardofs
comm = vel_pardof.comm.mpi4py
vel_globnums, vel_nglob = vel_pardof.EnumerateGlobally(V.FreeDofs())
vel_globnums = np.array(vel_globnums,dtype=psc.IntType)[V.FreeDofs()]
vel_lgmap = psc.LGMap().create(indices=vel_globnums, bsize=eh, comm=comm)

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

print(b_psc.getSize())
#TEMP TODO

#vecmap_Q = n2p.VectorMapping(b.mat.row_pardofs, Q.FreeDofs())

## PC for P
mass_p_psc = n2p.CreatePETScMatrix(mass_p.mat, Q.FreeDofs())
stage_ass.pop()
stage_ISt.push()
vecmap_mass = n2p.VectorMapping(mass_p.mat.row_pardofs, Q.FreeDofs())
stage_ISt.pop()

# Mat-Nest
stage_ass.push()
mats = [[a_psc,bT_psc],
        [b_psc,None]]

psc_mat = psc.Mat().create().createNest(mats)
psc_mat.assemble()
stage_ass.pop()
# Block size (1,1)

# Define fields
stage_ISt.push()
## Extract ISs from MatNest
ISs = psc_mat.getNestISs()
stage_ISt.pop()
## Define KSP
stage_ksp.push()
ksp = psc.KSP().create()
ksp.setOperators(psc_mat)
## Get PC, setup PC
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitIS(("vel",ISs[0][0]), ("pre",ISs[0][1]))
pc.setFieldSplitType(psc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(psc.PC.SchurFactType.FULL)
pc.setFieldSplitSchurPreType(psc.PC.SchurPreType.SELF)

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
# psc_mat.view()

# Runtime options
psc_mat.setFromOptions()
a_psc.setFromOptions()
b_psc.setFromOptions()
bT_psc.setFromOptions()
mass_p_psc.setFromOptions()
ksp.setFromOptions()
pc.setFromOptions()

psc_mat.setUp()
a_psc.setUp()
b_psc.setUp()
bT_psc.setUp()
mass_p_psc.setUp()
# THIS LINE CUDA
pc.setUp()
ksp.setUp()
stage_ksp.pop()

#TODO Solve
stage_ISt.push()
# TODO Create PETSc vectors

psc_up, psc_fg = psc_mat.createVecs()

psc_f = vecmap_V.N2P(f.vec)
psc_g = vecmap_mass.N2P(g.vec)

#psc_fg = psc.Vec().create().createNest([psc_f,psc_g])
psc_fg, is_fg = psc_fg.concatenate([psc_f, psc_g])

stage_ISt.pop()

stage_ksp.push()
psc_f.setFromOptions()
psc_g.setFromOptions()
psc_fg.setFromOptions()
psc_up.setFromOptions()

psc_f.setUp()
psc_g.setUp()
psc_fg.setUp()
psc_up.setUp()

ksp.solve(psc_fg, psc_up)
stage_ksp.pop()

stage_ISt.push()
psc_u = psc_up.getSubVector(ISs[0][0])
psc_p = psc_up.getSubVector(ISs[0][1])

vecmap_V.P2N(psc_u,gu.vec) 
vecmap_mass.P2N(psc_p,gp.vec)
stage_ISt.pop()

# print(comm.rank,'Vel',gu.vec)
# print(comm.rank,'Pre',gp.vec)
stage_ngs.push()
# ux = 2*ng.sin(ng.pi*x)*ng.sin(ng.pi*x)*ng.sin(ng.pi*y)
# uy = (-2)*ng.sin(ng.pi*x)*ng.sin(ng.pi*y)*ng.sin(ng.pi*y)
cf_u = ng.CF( (2*ng.sin(ng.pi*x)*ng.sin(ng.pi*x)*ng.sin(ng.pi*y),(-2)*ng.sin(ng.pi*x)*ng.sin(ng.pi*y)*ng.sin(ng.pi*y)) )
# p = 4*ng.pi*ng.sin(2*ng.pi*x)*sin(2*ng.pi*y)
cf_p = ng.CF( 4*ng.pi*ng.sin(2*ng.pi*x)*ng.sin(2*ng.pi*y) ) 

error_vel = ng.Integrate(ng.InnerProduct(cf_u - gu,cf_u - gu), mesh)
error_pre = ng.Integrate((cf_p - gp)*(cf_p - gp), mesh)
if comm.rank == 0:
    print('L2-error vel', ng.sqrt(error_vel))
    print('L2-error pre', ng.sqrt(error_pre))
stage_ngs.pop()
