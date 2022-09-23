# CHECK https://link.springer.com/content/pdf/10.1007/s10915-015-0090-8.pdf

from mpi4py import MPI
import petsc4py
import petsc4py.PETSc as psc

import netgen
import ngsolve as ng
from netgen.geom2d import unit_square
from ngsolve import Mesh, x, y, grad, dx, div
from ngsolve.ngstd import Timer
import ngsolve.ngs2petsc as n2p

import numpy as np
from tabulate import tabulate
import sys

comm = MPI.COMM_WORLD
if comm.rank == 0:
    print('Initializing PETSc...')

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

#TODO
stage_msh = psc.Log.Stage('Meshing')
stage_ngs = psc.Log.Stage('Setting in NGS')
stage_ass = psc.Log.Stage('Assembling in PETSc')
stage_ISt = psc.Log.Stage('Re-indexing')
stage_ksp = psc.Log.Stage('PETSc solver')

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

mass_pre = ng.BilinearForm(Q)
mass_pre += p*q*dx
mass_pre.Assemble()

# fx = 4*ng.pi*ng.pi*ng.sin(2*ng.pi*y)
# fy = 4*ng.pi*ng.pi*ng.sin(2*ng.pi*x)*(-1 + 4*ng.cos(2*ng.pi*y))
f = ng.LinearForm(V)
f += ng.CF((4*ng.pi*ng.pi*ng.sin(2*ng.pi*y),4*ng.pi*ng.pi*ng.sin(2*ng.pi*x)*(-1 + 4*ng.cos(2*ng.pi*y))))*v*dx
f.Assemble()

g = ng.LinearForm(Q)
g.Assemble()

gu = ng.GridFunction(V, name="vel")
gp = ng.GridFunction(Q, name="pre")


a_psc = n2p.CreatePETScMatrix(a.mat, V.FreeDofs())
vecmap_V = n2p.VectorMapping(a.mat.row_pardofs, V.FreeDofs())

mass_pre_psc = n2p.CreatePETScMatrix(mass_pre.mat, Q.FreeDofs())
vecmap_mass = n2p.VectorMapping(mass_pre.mat.row_pardofs, Q.FreeDofs())

b_locmat = b.mat.local_mat
eh, ew = b_locmat.entrysizes

b_val, b_col, b_ind = b_locmat.CSR()
b_ind = np.array(b_ind).astype(psc.IntType)
b_col = np.array(b_col).astype(psc.IntType)

vel_locfree = np.flatnonzero(V.FreeDofs()).astype(psc.IntType)
vel_locfree_is = psc.IS().createBlock(indices=vel_locfree, bsize=ew)

pre_locfree = np.flatnonzero(Q.FreeDofs()).astype(psc.IntType)
pre_locfree_is = psc.IS().createBlock(indices=pre_locfree, bsize=eh)

b_locmat_psc = psc.Mat().createAIJ(size=(eh*b_locmat.height, ew*b_locmat.width), csr=(b_ind,b_col,b_val), comm=MPI.COMM_SELF)
b_locmat_psc = b_locmat_psc.createSubMatrices(pre_locfree_is,iscols=vel_locfree_is)[0]

vel_pardof = b.mat.row_pardofs
pre_pardof = b.mat.T.row_pardofs

#comm = b_XXX_pardofs.comm.mpi4py

vel_globnums, vel_nglob = vel_pardof.EnumerateGlobally(V.FreeDofs())
vel_globnums = np.array(vel_globnums, dtype=psc.IntType)[V.FreeDofs()]
pre_globnums, pre_nglob = pre_pardof.EnumerateGlobally(Q.FreeDofs())
pre_globnums = np.array(pre_globnums, dtype=psc.IntType)[Q.FreeDofs()]

vel_lgmap = psc.LGMap().create(indices=vel_globnums, bsize=ew, comm=comm)
pre_lgmap = psc.LGMap().create(indices=pre_globnums, bsize=eh, comm=comm)

b_psc = psc.Mat().create(comm=comm)
b_psc.setSizes(size=[pre_nglob*eh,vel_nglob*ew], bsize=[eh,ew])
b_psc.setType(psc.Mat.Type.IS)
b_psc.setLGMap(pre_lgmap, cmap=vel_lgmap)
b_psc.setISLocalMat(b_locmat_psc)
b_psc.assemble()
b_psc.convert("mpiaij")

bT_psc = b_psc.duplicate(copy=True)
bT_psc.transpose()
bT_psc.assemble()

#vecmap_Q = n2p.VectorMapping(b.mat.row_pardofs, Q.FreeDofs())

mats = [[a_psc,bT_psc],
        [b_psc,None]]

psc_mat = psc.Mat().create().createNest(mats)
ISs = psc_mat.getNestISs()
psc_mat.convert("mpiaij")
psc_mat.assemble()

ksp = psc.KSP().create()
ksp.setOperators(psc_mat)
ksp.setTolerances(rtol=1e-14, atol=0, divtol=1e16, max_it=500)

pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitIS(("vel",ISs[0][0]), ("pre",ISs[0][1]))
pc.setFieldSplitType(psc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(psc.PC.SchurFactType.FULL)
pc.setFieldSplitSchurPreType(psc.PC.SchurPreType.SELF,pre=mass_pre_psc)

psc_up, psc_fg = psc_mat.createVecs()

psc_f = vecmap_V.N2P(f.vec)
psc_g = vecmap_mass.N2P(g.vec)
psc_fg, is_fg = psc_fg.concatenate([psc_f, psc_g])

psc_mat.setFromOptions()
a_psc.setFromOptions()
b_psc.setFromOptions()
bT_psc.setFromOptions()
mass_pre_psc.setFromOptions()
ksp.setFromOptions()
pc.setFromOptions()
psc_f.setFromOptions()
psc_g.setFromOptions()
psc_fg.setFromOptions()
psc_up.setFromOptions()

psc_mat.setUp()
a_psc.setUp()
b_psc.setUp()
bT_psc.setUp()
mass_pre_psc.setUp()
pc.setUp()
ksp.setUp()
psc_f.setUp()
psc_g.setUp()
psc_fg.setUp()
psc_up.setUp()


ksp.solve(psc_fg, psc_up)

#TEMP
psc_up.view()

psc_u = psc_up.getSubVector(ISs[0][0])
psc_p = psc_up.getSubVector(ISs[0][1])

vecmap_V.P2N(psc_u,gu.vec) 
vecmap_mass.P2N(psc_p,gp.vec)


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
