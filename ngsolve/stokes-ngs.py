# CHECK https://link.springer.com/content/pdf/10.1007/s10915-015-0090-8.pdf

import ngsolve as ng
import netgen
from mpi4py import MPI
# import petsc4py
# import petsc4py.PETSc as psc
import numpy as np
from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
from ngsolve import Mesh, x, y, grad, dx, div
from ngsolve.ngstd import Timer

from tabulate import tabulate

import ngsolve.ngs2petsc as n2p

import sys
comm = MPI.COMM_WORLD

nref = 0
hcoarse = 0.5
order_fes = 1
flg_par = True

if flg_par:
    if comm.rank == 0:
        print('Generating Mesh ...')
        ngmesh = unit_square.GenerateMesh(maxh=hcoarse)
        for i in range(nref):
            ngmesh.Refine()
        ngmesh = ngmesh.Distribute(comm)
    else:
        print('Rank [',comm.rank,'] receiving mesh...')
        ngmesh = netgen.meshing.Mesh.Receive(comm)
else:
    ngmesh = unit_square.GenerateMesh(maxh=hcoarse)
mesh=Mesh(ngmesh)
comm.Barrier()

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

print('Free Dofs V', len(V.FreeDofs()), ' - ', V.FreeDofs())
print('Free Dofs Q', len(Q.FreeDofs()), ' - ', Q.FreeDofs())
print('---')

if not flg_par:
    rows, cols, vals = a.mat.COO()
    import scipy.sparse as sp
    A = sp.csc_matrix((vals,(rows,cols)))
    print(A.shape)
    print('---')
    rows, cols, vals = b.mat.COO()
    B = sp.csc_matrix((vals,(rows,cols)))
    print(B.shape)
    from scipy.io import savemat, loadmat
    savemat('tempA',{'A':A})
    savemat('tempB',{'B':B})


#These are parallel matrices if using parallel mesh

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

gu = ng.GridFunction(V, name="vel")
gp = ng.GridFunction(Q, name="pre")

K = ng.BlockMatrix( [[a.mat, b.mat.T], [b.mat, None]] )
C = ng.BlockMatrix( [[a.mat.Inverse(V.FreeDofs()), None], [None, mass_p.mat.Inverse()]] )

rhs = ng.BlockVector( [f.vec, g.vec] )
sol = ng.BlockVector( [gu.vec, gp.vec] )

ng.solvers.MinRes( mat=K, pre=C, rhs=rhs, sol=sol, initialize=False)

print(comm.rank,'Vel',gu.vec.size)
print(comm.rank,'Pre',gp.vec.size)

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
