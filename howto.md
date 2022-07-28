---
How to install all the softwares I need?
---

1. Install NGSolve. See [here](https://docu.ngsolve.org/nightly/install/installlinux.html).

2. Install PETSc. See [here](https://petsc.org/release/install/).

	- To install PETSc with NGSolve (MPI-purposes), you must use the same MPI implementation. 
	- To install PETSc with Firedrake, you must append
	```
	--download-pastix --download-chaco --download-netcdf --download-metis --download-hdf5 --download-hwloc --download-hypre --download-ml --download-ptscotch --download-eigen=/home/gpin2/Firedrake/firedrake/src/eigen-3.3.3.tgz --download-mpich --with-zlib --download-scalapack --with-fortran-bindings=0 --with-debugging=0 --download-cmake --download-mumps --download-bison --with-shared-libraries=1 --download-superlu_dist --download-pnetcdf --with-cxx-dialect=C++11 --download-suitesparse --with-c2html=0
	```

3. Install Firedrake. See [here](https://www.firedrakeproject.org/download.html).


