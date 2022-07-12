# Modules in PETSc

* Index sets `IS`
* Vectors `Vec`, matrices `Mat`
* Mesh data structures and interaction with vectors and matrices `DM`
* Krylov subspace methods `KSP`
* Preconditioners `PC`
* Nonlinear solvers `SNES`
* timesteppers `TS`

## MPI

PETSc employs the Message Passing Interface.

`mpiexec -n num_processes ./program_name <options>`

# Writing code

Call: `PerscInitialize(int *argc, char ***argv, char *file, char *help)`

End: `PetscFinalize();`


