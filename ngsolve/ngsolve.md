# Programs

- `lapl.py` parallel vestion of NGSolve displaying the NGSolve-PETSc interface with `ngs2petsc`.
- `lapl-refine.py` is a refined version of the above. With more PETSc stages and timers for profiling purpuses. **Fixed:** Distribution of DoFs. Let rank 0 to refine, then distribute.
- `stokes-refine.py` aims to solve a Stokes problem with similar structure to `lapl-refine.py`. *In progress.*
- `mixed.py` solves the Laplace eq. with a mixed method. Figuring out how to use PETSc fieldsplit. *In Progress.*

## TODO
- Remove refine label. `lapl.py` to `OLD.lapl.py`. Analog for other methods.
- To finish serial version of the mixed method.
- To find a good example for setting up `mixed.py`.
- ~~To do the no-PETSc version of `stokes.py`.~~

