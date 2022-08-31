# How to use the cluster

Check [this list of tricks](tricks.md) and [these instructions](howto.md) to feel comfortable enough with the installation process.

1. Allocate a node with SLURM in interactive mode.
    - Check the nodes and partition with `sinfo` and `squeue`. To check a specific partition (e.g. `himem` a.k.a. V100 GPU, `gpu` a.k.a. RTX A5000 or A40) use `-p <partition_name>`. To filter by a user name (PSU ODIN), use `-u <user_name>`.
    - Allocate an interactive job. Do `salloc -J <job_name> -p <partition_name> -n <number_cores> -N <number_nodes> --immediate=<time_secs>`
    - Check job ID and *assigned node* with `squeue -u <your_user>`.
    - SSH into the assigned node `ssh <node_name>`
    - Check the status of the node. Do `nvidia-smi` and `htop`.

2. Load modules. For simplicity, load NGSolve module. The module manager should load all required modules (OpenMPI, GCC,...).
```
module purge
module load ngsolve/parallel/6.2.2203-openmpi_4.1.2
```

2. *[Suggested]* Create Python Virtual Enviroment.
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
mkdir $HOME/env
python3 -m venv $HOME/env/<name-env>
```
You can activate the enviroment with `source $HOME/env/<name-env>/bin/activate`. You can add an alias to your `.bashrc` file. Deactivate it with `deactivate`.

3. Install python prerequisites. **Do NOT install petsc4py**, it will install a copy of PETSc we don't want.
```
pip3 install wheel 
pip3 install --upgrade wheel
pip3 install scipy mpi4py cython numpy
```

4. Install PETSc. Check how to [download](https://petsc.org/release/download/) and [install](https://petsc.org/release/install/) PETSc. I will employ the release version; I tried the main branch but it didn't worked for me.
    - Optimized version.
```
git clone -b release https://gitlab.com/petsc/petsc.git PETSc
cd $HOME/PETSc
export PETSC_ARCH=arch-cuda-opt
./configure --with-mpi=1 --CPPFLAGS=-O3 --CFLAGS=-O3 --CXXPPFLAGS=-O3 --CXXFLAGS=-O3 --CXX_CXXFLAGS=-O3 --FPPFLAGS=-O3 --FFLAGS=-O3 --CUDAFLAGS=-O3 --with-cuda-dir=/usr/local/cuda-11.7 --with-debugging=0 --with-petsc4py=yes --download-triangle --download-hypre=1 --download-hwloc=1 --with-cuda --with-cudac=nvcc
```
Then follow the instructions.
    - Debug version.
```
export PETSC_ARCH=arch-cuda-debug
./configure --with-mpi=1 --CPPFLAGS="-O0 -g" --CFLAGS="-O0 -g" --CXXPPFLAGS="-O0 -g" --CXXFLAGS="-O0 -g" --CXX_CXXFLAGS="-O0 -g" --FPPFLAGS="-O0 -g" --FFLAGS="-O0 -g" --CUDAFLAGS="-O0 -g" --with-cuda-dir=/usr/local/cuda-11.7 --with-debugging=1 --with-petsc4py=yes --download-triangle --download-hypre=1 --download-hwloc=1 --with-cuda --with-cudac=nvcc
```
Then follow the instructions.
Notice you will have two copies of petsc4py. You can define different aliases for appending the corresponding petsc4py to `PYTHONPATH`.

5. *[Suggested]* Append your PETSc options.
```
export PETSC_OPTIONS="-logview -mat_type aijcusparse -vec_type cuda -ksp_view -snes_view -ksp_monitor -snes_monitor -ksp_converged_reason -snes_converged_reason ..."
```
