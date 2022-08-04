---
How to install all the softwares I need?
---

1. Install NGSolve. See [here](https://docu.ngsolve.org/nightly/install/installlinux.html). You might want to build on Linux locally.

    - Install **all** prerequisites.
    ```
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get -y install python3 python3-distutils python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev cmake git g++ libglu1-mesa-dev liblapacke-dev xorg-dev
    ```
    - Define `BASEDIR` with `export BASEDIR=~/senseful-name-here`.
    - Create `BASEDIR` with `mkdir -p $BASEDIR`.
    - Change directory and clone `NGSolve` source.
    ```
    cd $BASEDIR
    git clone https://github.com/NGSolve/ngsolve.git ngsolve-src
    ```
    - Fetch dependencies.
    ```
    cd $BASEDIR/ngsolve-src
    git submodule update --init --recursive
    ```
    - Build from souce! You might want to make sure you have `ccmake` if you want a graphical interface for configuring your install. Check [all the `cmake` options here](https://docu.ngsolve.org/latest/install/cmakeoptions.html). 
        * Install OCC.
        ``` 
        sudo apt-add-repository universe
        sudo apt-get update
        sudo apt-get install libocct-data-exchange-dev libocct-draw-dev occt-misc
        ```
        * Install MPI?
    - Configure cmake.
    ```
    cd $BASEDIR/ngsolve-build
    cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install ${BASEDIR}/ngsolve-src
    ```
    - Make. First `make -j8`, then `make install`. Pray in between.
    - Add to the `PATH`. (You might want to edit `.bashrc` again)
    ```
    export NETGENDIR="${BASEDIR}/ngsolve-install/bin"
    export PATH=$NETGENDIR:$PATH
    export PYTHONPATH=$NETGENDIR/../`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1,0,''))"`
    ```
    Usually, there is a prompt from `make` that gives you the right `PATH`.

2. Install PETSc. See [here](https://petsc.org/release/install/).

	- To install PETSc with NGSolve (MPI-purposes), you must use the same MPI implementation. 
	- To install PETSc with Firedrake, you must append
	```
	--download-pastix --download-chaco --download-netcdf --download-metis --download-hdf5 --download-hwloc --download-hypre --download-ml --download-ptscotch --download-eigen=/home/gpin2/Firedrake/firedrake/src/eigen-3.3.3.tgz --download-mpich --with-zlib --download-scalapack --with-fortran-bindings=0 --with-debugging=0 --download-cmake --download-mumps --download-bison --with-shared-libraries=1 --download-superlu_dist --download-pnetcdf --with-cxx-dialect=C++11 --download-suitesparse --with-c2html=0
	``` 
    (TBH, I am not sure if it works smoothly.)

    - To install PETSc with CUDA support you must append
    ```
    --with-cuda --download-openmpi
    ```
    It is *recommended* to employ `openmpi` in this context.

3. Install CUDA. See [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) and [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).

    - Check [pre-installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions). 
    - Run `uname -m && cat /etc/*release`. Check `gcc` installation with `gcc --version`.
    - Check the kernels `uname -r`; install the kernel headers `sudo apt-get install linux-headers-$(uname -r)`.
    - Remove outdated signing key `sudo apt-key del 7fa2af80`
    - Check the [second link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local). Do
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
    ```
    At this stage, the terminal should suggest the target path for `cp`.
    ```
    sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    ```
    - Install the GDS packages `sudo apt-get install nvidia-gds` and reboot.
    - Check your `PATH` with `echo $PATH`. You can need to append `export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}` to your `PATH`. Edit `~/.bashrc` file.
    - Pray. Move on.

0. Install OpenMPI. See [here](https://www.open-mpi.org/faq/?category=building#easy-build).
    - To install the most recent version of OpenMPI CUDA-aware requires the UCX software. See [here](https://github.com/openucx/ucx).
    - Install UCX.
        - Clone the repo. `git clone git@github.com:openucx/ucx.git`
        - Change directory. Run 
        ```
        ./autogen.sh
        ./contrib/configure-release --prefix=/where/to/install
        make -j4
        make install
        ```
        - Append the installation dir (see above `--prefix=`) to `PATH`.
    - Download OpenMPI from the site. Untar with `gunzip -c openmpi-4.1.4.tar.gz | tar xf -`.
    - Run `./configure --prefix=/usr/local --with-cuda=/usr/local/cuda --with-ucx=/path/to/ucx-cuda-install`.
    - Do `make -j4 install`.


5. Install Firedrake. See [here](https://www.firedrakeproject.org/download.html).

    - I still need to check if I can use my own PETSc instead of the FIredrake version, despite the firedrake version uses the most recent PETSc.

---

## Tricks

### Python envs
    
    - Install the python enviroments manager.
    - Create a directory for managing the different enviroments. E.g., `$HOME/.local/env`.
    - Create a python env with `venv`.
    - *Optional.* Set up BASH alias in `.bashrc` with `alias your_fav_alias='source <path>/activate'`.
    
### Stack dirs

    - Use `pushd` to push a dir in the `dirs` stack. Use `popd` to pop a dir from the stack. Use `cd -` to swap directory to the last visited directory.
