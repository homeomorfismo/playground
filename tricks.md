# Tricks

1. Python envs
    
    - Install the python enviroments manager.
    - Create a directory for managing the different enviroments. E.g., `$HOME/.local/env`.
    - Create a python env with `venv`.
    - *Optional.* Set up BASH alias in `.bashrc` with `alias your_fav_alias='source <path>/activate'`.
    
2. Stack dirs

    - Use `pushd` to push a dir in the `dirs` stack. Use `popd` to pop a dir from the stack. Use `cd -` to swap directory to the last visited directory.

3. `apt-file search <dir>` allows to find missing dependencies.

4. `git grep`-it 'til you find it.

## Other suggestions

1. `sftp` does not work if you have personalized `echo`'s messages in your `.bashrc` in the cluster. *Maybe put those in `.profile`.* **PENDING**

2. So far, the bash script for `sbatch` requires:
    - to start with `#!/bin/bash`.
    - to source modules `source /etc/profile.d/modules.sh`.

```
#!/bin/bash
#SBATCH --job-name=lapl-nref
#SBATCH --account=gpin2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gpin2@pdx.edu
#SBATCH --partition=gpu
#SBATCH --ntasks=5 
#SBATCH --output=log/lapl.out
echo "---"
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"
echo "---"
echo " "
echo "Source modules ..."
source /etc/profile.d/modules.sh
```
