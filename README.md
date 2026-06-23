# CatLearn

CatLearn utilizes machine learning, specifically the Gaussian Process or Student T process, to accelerate catalysis simulations.

The local optimization of a structure is accelerated with the `LocalAL` code.
The Nudged-elastic-band method (NEB) is accelerated with the `MLNEB` code.
Furthermore, a global adsorption search without local relaxation is accelerated with the `AdsorptionAL` code.
Additionally, a global adsorption search with local relaxation is accelerated with the `MLGO` code. 
At last, a random sampling of adsorbate positions, combined with local relaxation, accelerates the global adsorption search with the `RandomAdsorptionAL` code.

CalLearn uses ASE to handle atomic systems and the calculator interface to calculate the potential energy.

## Installation

You can install CatLearn by downloading it from GitHub as:
```shell
git clone https://github.com/avishart/CatLearn
pip install -e CatLearn/.
```

You can also install CatLearn directly from GitHub:
```shell
pip install git+https://github.com/avishart/CatLearn.git
```

However, it is recommended to install a specific tag to ensure it is a stable version:
```shell
pip install git+https://github.com/avishart/CatLearn.git@v.x.x.x
```

The dependency of ASE has only been thoroughly tested up to version 3.26.0.

## VASP compatibility

When using an external ASE calculator such as VASP, let VASP handle its own MPI
parallelization and prevent CatLearn from launching one VASP calculation per
Python rank. CatLearn detects ASE VASP calculators and evaluates the true
calculator only on rank 0 by default. Non-root CatLearn ranks wait in a low-CPU
MPI probe loop while rank 0 runs the external calculation. After VASP finishes,
rank 0 sends the evaluated structure, energy, and forces to the other CatLearn
ranks with tagged `mpi4py` point-to-point messages.

This behavior is controlled by the `parallel_eval` argument. Use
`parallel_eval=False` for VASP or any external executable calculator that should
not be launched once per Python rank. Use `parallel_eval=True` only for
calculators that are safe to call from every Python rank, for example in-process
calculators designed for that execution model.

You can also force this behavior explicitly:

```python
dyn = LocalAL(
    atoms=atoms,
    ase_calc=vasp_calc,
    parallel_run=True,
    parallel_eval=False,
)
```

For `MLNEB`, use the same setting when constructing the optimizer:

```python
mlneb = MLNEB(
    start=initial,
    end=final,
    ase_calc=vasp_calc,
    parallel_run=True,
    parallel_eval=False,
)
```

### What changed for VASP

The VASP compatibility path includes these changes:

- `ActiveLearning` and `MLNEB` accept `parallel_eval`. If it is not set,
  CatLearn automatically uses `parallel_eval=False` for ASE VASP calculators.
- With `parallel_eval=False`, only rank 0 calls the ASE calculator. Non-root
  ranks sleep in a low-CPU `mpi4py` wait loop until rank 0 sends the result.
- ASE's internal parallel IO is forced to behave serially while rank 0 reads
  VASP results. This avoids hangs in ASE's `vasprun.xml` reader, which otherwise
  tries to call `ase.parallel.broadcast()` while the other ranks are waiting
  outside ASE.
- CatLearn's parallel handoffs for candidates, predictions, convergence flags,
  copied NEB images, NEB energies, NEB forces, and per-candidate properties use
  tagged `mpi4py` messages instead of ASE collective broadcasts. This avoids
  message mixing between CatLearn communication and ASE/VASP result handling.
- Uncertainty values are converted to scalars before taking maxima, so harmless
  scalar versus length-one-array differences from calculators do not stop the
  optimization.

The implementation is split across these files:

- `catlearn/activelearning/activelearning.py` adds `parallel_eval`, detects ASE
  VASP calculators, evaluates external calculators only on rank 0, sends
  evaluation payloads with `mpi4py`, serializes ASE's internal VASP result IO,
  and replaces prediction/convergence/candidate handoffs that previously used
  ASE broadcasts.
- `catlearn/activelearning/mlneb.py` passes the `parallel_eval` argument through
  the MLNEB constructor, restart/copy paths, and saved argument handling.
- `catlearn/optimizer/localneb.py` replaces parallel copied-image sharing in
  `LocalNEB.get_structures_parallel()` with tagged `mpi4py` messages.
- `catlearn/optimizer/method.py` replaces generic per-candidate parallel
  sharing of copied candidates, energies, forces, uncertainties, and properties
  with tagged `mpi4py` messages, and converts uncertainty values to scalars
  before taking maxima.
- `catlearn/structures/neb/orgneb.py` replaces parallel NEB image energy and
  force sharing in `OriginalNEB.calculate_properties_parallel()` with tagged
  `mpi4py` messages.

For parallel MLNEB with VASP, run the Python side with `mpi4py`. Do not launch
multiple plain Python processes without `mpi4py`, because each process would act
as an independent script and could write the same files or launch duplicate VASP
calculations.

On a single Slurm node, a working pattern is to use fewer Python ranks for the
MLNEB path and all allocated ranks for VASP. Keep both the Python/CatLearn MPI
job and the VASP MPI launch on shared-memory fabric:

```shell
export MLNEB_NTASKS=18
export VASP_NTASKS=${SLURM_NTASKS}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export I_MPI_WAIT_MODE=1
export I_MPI_SPIN_COUNT=0
export I_MPI_FABRICS=shm
export FI_PROVIDER=shm
export I_MPI_OFI_PROVIDER=shm

export ASE_VASP_COMMAND="env I_MPI_FABRICS=shm FI_PROVIDER=shm \
I_MPI_OFI_PROVIDER=shm I_MPI_WAIT_MODE=1 I_MPI_SPIN_COUNT=0 \
mpirun -np ${VASP_NTASKS} vasp_std"

srun --unbuffered -n ${MLNEB_NTASKS} --cpu-bind=cores python3 -u -m mpi4py run_mlneb_restart.py
```

The thread variables prevent NumPy, BLAS, MKL, and OpenMP libraries from
starting extra hidden threads inside each MPI rank. The Intel MPI shared-memory
variables avoid single-node HFI/OFI startup errors such as `PSM2 can't open hfi
unit` or `OFI endpoint open failed` when VASP is launched from inside an already
running Python MPI job.

For large MLNEB runs, memory can grow with the number of Python ranks and stored
trajectory/calculator objects. If the job is killed by the scheduler for memory,
reduce `MLNEB_NTASKS`, avoid unnecessary restarts from old trajectory files, and
consider setting `copy_calc=False` in the MLNEB setup.

The VASP smoke test is skipped by default. To run it:

```shell
CATLEARN_RUN_VASP_TESTS=1 python -m unittest tests.test_mlneb_vasp
```

## Usage
The active learning class is generalized to work for any defined optimizer method for ASE `Atoms` structures. The optimization method is executed iteratively with a machine-learning calculator that is retrained for each iteration. The active learning converges when the uncertainty is low (`unc_convergence`) and the energy change is within `unc_convergence` or the maximum force is within the tolerance value set.

Predefined active learning methods are created: `LocalAL`, `MLNEB`, `AdsorptionAL`, `MLGO`, and `RandomAdsorptionAL`.

The outputs of the active learning are `predicted.traj`, `evaluated.traj`, `predicted_evaluated.traj`, `converged.traj`, `initial_struc.traj`, `ml_summary.txt`, and `ml_time.txt`:
- The `predicted.traj` file contains the structures that the machine-learning calculator predicts after each optimization loop.
- The training data and ASE calculator evaluated structures are within `evaluated.traj` file.
- The `predicted_evaluated.traj` file has the exact same structures as the `evaluated.traj` file, but with machine-learning predicted properties.
- The converged structures calculated with the machine-learning calculator are saved in the `converged.traj` file.
- The initial structure(s) is/are saved into the `initial_struc.traj` file.
- The summary of the active learning is saved into a table in the `ml_summary.txt` file.
- The time spent on structure evaluation, machine-learning training, and prediction at each iteration is stored in `ml_time.txt`.

### LocalAL
The following code shows how to use `LocalAL`:
```python
from catlearn.activelearning.local import LocalAL
from ase.io import read
from ase.optimize import FIRE

# Load initial structure
atoms = read("initial.traj")

# Make the ASE calculator
calc = ...

# Initialize local optimization
dyn = LocalAL(
    atoms=atoms,
    ase_calc=calc,
    unc_convergence=0.05,
    local_opt=FIRE,    
    local_opt_kwargs={},
    save_memory=False,
    use_restart=True,
    min_data=3,
    restart=False,
    verbose=True,
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=500,
)

```

The active learning minimization can be visualized by extending the Python script with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_minimize

fig, ax = plt.subplots()
plot_minimize("predicted_evaluated.traj", "evaluated.traj", ax=ax)
plt.savefig('AL_minimization.png')
plt.close()
```

### MLNEB
The following code shows how to use `MLNEB`:
```python
from catlearn.activelearning.mlneb import MLNEB
from ase.io import read
from ase.optimize import FIRE

# Load endpoints
initial = read("initial.traj")
final = read("final.traj")

# Make the ASE calculator
calc = ...

# Initialize MLNEB
mlneb = MLNEB(
    start=initial,
    end=final,
    ase_calc=calc,
    unc_convergence=0.05,
    n_images=15,
    neb_method="improvedtangentneb",
    neb_kwargs={},
    neb_interpolation="linear",
    start_without_ci=True,
    reuse_ci_path=True,
    save_memory=False,
    parallel_run=True,
    local_opt=FIRE,    
    local_opt_kwargs={},
    use_restart=True,
    min_data=3,
    restart=False,
    verbose=True,
)
mlneb.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=500,
)

```

The `MLNEB` optimization can be restarted from the last predicted path and reusing the training data with the argument `restart=True`. Alternatively, the optimization can be restarted from the last predicted path without reusing the training data by setting the `neb_interpolation="predicted.traj"`.

The obtained NEB band from the MLNEB optimization can be visualized in three ways.

The converged NEB band with uncertainties can be visualized by extending the Python code with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_neb

fig, ax = plt.subplots()
plot_neb(mlneb.get_structures(), use_uncertainty=True, ax=ax)
plt.savefig('Converged_NEB.png')
plt.close()
```

The converged NEB band can also be plotted with the predicted curve between the images by extending with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_neb_fit_mlcalc

fig, ax = plt.subplots()
plot_neb_fit_mlcalc(
    mlneb.get_structures(),
    mlcalc=mlneb.get_mlcalc(),
    use_uncertainty=True,
    include_noise=True,
    ax=ax,
)
plt.savefig('Converged_NEB_fit.png')
plt.close()
```

All the obtained NEB bands from `MLNEB` can also be visualized within the same figure by using the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_all_neb

fig, ax = plt.subplots()
plot_all_neb("predicted.traj", n_images=15, ax=ax)
plt.savefig('All_NEB_paths.png')
plt.close()
```

### AdsorptionAL
The following code shows how to use `AdsorptionAL`:
```python
from catlearn.activelearning.adsorption import AdsorptionAL
from ase.io import read

# Load the slab and the adsorbate
slab = read("slab.traj")
ads = read("adsorbate.traj")

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
    ]
)

# Initialize MLGO
dyn = AdsorptionAL(
    slab=slab,
    adsorbate=ads,
    adsorbate2=None,
    ase_calc=calc,
    unc_convergence=0.02,
    bounds=bounds,
    opt_kwargs={},
    parallel_run=False,
    min_data=3,
    restart=False,
    verbose=True
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=4000,
)

```

The `AdsorptionAL` optimization can be visualized in the same way as the `LocalAL` optimization.

### MLGO
The following code shows how to use `MLGO`:
```python
from catlearn.activelearning.mlgo import MLGO
from ase.io import read
from ase.optimize import FIRE

# Load the slab and the adsorbate
slab = read("slab.traj")
ads = read("adsorbate.traj")

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
    ]
)

# Initialize MLGO
mlgo = MLGO(
    slab=slab,
    adsorbate=ads,
    adsorbate2=None,
    ase_calc=calc,
    unc_convergence=0.02,
    bounds=bounds,
    opt_kwargs={},
    local_opt=FIRE,
    local_opt_kwargs={},
    reuse_data_local=True,
    parallel_run=False,
    min_data=3,
    restart=False,
    verbose=True
)
mlgo.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=4000,
)

```

The `MLGO` optimization can be visualized in the same way as the `LocalAL` optimization.

### RandomAdsorptionAL
The following code shows how to use `RandomAdsorptionAL`:
```python
from catlearn.activelearning.randomadsorption import RandomAdsorptionAL
from ase.io import read
from ase.optimize import FIRE

# Load the slab and the adsorbate
slab = read("slab.traj")
ads = read("adsorbate.traj")

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
    ]
)

# Initialize MLGO
dyn = RandomAdsorptionAL(
    slab=slab,
    adsorbate=ads,
    adsorbate2=None,
    ase_calc=calc,
    unc_convergence=0.02,
    bounds=bounds,
    n_random_draws=200,
    use_initial_opt=False,
    initial_fmax=0.2,
    use_repulsive_check=True,
    local_opt=FIRE,
    local_opt_kwargs={},
    parallel_run=False,
    min_data=3,
    restart=False,
    verbose=True
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=4000,
)

```

The `RandomAdsorptionAL` optimization can be visualized in the same way as the `LocalAL` optimization.
