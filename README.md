# Group Convolutional Neural Network for the Low-Energy Spectrum in the Quantum Dimer Model

This repository contains the code for training neural-network variational wavefunctions to find the ground state of the Quantum Dimer Model, as described in https://arxiv.org/abs/2505.23728.

## Installation

Create a new virtual environment by running

```bash
python -m venv Quantum_Dimer
```

and activate it with

```bash
source Quantum_Dimer/bin/activate
```

Install the required packages with

```bash
pip install --no-deps -r requirements.txt
```

`requirements.txt` has the exact version number for each package. We don't recommend using newer packages as they are liable to break things.

If you don't have a Nvidia GPU you can still evaluate the energies by installing from `requirements_no_gpu.txt`
## Training

`momentum_gcnn.py` is used to train the neural network; it requires six parameters.

```bash
python momentum_gcnn.py <L> <V> <req_char> <n_layers> <n_iter> <save_state_interval>
```

1. `<L>` is the lattice side length. Only even values are permitted.

2. `<V>` is the potential energy parameter. The value of $t$ is fixed to $-1$.

3. `<req_char>` is the irrep / symmetry sector to optimize. Internally this integer is the row index of the character table returned by NetKet's `space_group.character_table()`. `0` gives the global ground state; other values give the lowest state in the corresponding (excited) sector.

    - `0` : $A_1$ irrep at momentum $(0,0)$.
    - `2` : $B_1$ irrep at momentum $(0,0)$.
    - `5` : $B_2$ irrep at momentum $(\pi,\pi)$.
    - `10`: $p_{\text{long}}$ 2D irrep at momentum $(0,\pi)$.
4. `<n_layers>`: Number of hidden layers in the GCNN (recommended number of layers is 2).

5. `<n_iter>`: Total number of steps.

6. `<save_state_interval>`: Number of steps before we save a model.


#### Examples

For the $B_{1}$ state
```bash
python momentum_gcnn.py 8 0.6 2 2 2000 50
```

For the ground state
```bash
python momentum_gcnn.py 8 0.8 0 2 2000 500
```

### Other parameters

These parameters are not taken as input but are available in `momentum_gcnn.py` and can be changed.

- `chunk size` calculations are split into chunks where the neural network is evaluated at most on `chunk_size` samples at once. This does not change the mathematical results, but will trade a higher computational cost for lower memory cost.
- `n_samples = 8192` Total number of samples per training iteration. We recommend larger values for more accurate energies.

---

## Outputs and checkpointing

During training the checkpoints are saved in the `saved_states/` directory, while other logs are saved in the working directory.

The checkpoint filenames encode the run configuration (`L`, `V`, `req_char`, `n_layers`, channel widths, iteration count, and chunk size). On startup the script looks for an existing checkpoint matching the current configuration and, if found, resumes from there.

## Evaluating energies

```python
import sys
import pickle
import jax
import jax.numpy as jnp
import netket as nk
import equivariant_p4m as p4m
from netket.utils import HashableArray
from dimer_utils_gpu import Dimer, DimerHamiltonian, WormRule


L = 8
V = 0.6
t=-1.0
req_char = 0
pickle_path = "saved_states/171_state_nLayers_2_nCh_4_2_t_-1.0_V_0.6_reqchar_0_L_8_iters_passed_1100_chunkSize_4096.pickle"
#603 is the job id and should be ignored.
#Every variable except the chunk size and iters_passed must match with the filename.
#If you are trying this without a GPU, then it would take some time.
n_discard_per_chain=400
chunk_size=256
lattice = nk.graph.Square(L, max_neighbor_order=1)
hi = Dimer(N=lattice.n_nodes)
H = DimerHamiltonian(hi, V=V, t=t, dtype=jnp.float32)


ct_full = lattice.space_group().character_table()
machine = p4m.GCNN(
    symmetries=lattice,
    layers=2,
    mode="p4m",
    features=(4, 2),  # Must match the nCh string in the filename
    param_dtype=jnp.complex64,
    characters=HashableArray(ct_full[req_char, :])
)


sampler = nk.sampler.MetropolisSampler(hi, WormRule(), n_chains=128, sweep_size=4)



vstate = nk.vqs.MCState(sampler=sampler, model=machine, n_samples=16384,n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size)


with open(pickle_path, "rb") as file:
    data = pickle.loads(file.read())

    vstate.parameters = jax.tree_util.tree_map(lambda x: x.astype(jnp.complex64), data)

print(f"Loaded parameters from: {pickle_path}")
print(f"Sampling with {vstate.n_samples} samples and chunk size {vstate.chunk_size} to evaluate energy...")


energy_stats = vstate.expect(H)

print("\n--- Results ---")
print(f"Total Energy: {energy_stats.mean}")
print(f"Energy Density: {energy_stats.mean / (L**2)}")
print(f"Variance: {energy_stats.variance}")
print(f"MC Error: {energy_stats.error_of_mean}")
#--- Results ---
#Total Energy: (-7.392194154337827+0.0004310530263463118j)
#Energy Density: (-0.11550303366152855+6.7352035366611215e-06j)
#Variance: 0.005512039438871472
#MC Error: 0.000677387426279752

```

## Evaluating the wavefunction

```python
import pickle
import jax
import jax.numpy as jnp
import netket as nk
import equivariant_p4m as p4m
from netket.utils import HashableArray
from dimer_utils_gpu import Dimer, WormRule, columnar_state_jax

L = 12
req_char = 2
pickle_path = "saved_states/603_state_nLayers_2_nCh_4_2_t_-1.0_V_0.6_reqchar_2_L_12_iters_passed_1550_chunkSize_256.pickle"  # Path to your saved weights.
#603 is the job id and should be ignored.
#Every variable except the chunk size and iters_passed must match with the filename.

lattice = nk.graph.Square(L, max_neighbor_order=1)
ct_full = lattice.space_group().character_table()

machine = p4m.GCNN(
    symmetries=lattice,
    layers=2,
    mode="p4m",
    features=(4, 2),
    param_dtype=jnp.complex64,
    characters=HashableArray(ct_full[req_char, :])
)


# Use the Dimer Hilbert space the model was trained on.
# A sampler is required to build the MCState but is never used for log_value.
hi = Dimer(N=lattice.n_nodes)
sampler = nk.sampler.MetropolisSampler(hi, WormRule(), n_chains=128, sweep_size=4)
vstate = nk.vqs.MCState(sampler=sampler, model=machine)


with open(pickle_path, "rb") as file:
    data = pickle.loads(file.read())

vstate.parameters = jax.tree_util.tree_map(lambda x: x.astype(jnp.complex64), data)


# We evaluate the wavefunction on the columnar state
sigma = columnar_state_jax(L).reshape(1, -1)


log_psi = vstate.log_value(sigma)
psi = jnp.exp(log_psi)

print(f"Log Amplitude: {log_psi[0]}")
print(f"Amplitude psi(sigma): {psi[0]}")
print(f"Unormalized probability |psi|^2: {jnp.abs(psi[0])**2}")
```
## Citations

Simulations for this work were performed with codes built on top of NetKet
```bibtex
@Article{netket3:2022,
    title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
    author={Filippo Vicentini and Damian Hofmann and Attila Szabó and Dian Wu and Christopher Roth and Clemens Giuliani and Gabriel Pescia and Jannes Nys and Vladimir Vargas-Calderón and Nikita Astrakhantsev and Giuseppe Carleo},
    journal={SciPost Phys. Codebases},
    pages={7},
    year={2022},
    publisher={SciPost},
    doi={10.21468/SciPostPhysCodeb.7},
    url={https://scipost.org/10.21468/SciPostPhysCodeb.7}
}

@article{netket2:2019,
    title={NetKet: A machine learning toolkit for many-body quantum systems},
    author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
    journal={SoftwareX},
    volume={10},
    pages={100311},
    year={2019},
    publisher={Elsevier},
    doi={10.1016/j.softx.2019.100311},
    url={https://www.sciencedirect.com/science/article/pii/S2352711019300974}
}

@misc{jax2018github,
  title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and {Wanderman-Milne}, Skye and Zhang, Qiao},
  year = {2018}
}

@misc{flax2020github,
  title = {Flax: {{A}} Neural Network Library and Ecosystem for {{JAX}}},
  author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and {van Zee}, Marc},
  year = {2024}
}
```
