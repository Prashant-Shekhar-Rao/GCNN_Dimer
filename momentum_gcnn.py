import json
import pickle
import random
import sys
import time
import glob

import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import jax
jax.config.update('jax_platform_name', 'gpu')

import netket as nk
from netket.utils import HashableArray
import equivariant_p4m as p4m
import numpy as np

from dimer_utils_gpu import *

L = int(sys.argv[1])
t = -1.0
V = float(sys.argv[2])
req_char = int(sys.argv[3])   # required character index 0-> (0,0, A1 ; 2 -> (0,0), B1; 5->B2, (pi,pi) ; 10-> P_long (0,pi)
n_layers = int(sys.argv[4])   # number of layers in GCNN
n_iter = int(sys.argv[5])
save_state_interval = int(sys.argv[6])

num_intervals = int(n_iter / save_state_interval)
jobc = None


n_samples = 8192
chunk_size = int(get_optimal_chunk_size(n_samples, L)//2)
print("--------------chunk_size----------------------", chunk_size)
n_chains = 128
sweep_size = 4
n_discard_per_chain = 400

jobc = random.randint(100, 999)
print(" L : ", L, " t : ", t, " V : ", V, " jobc ", jobc)

lattice = nk.graph.Square(L, max_neighbor_order=1)
space_group = lattice.space_group()
ct_full = space_group.character_table()              
elem_labels = [str(g) for g in space_group.elems]
classes, character_table = elem_labels, ct_full

hi = Dimer(N=lattice.n_nodes)
H = DimerHamiltonian(hi, V=V, t=t, dtype=jnp.float32)

#channel_options = np.empty(n_layers, dtype=int)
#channel_options=[4,2]
channel_options = [2*(n_layers - i) for i in range(n_layers)]
channel_str = "_".join(map(str, channel_options))

machine = p4m.GCNN(
    symmetries=lattice, 
    product_table=None,
    layers=n_layers, 
    mode="p4m", 
    features=tuple(channel_options), 
    param_dtype=jnp.complex64,
    characters=HashableArray(character_table[req_char, :])) 

sampler = nk.sampler.MetropolisSampler(hi, WormRule(), n_chains=n_chains, sweep_size=sweep_size)

opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=False)

vstate = nk.vqs.MCState(
    sampler=sampler,
    model=machine,
    n_samples=n_samples,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size,)

pattern = f"saved_states/*_state_nLayers_{n_layers}_nCh_{channel_str}_t_{t}_V_{V}_reqchar_{req_char}_L_{L}_iters_passed_*_chunkSize_*.pickle"
matching_files = glob.glob(pattern)
restart_iter = 0

if matching_files:
    iter_files = []
    for filepath in matching_files:
        filename = os.path.basename(filepath)
        iter_str = filename.split("passed_")[1].split("_chunkSize")[0]
        try:
            iterations = int(iter_str)
            iter_files.append((iterations, filepath))
        except ValueError:
            continue
    
    if iter_files:
        restart_iter, filepath = max(iter_files, key=lambda x: x[0])
        print(f"Loading state from iteration {restart_iter}: {filepath}")
        
        with open(filepath, "rb") as file:
            data = pickle.loads(file.read())
            vstate.parameters = jax.tree.map(lambda x: x.astype(jnp.complex64), data)
            jobc = int(filename.split("_")[0])
        
        print(f"Successfully loaded state from iteration {restart_iter}")
    else:
        print("No valid previous states found. Starting from scratch.")
else:
    print("No previous states found. Starting from scratch.")

#print(vstate.parameters)

#print("machine.layers = ", machine.layers)
#print("machine.features = ", machine.features)
print(" vstate.n_samples ", vstate.n_samples, "\n")
print(" vstate.n_samples_per_rank ", vstate.n_samples_per_rank, "\n")
print(" vstate.chain_length ", vstate.chain_length, "\n")
print(" sampler.sweep_size ", sampler.sweep_size, "\n")
print(" vstate.chunk_size ", vstate.chunk_size, "\n")

gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

base_name_params = f"nLayers_{n_layers}_nCh_{channel_str}_t_{t}_V_{V}_reqchar_{req_char}_L_{L}_chunkSize_{chunk_size}"
json_data_filename = f"{jobc}_jsondata_{base_name_params}"
text_data_filename = f"{jobc}_textdata_{base_name_params}.txt"
params_data_filename = f"{jobc}_params_{base_name_params}.txt"
for interval in range(1, num_intervals + 1):
    print("-----------interval -----------", interval)
    print(vstate.parameters)

    start_time = time.time()
    gs.run(n_iter=save_state_interval, out=json_data_filename, show_progress=True)
    finish_time = time.time()

    if nk.utils.mpi.rank == 0:
        current_iters = restart_iter + (interval * save_state_interval)
        os.makedirs("saved_states", exist_ok=True)
        state_filename = f"saved_states/{jobc}_state_nLayers_{n_layers}_nCh_{channel_str}_t_{t}_V_{V}_reqchar_{req_char}_L_{L}_iters_passed_{current_iters}_chunkSize_{chunk_size}.pickle"
        print(" Time ", finish_time - start_time, " Interval ", interval)

        with open(state_filename, "wb") as file:
            file.write(pickle.dumps(vstate.parameters))
            
        with open(params_data_filename, "a") as file:
            file.write(f"num_samples: {n_samples}, n_chains: {n_chains}, current_iters: {current_iters}, n_discard_per_chain: {n_discard_per_chain}, chunk_size: {chunk_size}, sweep_size: {sweep_size}\n")
        
        with open(text_data_filename, "a") as file:
            log_path = json_data_filename + ".log"
            if os.path.exists(log_path):
                 data = json.load(open(log_path))
                 for key, value in data.items():
                    file.write("%s\n" % (data[key]))
