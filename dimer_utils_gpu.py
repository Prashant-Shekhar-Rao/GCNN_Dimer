from copy import deepcopy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from netket.utils import StaticRange
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.operator._abstract_operator import AbstractOperator
from netket.sampler.rules import MetropolisRule
from netket.utils.dispatch import dispatch
from netket.vqs.mc.kernels import *
import netket.jax as nkjax

from jax import lax
from netket.hilbert import AbstractHilbert
from netket.utils.types import DType
import math
#jax.config.update("jax_enable_x64", True)



def get_optimal_chunk_size(num_samples, L):
   
    limit = int(2048 * (6 / L) ** 2)

    divisors = set()
    for i in range(limit, 1, -1):
        if num_samples % i == 0:
            return int(i)




DIRECTION_ARRAY = jnp.array([
    [0, 0, 0, 0],       # 0 filler
    [4, 3, 1, 2],       # Dir 1
    [3, 4, 2, 1],       # Dir 2
    [1, 2, 3, 4],       # Dir 3
    [2, 1, 4, 3],       # Dir 4
], dtype=jnp.int8)

# Maps dir -> opposite_dir (1<->2, 3<->4)
OPPOSITE_DIR = jnp.array([0, 2, 1, 4, 3], dtype=jnp.int8)

# Basis vectors for directions 1, 2, 3, 4
# 1: (0, -1), 2: (0, 1), 3: (1, 0), 4: (-1, 0)
BASIS_VECTORS = jnp.array([
    [0, 0],   # 0 filler
    [0, -1],  # 1
    [0, 1],   # 2
    [1, 0],   # 3
    [-1, 0]   # 4
], dtype=jnp.int8)

@jax.jit
def get_basis(direction):
    return BASIS_VECTORS[direction]

@jax.jit
def coordinate_jax(i, L):
    """Matches coordinate(i, L) from dimer_utils: (i % L, i // L)"""
    return (i % L, i // L)

@partial(jax.jit, static_argnums=(0,))
def columnar_state_jax(L):
    """Generates columnar state matching dimer_utils logic and 'F' ordering."""
    x = jnp.arange(L)
    y = jnp.arange(L)
   
    X, _ = jnp.meshgrid(x, y, indexing='ij') 
    
    state_2d = jnp.where(X % 2 == 0, 4, 3)
    state_2d = state_2d.astype(jnp.int8)
    return jnp.ravel(state_2d, order='F')

@partial(jax.jit, static_argnums=(0,))
def get_random_state_jax(L, key, sigma=None):
    if sigma is None:
        state = jnp.reshape(columnar_state_jax(L), (L, L), order='F')
    else:
        state = jnp.reshape(sigma, (L, L), order='F')
    
    state = state.astype(jnp.int8)

    key, subkey = jax.random.split(key)
    initial_site = jax.random.randint(subkey, (), 0, L * L)
    x, y = coordinate_jax(initial_site, L)

    local_state = state[x, y]
    b = get_basis(local_state)
    
    site_behind = jnp.array([x, y], dtype=jnp.int8)
    site_ahead = jnp.array([(x + L + b[0]) % L, (y + L + b[1]) % L], dtype=jnp.int8)
    
    state = state.at[x, y].set(-1)
    
    
    # tuple structure: (state, behind, dir, ahead, parity, KEY, finished)
    # Key is at index 5
    init_val = (state, site_behind, local_state, site_ahead, -1, key, False)

    def cond_fun(val):
        return ~val[-1]

    def body_fun(val):
        s, behind, c_dir, ahead, parity, k, _ = val
        k, step_key = jax.random.split(k)
        
        # --- Branch 1: Parity == -1 ---
        def case_parity_neg(args):
            s_curr, ahead_curr, cd_curr, rng = args
            
            rand_idx = jax.random.randint(rng, (), 0, 3)
            new_dir = DIRECTION_ARRAY[cd_curr, rand_idx].astype(jnp.int8)
            
            b_vec = get_basis(new_dir)
            new_ahead = jnp.array([(ahead_curr[0] + L + b_vec[0]) % L, 
                                   (ahead_curr[1] + L + b_vec[1]) % L], dtype=jnp.int8)
            
            s_new = s_curr.at[ahead_curr[0], ahead_curr[1]].set(new_dir)
            
            return s_new, ahead_curr, new_dir, new_ahead, 1, False

        # --- Branch 2: Parity == 1 ---
        def case_parity_pos(args):
            s_curr, ahead_curr, cd_curr, rng = args
            state_ahead_val = s_curr[ahead_curr[0], ahead_curr[1]]
            
            def continue_worm(_):
                new_dir_inner = state_ahead_val
                b_vec = get_basis(new_dir_inner)
                new_ahead_inner = jnp.array([(ahead_curr[0] + L + b_vec[0]) % L, 
                                             (ahead_curr[1] + L + b_vec[1]) % L], dtype=jnp.int8)
                
                s_new = s_curr.at[ahead_curr[0], ahead_curr[1]].set(OPPOSITE_DIR[cd_curr])
                return s_new, ahead_curr, new_dir_inner, new_ahead_inner, -1, False

            def close_loop(_):
                s_new = s_curr.at[ahead_curr[0], ahead_curr[1]].set(OPPOSITE_DIR[cd_curr])
                return s_new, behind, cd_curr, ahead, parity, True

            return lax.cond(state_ahead_val != -1, continue_worm, close_loop, None)

        res = lax.cond(parity == -1,
                       case_parity_neg,
                       case_parity_pos,
                       (s, ahead, c_dir, step_key))
        
        return (res[0], res[1], res[2], res[3], res[4], k, res[5])

    final_val = lax.while_loop(cond_fun, body_fun, init_val)
    
    final_state = final_val[0]
    final_key = final_val[5] 
    
    return jnp.ravel(final_state, order='F'), final_key

def detectFlippablePlaq(state, x, y, L):
    state_2d = state.reshape((L, L), order="F")
    s1 = state_2d[x, y]

    
    def check_horizontal(_):
        b_vec = get_basis(4) # (-1, 0)
        nx, ny = (x + b_vec[0]) % L, (y + b_vec[1]) % L
        s2 = state_2d[nx, ny]
        return jnp.where(s2 == 2, jnp.array([1, 0], dtype=jnp.int8), jnp.array([0, 0], dtype=jnp.int8))

    
    def check_vertical(_):
        b_vec = get_basis(2) # (0, 1)
        nx, ny = (x + b_vec[0]) % L, (y + b_vec[1]) % L
        s2 = state_2d[nx, ny]
        return jnp.where(s2 == 4, jnp.array([0, 1], dtype=jnp.int8), jnp.array([0, 0], dtype=jnp.int8))

   
    return lax.cond(
        s1 == 2,
        check_horizontal,
        lambda _: lax.cond(
            s1 == 4,
            check_vertical,
            lambda _: jnp.array([0, 0], dtype=jnp.int8),
            None
        ),
        None
    )

@jax.jit
def countNumOfFlippablePlaq(state):
    """
    Vectorized counting of flippable plaquettes.
    """
    L = int(np.sqrt(state.shape[-1]))
    
  
    x_grid, y_grid = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing='ij')
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    
    # in_axes: state=None (fixed), x=0 (mapped), y=0 (mapped), L=None (fixed)
    vmap_detect = jax.vmap(detectFlippablePlaq, in_axes=(None, 0, 0, None))
    
    
    results = vmap_detect(state, x_flat, y_flat, L)
    
    return jnp.sum(results)



#===================++++++++++SAMPLER RULE++++++++++============================            


class WormRule(MetropolisRule):
    r"""
    A Rule that generates a valid dimer configuration.
    """

    def __init__(
        self,
        *,
        graph = None,
    ):
        r"""
        Constructs the Worm Rule.

        """
    def transition(rule, sampler, machine, parameters, state, key, σ):
        """
        Performs a parallel MC sweep over a batch of configurations.

        Args:
            key (jax.random.PRNGKey): A single key for the entire transition.
            σ_batch (jnp.ndarray): A batch of configurations, shape (n_chains, L, L).

        Returns:
            jnp.ndarray: The updated batch of configurations.
        """
        n_chains = σ.shape[0]
        L = int(np.sqrt(σ.shape[1]))
       

        keys = jax.random.split(key, n_chains)
        vmapped_sweep = jax.vmap(lambda key, config: get_random_state_jax(L,key, config), in_axes=(0, 0))
        σp, _ = vmapped_sweep(keys, σ)
        
        
        return σp, None
    
    
    def random_state(self, sampler, machine, params, sampler_state, key):
        """Generate thermally weighted random states using MC sweeps."""
        
        # Get system size and number of chains
        L = int(np.sqrt(sampler.hilbert.size))
        n_chains = sampler.n_chains  # Use n_chains_per_rank for parallel compatibility
        
        # Split keys for each chain
        chain_keys = jax.random.split(key, n_chains)
        
        # Generate single thermalized state for one chain
        def generate_single_chain_state(chain_key):
            # Create initial configuration
            initial_config = columnar_state_jax(L).astype(jnp.int8)
            
            
            # Define scan function for multiple sweeps
            def sweep_step(carry, _):
                config, key_current = carry
                config_new, key_new = get_random_state_jax(L,key_current, config)
                return (config_new, key_new), None  # Return updated carry and no output
            
            # Perform multiple MC sweeps for thermalization
            (config, return_key), _ = jax.lax.scan(
                sweep_step, 
                (initial_config, chain_key), 
                None,
                length=400
            )
        
            
            
            return config  # Flatten to match expected shape
        
        # Vectorize over all chains using vmap
        states = jax.vmap(generate_single_chain_state)(chain_keys)
        
        # Ensure correct dtype and shape
        states = jnp.asarray(states, dtype=sampler.dtype)
        
        return states

    def __repr__(self):
        return "WormRule"

############################################################################



###########################################################################################################################################

# --------------------------DIMER HILBERT SPACE--------------------------------------------------------------------------------------------

###########################################################################################################################################


class Dimer(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local dimer state o lattice nodes."""

    def __init__(
        self,
        N: int = 1,
    ):
        r"""Hilbert space obtained as tensor product of local dimer states.

        Args:

           N: Number of sites (default=1)


        Examples:
           Simple dimer hilbert space.

           >>> import netket as nk
           >>> hi = nk.hilbert.Dimer(s=1/2, N=4)
           >>> print(hi.size)
           4
        """
        local_size = 4
        local_states = StaticRange(1,1,4, dtype=jnp.int8)

        super().__init__(local_states, N)

    def __pow__(self, n):
        if not self.constrained:
            return Dimer(self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if not self.constrained and not other.constrained:
                return Dimer(N=self.size + other.size)

        return NotImplemented

    def __repr__(self):
        return f"Dimer(N={self.size})"

    @property
    def _attrs(self):
        return self.size


############################################################################




###########################################################################################################################################

# --------------------------DIMER Hamiltonian --------------------------------------------------------------------------------------------

###########################################################################################################################################


class DimerHamiltonian(AbstractOperator):
    def __init__(
        self,
        hilbert: HomogeneousHilbert,
        V: float,
        t: float,
        dtype:  float,
    ):
        super().__init__(hilbert)

        self._t = jnp.array(t, dtype=dtype)
        self._V = jnp.array(V, dtype=dtype)

    @property
    def t(self) -> float:
        """The magnitude of the hopping term"""
        return self._t

    @property
    def V(self) -> float:
        """The magnitude of the potential term"""
        return self._V

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True
    

def get_conn_elements_jax(state, L):
    
    
    x_grid, y_grid = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing='ij')
    all_coords = jnp.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

    def process_plaquette(coords):
        x, y = coords[0], coords[1]
        
        state_2d = state.reshape((L, L), order="F")
        s1 = state_2d[x, y]
        
       
        def do_update_horizontal(_):
            # Sets plaquette to Horizontal (4, 3, 4, 3)
            s_new = state_2d.at[x, y].set(4)
            s_new = s_new.at[(x - 1) % L, y].set(3)
            s_new = s_new.at[x, (y + 1) % L].set(4)
            s_new = s_new.at[(x - 1) % L, (y + 1) % L].set(3)
            return jnp.ravel(s_new, order="F"), 1.0

        def do_update_vertical(_):
            # Sets plaquette to Vertical (2, 2, 1, 1)
            s_new = state_2d.at[x, y].set(2)
            s_new = s_new.at[(x - 1) % L, y].set(2)
            s_new = s_new.at[x, (y + 1) % L].set(1)
            s_new = s_new.at[(x - 1) % L, (y + 1) % L].set(1)
            return jnp.ravel(s_new, order="F"), 1.0

        def do_nothing(_):
            return state, 0.0

       
        n_left = state_2d[(x - 1) % L, y]
        b2 = get_basis(2)
        n_up = state_2d[(x + b2[0]) % L, (y + b2[1]) % L]

        
        can_flip_horiz = (s1 == 2) & (n_left == 2) #currently Vertical
        can_flip_vert  = (s1 == 4) & (n_up == 4)   # currently Horizontal

        
        
        res = lax.cond(
            can_flip_horiz,
            do_update_horizontal,
            lambda _: lax.cond(
                can_flip_vert,
                do_update_vertical,
                do_nothing,
                None
            ),
            None
        )
        return res

   
    connected_states, weights = jax.vmap(process_plaquette)(all_coords)

    return connected_states, weights

def _get_conns_and_mels_single(sigma, t, V):
    """
    Compute connected states and matrix elements for a SINGLE configuration.

    sigma : (N,)  — one dimer configuration
    t, V  : scalars

    Returns
    -------
    eta   : (L²+1, N)   — [sigma itself, then L² connected states]
    mels  : (L²+1,)     — [diagonal weight, then L² off-diagonal weights]
    """
   
    L = int(np.sqrt(sigma.shape[-1]))

    beta, wts = get_conn_elements_jax(sigma, L)
    # beta : (L², N),  wts : (L²,)

    eta = jnp.concatenate([sigma[None, :], beta], axis=0)   # (L²+1, N)

    num_flippable   = jnp.sum(wts)
    diagonal_weight = num_flippable * V                      # scalar
    off_diag_weights = t * wts                               # (L²,)

    
    final_weights = jnp.concatenate(
        [jnp.expand_dims(diagonal_weight, 0), off_diag_weights], axis=0
    )                                                        # (L²+1,)

    return eta, final_weights


def get_conns_and_mels(sigma_batch, t, V):
    """
    Batched version of _get_conns_and_mels_single.

    sigma_batch : (batch, N)
    Returns
    -------
    eta   : (batch, L²+1, N)
    mels  : (batch, L²+1)
    """
    return jax.vmap(lambda s: _get_conns_and_mels_single(s, t, V))(sigma_batch)

def custom_chunked_kernel(logpsi, pars, sigma, args, *, chunk_size=None):
    t, V = args
    
    def process_single_sample(s):
        eta, mels = _get_conns_and_mels_single(s, t, V)
        
        lp_sigma = logpsi(pars, s[None, :])[0]
        lp_eta = logpsi(pars, eta)
        
        return jnp.sum(mels * jnp.exp(lp_eta - lp_sigma))

   
    chunked_fn = nkjax.vmap_chunked(
        process_single_sample, in_axes=0, chunk_size=chunk_size
    )
    
    N = sigma.shape[-1]
    
    return chunked_fn(sigma.reshape(-1, N))
    


def e_loc(logpsi, pars, sigma, extra_args):
    t_arr, V_arr = extra_args
    t = t_arr[0]
    V = V_arr[0]

    # Compute connected elements for just this chunk (chunk_size samples)
    eta, mels = get_conns_and_mels(sigma, t, V)

    # sigma is (chunk_size, Nsites)
    assert sigma.ndim == 2
    # eta is (chunk_size, Nconnected, Nsites)
    assert eta.ndim == 3

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)


# ---- Kernel dispatches -------------------------------------------------------

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: DimerHamiltonian):
    return e_loc

@dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: DimerHamiltonian, chunk_size: int):
    return custom_chunked_kernel


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: DimerHamiltonian):
    sigma = vstate.samples       # (..., N)
    return sigma, (op.t, op.V)



@nk.vqs.expect.dispatch
def expect_mcstate_operator_chunked(
    vstate: nk.vqs.MCState, Ô: DimerHamiltonian, chunk_size: int
):

    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    return _expect_chunking(
        chunk_size,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_chunking(
    chunk_size: int,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    args: PyTree,
):
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    _, Ō_stats = nkjax.expect(
        log_pdf,
        partial(local_value_kernel, logpsi, chunk_size=chunk_size),
        parameters,
        σ,
        args,
        n_chains=σ_shape[0],
    )

    return Ō_stats

