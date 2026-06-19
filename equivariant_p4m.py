# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
from numpy import prod

from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, lecun_normal
from jax.scipy.special import logsumexp

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc, Array
from netket.utils.group import PermutationGroup
from netket.graph import Graph, Lattice
from netket.jax import logsumexp_cplx, is_complex_dtype
from netket.nn.activation import reim_selu
from symmetric_linear_p4m import (
    DenseSymmMatrix,
    DenseSymmFFT,
    DenseSymmFFT_p4m,
    DenseEquivariantFFT,
    DenseEquivariantIrrep,
)

# Same as netket.nn.symmetric_linear.default_equivariant_initializer
# All GCNN layers have kernels of shape [out_features, in_features, n_symm]
default_gcnn_initializer = lecun_normal(in_axis=1, out_axis=0)


def identity(x):
    return x

ide=lambda r,L: r
rot90=lambda r,L: np.array([(L-r[1])%L,r[0]])
rot180=lambda r,L: np.array([(-r[0]+L)%L,(L-r[1])%L])
rot270=lambda r,L: np.array([r[1],(L-r[0])%L])
mirrory=lambda r,L: np.array([(L-r[0])%L,r[1]])
mirrorx=lambda r,L: np.array([r[0],(L-r[1])%L ])
mirrorxy=lambda r,L: np.array([r[1],r[0]])
mirror_xy=lambda r,L: np.array([(L-r[1])%L,(L-r[0])%L])
#key=["id",'irot270','irot180','irot90','imirrorx','imirrory','imirrorxy','imirror_xy']
#val=[[0,1,2,3,4],[0,3,4,2,1],[0,2,1,4,3],[0,4,3,1,2],[0,2,1,3,4],[0,1,2,4,3],[0,4,3,2,1],[0,3,4,1,2]]
#transformations_dict = dict(zip(key, val))

key=["id",'rot270','rot180','rot90','mirrorx','mirrory','mirrorxy','mirror_xy']
val=[[0,1,2,3,4],[0,4,3,1,2],[0,2,1,4,3],[0,3,4,2,1],[0,2,1,3,4],[0,1,2,4,3],[0,4,3,2,1],[0,3,4,1,2]]
transformations_dict = dict(zip(key, val))

def D4xTDecomposition(permutation,L,label="name",test=False):
    coordiates=[np.array([i,j]) for j in range(L) for i in range(L)]
    delta=coordiates[permutation[0]]+L
    if label=="name":
        label=list(transformations_dict.keys())#["id",'rot270','rot180','rot90','mirrorx','mirrory','mirrorxy','mirror_xy']
    elif label=="list":
        label=list(transformations_dict.values())
        
        #[[0,1,2,3,4],[0,4,3,1,2],[0,2,1,4,3],[0,3,4,2,1],[0,2,1,3,4],[0,1,2,4,3],[0,4,3,2,1],[0,3,4,1,2]]
        #label=[[0,1,2,3,4],[0,4,3,1,2],[0,2,1,4,3],[0,3,4,2,1],[0,2,1,3,4],[0,1,2,4,3],[0,4,3,2,1],[0,3,4,1,2]]
        #label[5], label[4] = label[4], label[5]#because these are not self inverse and are instead inverse of each other. Inverseis what is needed here
    PofC=[coordiates[p] for p in permutation]
    R=[(c-delta)%L for c in PofC]
    found=0
    for f,name in zip([ide,rot90,rot180,rot270,mirrorx,mirrory,mirrorxy,mirror_xy], label):
        if (f(R[1],L)==coordiates[1]).all() and (f(R[L],L)==coordiates[L]).all():
            if test:
                print(name,np.linalg.norm([(f(R[i],L)-coordiates[i]) for i in range(len(R))]))
            found=1;
        if found:
            break
    if not found:
        #zprint(R[1],R[L],R[0])
        raise ValueError("No decomposition found")
    return (name, coordiates[permutation[0]])

    
    

class GCNN_FFT_p4m(nn.Module):
    
    r"""Implements a GCNN using a fast fourier transform over the translation group.

    The group convolution can be written in terms of translational convolutions with
    symmetry transformed filters as described in ` Cohen et. *al* <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    The translational convolutions are then implemented with Fast Fourier Transforms.
    """
    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the network should be equivariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    D4: HashableArray
    product_table: HashableArray
    """Product table describing the algebra of the symmetry group
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    shape: tuple
    """Shape of the translation group"""
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: tuple
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    characters: HashableArray
    """Array specifying the characters of the desired symmetry representation"""
    param_dtype: Any = float
    """The dtype of the weights."""
    activation: Any = reim_selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = identity
    """The nonlinear activation before the output. Defaults to the identity."""
    input_mask: Array = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used."""
    hidden_mask: Array = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
    equal_amplitudes: bool = False
    """If true forces all basis states to have the same amplitude by setting `Re[logψ] = 0`"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_gcnn_initializer
    """Initializer for the kernels of all layers."""
    bias_init: NNInitFunc = zeros
    """Initializer for the biases of all layers."""
    complex_output: bool = True
    """Use complex-valued `logsumexp`. Necessary when parameters are real but some
    `characters` are negative."""


    def setup(self):
        self.n_symm = np.asarray(self.symmetries).shape[0]
        
        self.dense_symm = DenseSymmFFT_p4m(
            space_group=self.symmetries,
            D4=self.D4,
            shape=self.shape,
            features=self.features[0],
            param_dtype=self.param_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
            mask=self.input_mask,
        )

        self.equivariant_layers = [
            DenseEquivariantFFT(
                product_table=self.product_table,
                shape=self.shape,
                features=self.features[layer + 1],
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                mask=self.hidden_mask,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x):
        #print("GCNN_FFT_p4m")

        if x.ndim < 3:
            x = jnp.expand_dims(x, -2)  # add a feature dimension
        x = self.dense_symm(x)

        for layer in range(self.layers - 1):
            x = self.activation(x)
            x = self.equivariant_layers[layer](x)

        x = self.output_activation(x)

        if self.complex_output:
            x = logsumexp_cplx(x, axis=(-2, -1), b=jnp.asarray(self.characters))
        else:
            x = logsumexp(x, axis=(-2, -1), b=jnp.asarray(self.characters))

        if self.equal_amplitudes:
            return 1j * jnp.imag(x)
        else:
            return x

def GCNN(
    symmetries=None,
    product_table=None,
    irreps=None,
    point_group=None,
    mode="auto",
    shape=None,
    layers=None,
    features=None,
    characters=None,
    parity=None,
    param_dtype=np.float64,
    complex_output=True,
    input_mask=None,
    hidden_mask=None,
    **kwargs,
):
    r"""Implements a Group Convolutional Neural Network (G-CNN) that outputs a wavefunction
    that is invariant over a specified symmetry group.

    The G-CNN is described in `Cohen et al. <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in `Roth et al. <https://arxiv.org/pdf/2104.05085.pdf>`_ .

    The G-CNN alternates convolution operations with pointwise non-linearities. The first
    layer is symmetrized linear transform given by DenseSymm, while the other layers are
    G-convolutions given by DenseEquivariant. The hidden layers of the G-CNN are related by
    the following equation:

    .. math ::

        {\bf f}^{i+1}_h = \Gamma( \sum_h W_{g^{-1} h} {\bf f}^i_h).


    Args:
        symmetries: A specification of the symmetry group. Can be given by a
            :class:`netket.graph.Graph`, a :class:`netket.utils.group.PermutationGroup`, or an
            array :code:`[n_symm, n_sites]` specifying the permutations
            corresponding to symmetry transformations of the lattice.
        product_table: Product table describing the algebra of the symmetry group.
            Only needs to be specified if mode='fft' and symmetries is specified as an array.
        irreps: List of 3D tensors that project onto irreducible representations of the symmetry group.
            Only needs to be specified if mode='irreps' and symmetries is specified as an array.
        point_group: The point group, from which the space group is built. If symmetries is a
            graph the default point group is overwritten.
        mode: string "fft, irreps, matrix, auto" specifying whether to use a fast
            fourier transform over the translation group, a fourier transform using
            the irreducible representations or by constructing the full kernel matrix.
        shape: A tuple specifying the dimensions of the translation group.
        layers: Number of layers (not including sum layer over output).
        features: Number of features in each layer starting from the input. If a single
            number is given, all layers will have the same number of features.
        characters: Array specifying the characters of the desired symmetry representation.
        parity: Optional argument with value +/-1 that specifies the eigenvalue
            with respect to parity (only use on two level systems).
        param_dtype: The dtype of the weights.
        activation: The nonlinear activation function between hidden layers. Defaults to
            :class:`netket.nn.activation.reim_selu` .
        output_activation: The nonlinear activation before the output.
        equal_amplitudes: If True forces all basis states to have equal amplitude
            by setting :math:`\Re(\psi) = 0` .
        use_bias: If True uses a bias in all layers.
        precision: Numerical precision of the computation see :class:`jax.lax.Precision` for details.
        kernel_init: Initializer for the kernels of all layers. Defaults to
            :code:`lecun_normal(in_axis=1, out_axis=0)` which guarantees the correct variance of the
            output. See the documentation of :func:`flax.linen.initializers.lecun_normal`
            for more information.
        bias_init: Initializer for the biases of all layers.
        complex_output: If True, ensures that the network output is always complex.
            Necessary when network parameters are real but some `characters` are negative.
        input_mask: Optional array of shape :code:`(n_sites,)` used to restrict the convolutional
            kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
            boolean mask should be used.
        hidden_mask: Optional array of shape :code:`(n_symm,)` where
            :code:`(n_symm,) = len(graph.automorphisms())` used to restrict the convolutional
            kernel. Only parameters with mask :math:'\ne 0' are used.
            For best performance a boolean mask should be used.

    """

    if input_mask is not None:
        input_mask = HashableArray(input_mask)
    if hidden_mask is not None:
        hidden_mask = HashableArray(hidden_mask)

    if isinstance(symmetries, Lattice) and (
        point_group is not None or symmetries._point_group is not None
    ):
        # With graph try to find point group, otherwise default to automorphisms
        shape = tuple(symmetries.extent)
        sg = symmetries.space_group(point_group)
        if mode == "auto":
            mode = "fft"
    elif isinstance(symmetries, Graph):
        sg = symmetries.automorphisms()
        if mode == "auto":
            mode = "irreps"
        if mode == "fft":
            raise ValueError(
                "When requesting 'mode=fft' a valid point group must be specified"
                "in order to construct the space group"
            )
    elif isinstance(symmetries, PermutationGroup):
        # If we get a group and default to irrep projection
        if mode == "auto":
            mode = "irreps"
        sg = symmetries
    else:
        if irreps is not None and (mode == "irreps" or mode == "auto"):
            mode = "irreps"
            sg = symmetries
            irreps = tuple(HashableArray(irrep) for irrep in irreps)
        elif product_table is not None and (mode == "fft" or mode == "auto"):
            mode = "fft"
            sg = symmetries
            product_table = HashableArray(product_table)
        else:
            raise ValueError(
                "Specification of symmetries is wrong or incompatible with selected mode"
            )

    if mode == "fft" or mode == "p4m":
        if shape is None:
            raise TypeError(
                "When requesting `mode=fft`, the shape of the translation group must be specified. "
                "Either supply the `shape` keyword argument or pass a `netket.graph.Graph` object to "
                "the symmetries keyword argument."
            )
        else:
            shape = tuple(shape)

    if isinstance(features, int):
        features = (features,) * layers

    if characters is None:
        characters = HashableArray(np.ones(len(np.asarray(sg))))
    else:
        if (
            not jnp.iscomplexobj(characters)
            and not is_complex_dtype(param_dtype)
            and not complex_output
            and jnp.any(characters < 0)
        ):
            raise ValueError(
                "`complex_output` must be used with real parameters and negative "
                "characters to avoid NaN errors."
            )
        characters = HashableArray(characters)

    if mode == "fft":
        sym = HashableArray(np.asarray(sg))
        print(sym.shape)
        print(shape)
        if product_table is None:
            product_table = HashableArray(sg.product_table)
        if parity:
            return GCNN_Parity_FFT(
                symmetries=sym,
                product_table=product_table,
                layers=layers,
                features=features,
                characters=characters,
                shape=shape,
                parity=parity,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )
        else:
            return GCNN_FFT(
                symmetries=sym,
                product_table=product_table,
                layers=layers,
                features=features,
                characters=characters,
                shape=shape,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )


    elif mode == "p4m":
        print("p4m mode")
        sym = HashableArray(np.asarray(sg))
        
        sga=np.asarray(sg)
        ncells=np.prod(shape)
        D4=[D4xTDecomposition(sga[i],shape[0],"list")[0] for i in range(0,np.shape(sga)[0]//ncells)]
        D4=HashableArray(np.array(D4).astype(np.float64))

        if product_table is None:
            product_table = HashableArray(sg.product_table)
        if parity:
            return GCNN_Parity_FFT(
                symmetries=sym,
                product_table=product_table,
                layers=layers,
                features=features,
                characters=characters,
                shape=shape,
                parity=parity,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )
        else:
            print("GCNN_FFT_p4m!")
            return GCNN_FFT_p4m(
                symmetries=sym,
                D4=D4,
                product_table=product_table,
                layers=layers,
                features=features,
                characters=characters,
                shape=shape,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )

    elif mode in ["irreps", "auto"]:
        sym = HashableArray(np.asarray(sg))

        if irreps is None:
            irreps = tuple(HashableArray(irrep) for irrep in sg.irrep_matrices())

        if parity:
            return GCNN_Parity_Irrep(
                symmetries=sym,
                irreps=irreps,
                layers=layers,
                features=features,
                characters=characters,
                parity=parity,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )
        else:
            return GCNN_Irrep(
                symmetries=sym,
                irreps=irreps,
                layers=layers,
                features=features,
                characters=characters,
                param_dtype=param_dtype,
                complex_output=complex_output,
                hidden_mask=hidden_mask,
                input_mask=input_mask,
                **kwargs,
            )
    else:
        raise ValueError(
            f"Unknown mode={mode}. Valid modes are 'fft',irreps' or 'auto'."
        )



