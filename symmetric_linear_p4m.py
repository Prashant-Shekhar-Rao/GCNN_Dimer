# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, cast

import numpy as np
import jax.numpy as jnp

import jax
from jax import lax
from jax.nn.initializers import zeros, lecun_normal
from flax.linen.module import Module, compact
from flax.linen.dtypes import promote_dtype

from netket.utils import HashableArray, warn_deprecation
from netket.utils.types import Array, DType, NNInitFunc
from netket.utils.group import PermutationGroup
from collections.abc import Sequence
from netket.graph import Graph, Lattice
from netket.errors import SymmModuleInvalidInputShape

# All layers defined here have kernels of shape [out_features, in_features, n_symm]
default_equivariant_initializer = lecun_normal(in_axis=1, out_axis=0)


    

class DenseSymmMatrix(Module):
    
    r"""Implements a symmetrized linear transformation over a permutation group
    using matrix multiplication."""

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    features: int
    """The number of output features. Will be the second dimension of the output."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    param_dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.n_symm, self.n_sites = np.asarray(self.symmetries).shape
        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        
        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        in_features = x.shape[-2]

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_sites], self.param_dtype
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_sites),
                self.param_dtype,
            )
        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)

        # Converts the convolutional kernel of shape (self.features, in_features, n_sites)
        # to a full dense kernel of shape (self.features, in_features, n_symm, n_sites).
        # result[out, in, g, r] == kernel[out, in, g^{-1}r]
        kernel = jnp.take(kernel, jnp.asarray(self.symmetries), 2)

        # x is      (batches,       in_features,         n_sites)
        # kernel is (self.features, in_features, n_symm, n_sites)
        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (1, 3)), ((), ())),
            precision=self.precision,
        )
        
        

        if bias is not None:
            # Convert symmetry-reduced bias of shape (features,) to the full bias of
            # shape (..., features, 1).
            x += jnp.expand_dims(bias, 1)

        return x


class DenseSymmFFT(Module):
    
    r"""Implements a symmetrized projection onto a space group using a Fast Fourier Transform"""

    space_group: HashableArray
    """Array that lists the space group as permutations"""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        sg = np.asarray(self.space_group)

        self.n_cells = np.prod(np.asarray(self.shape))
        self.n_symm = len(sg)
        self.n_point = self.n_symm // self.n_cells
        self.sites_per_cell = sg.shape[1] // self.n_cells

        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

        # maps (n_sites) dimension of kernels to (sites_per_cell, n_point, *shape)
        # as used in FFT-based group convolution
        self.mapping = (
            sg[:, : self.sites_per_cell]
            .reshape(self.n_cells, self.n_point, self.sites_per_cell)
            .transpose(2, 1, 0)
            .reshape(self.sites_per_cell, self.n_point, *self.shape)
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        in_features = x.shape[-2]
        batch_shape = x.shape[:-2]

        x = x.reshape(-1, in_features, self.n_cells, self.sites_per_cell)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_cells * self.sites_per_cell],
                self.param_dtype,
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_cells * self.sites_per_cell),
                self.param_dtype,
            )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)
        dtype = x.dtype

        # Converts the convolutional kernel of shape (features, in_features, n_sites)
        # to the expanded kernel of shape (features, in_features, sites_per_cell,
        # n_point, *shape) used in FFT-based group convolutions.

        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        # TODO: the batch ordering should be revised: batch dimensions should
        # be leading
        x = lax.dot_general(
            x, kernel, (((1, 2), (1, 2)), ((3,), (4,))), precision=self.precision
        )
        x = x.transpose(1, 2, 3, 0)
        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*batch_shape, self.features, self.n_symm)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseSymmFFT_p4m(Module):
    
    print("DenseSymmFFT")
    r"""Implements a symmetrized projection onto a space group using a Fast Fourier Transform"""

    space_group: HashableArray
    """Array that lists the space group as permutations"""
    D4: HashableArray
    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_sites,)` used to restrict the convolutional
        kernel. Only parameters with mask :math:'\ne 0' are used. For best performance a
        boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        sg = np.asarray(self.space_group)

        self.n_cells = np.prod(np.asarray(self.shape))
        self.n_symm = len(sg)
        self.n_point = self.n_symm // self.n_cells
        self.sites_per_cell = sg.shape[1] // self.n_cells

        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

        # maps (n_sites) dimension of kernels to (sites_per_cell, n_point, :*shape)
        # as used in FFT-based group convolution
        self.mapping = (
            sg[:, : self.sites_per_cell]
            .reshape(self.n_cells, self.n_point, self.sites_per_cell)
            .transpose(2, 1, 0)
            .reshape(self.sites_per_cell, self.n_point, *self.shape)
        ) 
        

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        # ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            raise SymmModuleInvalidInputShape("DenseSymmMatrix", x)

        in_features = x.shape[-2]##order is (batch,infeatures,L^2,sitespercell)
        batch_shape = x.shape[:-2]
        
        x = x.reshape(-1, in_features, self.n_cells, self.sites_per_cell)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)
        #shape is (totalbatch,infeatures,sitespercell,L,L)
        
        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_cells * self.sites_per_cell],
                self.param_dtype,
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_cells * self.sites_per_cell),
                self.param_dtype,
            )


        
        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)
        dtype = x.dtype



        # Converts the convolutional kernel of shape (features, in_features, n_sites)
        # to the expanded kernel of shape (features, in_features, sites_per_cell,
        # n_point, *shape) used in FFT-based group convolutions.

        kernel = kernel[..., self.mapping]

        #x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        D4=np.asarray(self.D4)
        x=jnp.stack([jnp.take(D4[i], x.astype(int)) for i in range(self.n_point)])
        #shape (npoint,totalbatch,infeatures,sitespercell, L,L)
        x=x.transpose(1,2,3,0,4,5)
        #shape (totalbatch,infeatures,sitespercell, npoint, L,L)
        x=jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:4],self.n_cells)
        #fourier transform of the coordinates.

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        # TODO: the batch ordering should be revised: batch dimensions should
        # be leading
        #x = lax.dot_general(
        #    x, kernel, (((1, 2), (1, 2)), ((3,), (4,))), precision=self.precision
        #)
        x = lax.dot_general(x, kernel, (((1, 2), (1, 2)), ((3,4), (3,4))), precision=self.precision)

        #x = x.transpose(1, 2, 3, 0)
        x=x.transpose(2,3,0,1)

        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*batch_shape, self.features, self.n_symm)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real

class DenseEquivariantFFT(Module):
    r"""Implements a group convolution using a fast fourier transform over the translation group.

    The group convolution can be written in terms of translational convolutions with
    symmetry transformed filters as described in
    `Cohen et. al <http://proceedings.mlr.press/v48/cohenc16.pdf>_`

    The translational convolutions are then implemented with Fast Fourier Transforms.
    """

    product_table: HashableArray
    """Product table for space group."""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        pt = np.asarray(self.product_table)

        self.n_symm = len(pt)
        self.n_cells = np.prod(np.asarray(self.shape))
        self.n_point = self.n_symm // self.n_cells
        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

        # maps (n_sites) dimension of kernels to (n_point, n_point, *shape)
        # as used in FFT-based group convolution
        self.mapping = (
            pt[: self.n_point]
            .reshape(self.n_point, self.n_cells, self.n_point)
            .transpose(0, 2, 1)
            .reshape(self.n_point, self.n_point, *self.shape)
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        in_features = x.shape[-2]
        batch_shape = x.shape[:-2]

        x = x.reshape(-1, in_features, self.n_cells, self.n_point)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_point * self.n_cells],
                self.param_dtype,
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_point * self.n_cells),
                self.param_dtype,
            )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)
        dtype = x.dtype

        # Convert the convolutional kernel of shape (features, in_features, n_symm)
        # to the expanded kernel of shape (features, in_features, n_point(in),
        # n_point(out), *shape) used in FFT-based group convolutions
        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        x = lax.dot_general(
            x, kernel, (((1, 2), (1, 2)), ((3,), (4,))), precision=self.precision
        )
        x = x.transpose(1, 2, 3, 0)
        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*batch_shape, self.features, self.n_symm)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseEquivariantIrrep(Module):
    r"""Implements a group convolutional layer by projecting onto irreducible
    representations of the group.

    Acts on a feature map of shape [batch_size, in_features, n_symm] and
    returns a feature map of shape [batch_size, features, n_symm].
    The input and the output are related by

    .. math ::

        y^{(i)}_g = \sum_{h,j} f^{(j)}_h W^{(ij)}_{h^{-1}g}.

    Note that this switches the convention of Cohen et al. to use an actual group
    convolution, but this doesn't affect equivariance.
    The convolution is implemented in terms of a group Fourier transform.
    Therefore, the group structure is represented internally as the set of its
    irrep matrices. After Fourier transforming, the convolution translates to

    .. math ::

        y^{(i)}_\rho = \sum_j f^{(j)}_\rho W^{(ij)}_\rho,

    where all terms are d x d matrices rather than numbers, and the juxtaposition
    stands for matrix multiplication.
    """

    irreps: tuple[HashableArray, ...]
    """Irrep matrices of the symmetry group. Each element of the list is an
    array of shape [n_symm, d, d]; irreps[i][j] is the representation of the
    jth group element in irrep #i."""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""

    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        self.n_symm = self.irreps[0].shape[0]
        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

        self.forward = jnp.concatenate(
            [jnp.asarray(irrep).reshape(self.n_symm, -1) for irrep in self.irreps],
            axis=1,
        )
        self.inverse = jnp.concatenate(
            [
                jnp.asarray(irrep).conj().reshape(self.n_symm, -1)
                * (irrep.shape[-1] / self.n_symm)
                for irrep in self.irreps
            ],
            axis=1,
        ).transpose()

        # Convert between vectors of length n_symm and tuples of arrays of shape
        # n_irrep × irrep_size^2
        self.assemble = lambda arrays: jnp.concatenate(
            [array.reshape(array.shape[:-3] + (-1,)) for array in arrays], axis=-1
        )

        irrep_size = 1
        n_same_size = 0
        shapes = []
        for irrep in self.irreps:
            if irrep_size == irrep.shape[-1]:
                n_same_size += 1
            else:
                shapes.append((n_same_size, irrep_size, irrep_size))
                irrep_size = irrep.shape[-1]
                n_same_size = 1
        shapes.append((n_same_size, irrep_size, irrep_size))
        limits = np.cumsum([0] + [np.prod(shape) for shape in shapes])

        self.disassemble = lambda vecs: tuple(
            vecs[..., limits[i] : limits[i + 1]].reshape(vecs.shape[:-1] + shape)
            for i, shape in enumerate(shapes)
        )

    def forward_ft(self, inputs: Array) -> tuple[Array, ...]:
        r"""Performs a forward group Fourier transform on the input.
        This is defined by

        .. math ::

            \hat{f}_\rho = \sum_g f(g) \rho(g),

        where :math:`\rho` is an irrep of the group.
        The Fourier transform is performed over the last index, and is returned
        as a tuple of arrays, each entry corresponding to the entry of `irreps`
        in the same position, and the last dimension of length `n_symm` replaced
        by two dimensions of length `d` each.
        """
        return self.disassemble(jnp.tensordot(inputs, self.forward, axes=1))

    def inverse_ft(self, inputs: tuple[Array, ...]) -> Array:
        r"""Performs an inverse group Fourier transform on the input.
        This is defined by

        .. math ::

            f(g) = \frac{1}{|G|} \sum_\rho d_\rho {\rm Tr}(\rho(g^{-1}) \hat{f}_\rho)

        where the sum runs over all irreps of the group.
        The input is a tuple of arrays whose the last two dimensions match the
        dimensions of each irrep. The inverse Fourier transform is performed
        over these indices and is returned as an array where those dimensions
        are replaced by a single dimension of length `n_symm`
        """
        return jnp.asarray(
            jnp.tensordot(self.assemble(inputs), self.inverse, axes=1),
            # Irrep matrices might be complex, so `result` might be complex
            # even if the inputs are real
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        in_features = x.shape[-2]
        batch_shape = x.shape[:-2]
        x = x.reshape(-1, in_features, self.n_symm)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_symm], self.param_dtype
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_symm),
                self.param_dtype,
            )

        x, kernel, bias = promote_dtype(x, kernel, bias, dtype=None)
        dtype = x.dtype

        x_fourier = self.forward_ft(x)

        kernel_fourier = self.forward_ft(kernel)

        x_fourier = tuple(
            lax.dot_general(
                x_fourier[i], kernel_fourier[i], (((1, 4), (1, 3)), ((2,), (2,)))
            ).transpose(1, 3, 0, 2, 4)
            for i in range(len(x_fourier))
        )

        x = self.inverse_ft(x_fourier).reshape(*batch_shape, self.features, self.n_symm)
        x = cast(jax.Array, x)

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseEquivariantMatrix(Module):
    r"""Implements a group convolution operation that is equivariant over a symmetry group
    by multiplying by the full kernel matrix"""

    product_table: HashableArray
    """Product table for the space group."""
    features: int
    """The number of symmetry-reduced output features. The full output size
    is n_symm*features."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: HashableArray | None = None
    """Optional array of shape `(n_symm,)` where `(n_symm,)` = `len(graph.automorphisms())`
        used to restrict the convolutional kernel. Only parameters with mask :math:'\ne 0' are used.
        For best performance a boolean mask should be used"""
    param_dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        self.n_symm = np.asarray(self.product_table).shape[0]
        if self.mask is not None:
            (self.kernel_indices,) = np.nonzero(self.mask.wrapped)

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        in_features = x.shape[-2]

        kernel: jax.Array
        if self.mask is not None:
            kernel_params = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, len(self.kernel_indices)),
                self.param_dtype,
            )

            kernel = jnp.zeros(
                [self.features, in_features, self.n_symm], self.param_dtype
            )
            kernel = kernel.at[:, :, self.kernel_indices].set(kernel_params)
        else:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.features, in_features, self.n_symm),
                self.param_dtype,
            )

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel, bias, x = promote_dtype(kernel, bias, x, dtype=None)

        # Converts the convolutional kernel of shape (features, in_features, n_symm)
        # to a full dense kernel of shape (features, in_features, n_symm, n_symm)
        # result[out, in, g, h] == kernel[out, in, g^{-1}h]
        # input dimensions are [in, g], output dimensions are [out, h]
        kernel = jnp.take(kernel, jnp.asarray(self.product_table), 2)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (1, 2)), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            x += jnp.expand_dims(bias, 1)

        return x

