import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partialmethod
from typing import Tuple, List, Union, Literal
Number = Union[float, int]



from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.channel_mlp import ChannelMLP, LinearChannelMLP
from neuralop.layers.complex import ComplexValued
from neuralop.models.base_model import BaseModel
from neuralop.layers.padding import DomainPadding
from neuralop.models.fno import FNO
from neuralop.losses.differentiation import *


from FC_Legendre import FC_Legendre

from functools import partialmethod
from typing import Tuple, List, Union, Literal
from functools import partial





class FC_FNO(FNO):
   

    _dQ_DOCSTRING = """
        Compute the model's derivatives via the chain rule.

        More specifically:
        --------
        1) We first compute spectral derivatives before the final projection. Denote these as dV/dx, where
        V is the pre-projection feature field and x is the input grid.
        2) We then use projection-layer derivatives (e.g., dQ/dV) and the chain tule to 
        compute the derivatives of the final model output with respect to input x.
        3) We use `einsum` to perform the required matrix multiplications efficiently.

        Notes
        -----
        - Derivatives are up to second order
        - Assumes the final nonlinearity is `tanh` or `sigmoid`

        Chain Rule Identities
        ---------------------
        Gradient:
            D(f ∘ g)(x) = Df(g(x)) · Dg(x)

        Hessian (2nd derevative):
            D²(f ∘ g)(x) = D²f(g(x))[Dg(x), Dg(x)] + Df(g(x)) · D²g(x)

        Higher-order:
            Dⁿ(f ∘ g)(x) = Σ_{k=0}^{n-1} C(n-1, k) · D^{k+1}g(x) · D^{n-k}(f'(g(x)))

        The same approach generalizes to higher dimensions and derivative orders by
        applying the corresponding multivariate chain rule. These idenities follow easily from
        the chain and product rule. 

        """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating"] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating"] = "linear",
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = dict(),
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        projection_nonlinearity=F.tanh,
        Lengths=Tuple[float],
    ):

        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
        )

        ## Convert FNO block weights to complex double precision for higher precision
        for i in range(n_layers):
            self.fno_blocks.convs[i].weight = self.fno_blocks.convs[i].weight.to(torch.cdouble)

        self.FC_obj = FC_Legendre()
        self.projection_nonlinearity = projection_nonlinearity
        self.Lengths = Lengths

        ## Linear projection MLP
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, self.projection_channels, out_channels],
            non_linearity=self.projection_nonlinearity,
        )


       
       
   

    ## say if we want derivs to compute in the forward
    def forward(self, x, output_shape=None, **kwargs):
       
        """FC_FNO's forward pass"""

        derivs_to_compute = {}

        print("THE SHAPE IS ", x.shape)

        # if (x.dtype != torch.float64):
        #     x = x.to(torch.float64)

        
        output_shape = [None] * self.n_layers

        # ==================== EXTENSION OPERATION ===============================
        if self.n_dim == 3:
            __, __, x_res, y_res, z_res = x.shape
            original = x
            x = self.FC_obj.extend_signal(x)
        else:
            raise ValueError(f"Error: expected 3 dimensions, got {self.n_dim}")
        # =========================================================================


        # append spatial pos embedding if set (extra features)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)


        #======================== LIFTING OPERATION ===============================
            
        x = self.lifting(x)

        #=========================================================================


        for layer_idx in range(self.n_layers):
            assert output_shape[layer_idx] is None, "Output shape must be None for FC_FNO"
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
    

      
        

        x = self.FC_obj.restrict_signal(x)

        print(x.shape)

            # ==================== PROJECTION OPERATION ===============================

        if self.n_dim == 3:
            x = self.projection(x.permute(0, 2, 3, 4, 1))
            x = x.permute(0, 4, 1, 2, 3)


        # x = x.to(torch.float32)

        return x
        
            # ==================== PROJECTION OPERATION ===============================

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


