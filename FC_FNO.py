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
        # for i in range(n_layers):
        #     self.fno_blocks.convs[i].weight = self.fno_blocks.convs[i].weight.to(torch.cdouble)

        self.FC_obj = FC_Legendre()
        self.projection_nonlinearity = projection_nonlinearity
        self.Lengths = Lengths

        ## Linear projection MLP
        self.projection = LinearChannelMLP(
            layers=[hidden_channels, self.projection_channels, out_channels],
            non_linearity=self.projection_nonlinearity,
        )


    #    #make the entire model to float64 for extra precision in calculations
    #     self.double()
       

    def dQ_3D(self, X1, Dx_arr, Q1, Q2, derivs_to_compute):
            """
            Chain rule of model, call it Q, to compute dQ/d(inputs). Uses einsum for efficiency.

            ***** Assumes that the final nonlinearity is tanh. ******

            Gradient chain rule: D(f o g) = D(f(g)) o Dg

            Hessian chain rule: D2(f o g) = D2(f(g)) o Dg^2 + D(f(g)) o D2g

            As implemented, Q1 is the first projection layer of the model, and Q2 is the last projection layer.

            You can generalize this to higher dimensions by using the chain rule for higher derivatives.

            In the 3rd derivative case, the  chain rule is:
            D3(f o g) = D3(f(g)) o Dg^3 + 3 * D2(f(g)) o Dg * D2g + D(f(g)) o D3g

            In the general case, the chain rule is:
            D^n(f o g) = summation(k=0 to n-1) (n-1 choose k) * D^(k+1)g(x)D^(n-k)(f'(g(x)))


            """

            # EINSUM DIMENSION KEY:
            # B: batch size
            # C: hidden/intermediate dimension
            # I: input dimension
            # O: output dimension
            # T: time dimension
            # X: spatial dimension (x-coordinate)
            # Z: spatial dimension (z-coordinate, equivalent to y in 2D)

            # TENSOR SHAPES:
            # dW1: first projection layer weights (C, I)
            # dW2: final projection layer weights (O, C) - transposed from (C, O)
            # X1: intermediate representation after first projection (B, C, T, X, Z)
            # Dtanh: derivative of tanh activation (B, C, T, X, Z)
            # dQ: Jacobian of the composed function (B, O, I, T, X, Z)
            # H2: Hessian of final projection at intermediate point (B, O, I, T, X, Z)
            # dx, dy, dt: spatial and time derivatives (B, I, T, X, Z)
            # dxx, dyy: second spatial derivatives (B, I, T, X, Z)

            Dx_out = []

            need_dx = "dx" in derivs_to_compute or "dxx" in derivs_to_compute
            need_dy = "dy" in derivs_to_compute or "dyy" in derivs_to_compute
            need_dz = "dz" in derivs_to_compute or "dzz" in derivs_to_compute
            
            dx = Dx_arr["dx"] if need_dx else None
            dy = Dx_arr["dy"] if need_dy else None
            dz = Dx_arr["dz"] if need_dz else None

            dxx = Dx_arr["dxx"] if "dxx" in derivs_to_compute else None
            dyy = Dx_arr["dyy"] if "dyy" in derivs_to_compute else None
            dzz = Dx_arr["dzz"] if "dzz" in derivs_to_compute else None

            X1 = X1.permute(0, 4, 1, 2, 3)

            B, C, T, X, Z = X1.shape
            I = self.hidden_channels
            O = self.out_channels

            #########################################################
            ### First Derivative: D(f∘g) = Df(g) · Dg (Chain Rule)
            #########################################################

            # dW1: weights of the first projection layer Q1
            dW1 = Q1.weight

            # dW2: weights of the final projection layer Q2
            dW2 = Q2.weight.t()

            # Dtanh: derivative of tanh activation function
            if self.projection_nonlinearity == F.tanh:
                dP1 = 1 / torch.cosh(X1) ** 2  # (B, C, T, X, Z)
            elif self.projection_nonlinearity == F.silu:
                dP1 = torch.sigmoid(X1) * (1 + X1 * (1 - torch.sigmoid(X1)))
            else:
                raise ValueError(
                    f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be tanh or sigmoid"
                )

            # Compute Jacobian dQ = D(f∘g) using chain rule
            dQ = torch.einsum("ci, bctxz, co -> boitxz", dW1, dP1, dW2)

            # wxQ: spatial derivative of output in x-direction
            if "dx" in derivs_to_compute:
                wxQ = torch.einsum("boitxz,bitxz->botxz", dQ, dx)
                Dx_out.append(wxQ)

            # wyQ: spatial derivative of output in y-direction
            if "dy" in derivs_to_compute:
                wyQ = torch.einsum("boitxz,bitxz->botxz", dQ, dy)
                Dx_out.append(wyQ)

            # wzQ: time derivative of output
            if "dz" in derivs_to_compute:
                wzQ = torch.einsum("boitxz,bitxz->botxz", dQ, dz)
                Dx_out.append(wzQ)


            

            ##############################################################
            ### Second Derivative: D²(f∘g) = D²f(g) · (Dg)² + Df(g) · D²g
            ##############################################################

            # Htanh: second derivative of tanh activation function
            # Htanh = -2 * Dtanh * tanh(X1) = -2 * sech²(X1) * tanh(X1)
            if self.projection_nonlinearity == F.tanh:
                dP2 = -2 * dP1 * torch.tanh(X1)
            elif self.projection_nonlinearity == F.silu:
                dP2 = (
                    torch.sigmoid(X1)
                    * (1 - torch.sigmoid(X1))
                    * (2 + X1 * (1 - 2 * torch.sigmoid(X1)))
                )
            else:
                raise ValueError(
                    f"Projection nonlinearity {self.projection_nonlinearity} not supported. Must be tanh or sigmoid"
                )

            # H2: Hessian of the final projection layer at the intermediate point
            # H2 = D²f(g) where f is the final projection layer
            H2 = torch.einsum("co,bctxz->bcotxz", dW2, dP2)  # (B, C, O, T, X, Z)

            # Compute second derivative in x-direction using chain rule
            # wxx1: first term of chain rule: J_g^T · H_f · J_g
            if "dxx" in derivs_to_compute:
                wxx1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dx, dW1, H2, dW1, dx)
                # wxx2: second term of chain rule: Df(g) · D²g
                wxx2 = torch.einsum("boitxz,bitxz->botxz", dQ, dxx)
                # Combine both terms: D²(f∘g) = D²f(g) · (Dg)² + Df(g) · D²g
                wxxQ = wxx1 + wxx2
                Dx_out.append(wxxQ)
            # Compute second derivative in z-direction using chain rule
            # wzz1: first term of chain rule: J_g^T · H_f · J_g
            if "dyy" in derivs_to_compute:
                wyy1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dy, dW1, H2, dW1, dy)
                wyy2 = torch.einsum("boitxz,bitxz->botxz", dQ, dyy)
                wyyQ = wyy1 + wyy2
                Dx_out.append(wyyQ)
            if "dzz" in derivs_to_compute:
                wzz1 = torch.einsum("bitxz,ci,bcotxz,cj,bjtxz->botxz", dz, dW1, H2, dW1, dz)
                wzz2 = torch.einsum("boitxz,bitxz->botxz", dQ, dzz)
                wzzQ = wzz1 + wzz2
                Dx_out.append(wzzQ)

            # Return first and second derivatives as dictionary

            print("second derivatives calculated")

            return Dx_out
    


    ## say if we want derivs to compute in the forward
    def forward(self, x, output_shape=None, **kwargs):
       
        """FC_FNO's forward pass"""

        derivs_to_compute = {"dx", "dxx", "dy", "dyy", "dz", "dzz"} #correspond to grid axis: time frames, height and width

        print("THE SHAPE IS ", x.shape)

        # if (x.dtype != torch.float64):
        #     x = x.to(torch.float64)

        
        output_shape = [None] * self.n_layers

        # ==================== EXTENSION OPERATION ===============================
        if self.n_dim == 3:
            __, __, x_res, y_res, z_res = x.shape
            original = x
            x = self.FC_obj.extend_signal(x)

            # self.FC_obj.plot_results(original, x)
            
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
    



        new_Lengths = (
            self.Lengths[0] * (x_res + self.FC_obj.get_n_additional_points()) / x_res,
            self.Lengths[1] * (y_res + self.FC_obj.get_n_additional_points()) / y_res,
            self.Lengths[2] * (z_res + self.FC_obj.get_n_additional_points()) / z_res
        )

        FourierDiff3d = FourierDiff(dim=self.n_dim, L=new_Lengths)

        dz_tuple = (0, 0, 1)
        dy_tuple = (0, 1, 0)
        dx_tuple = (1, 0, 0)

        dzz_tuple = (0,0,2)
        dyy_tuple = (0,2,0)
        dxx_tuple = (2,0,0)

        Dx_arr = {}
        deriv_tuples = [dz_tuple, dy_tuple, dx_tuple, dxx_tuple, dyy_tuple, dzz_tuple]

        deriv_array = FourierDiff3d.compute_multiple_derivatives(x, derivatives=deriv_tuples)


        for i, deriv_tuple in enumerate(deriv_tuples):
            if deriv_tuple == (0, 0, 1):
                Dx_arr["dz"] = self.FC_obj.restrict_signal(deriv_array[i])
            elif deriv_tuple == (0, 1, 0):
                Dx_arr["dy"] = self.FC_obj.restrict_signal(deriv_array[i])
            elif deriv_tuple == (1, 0, 0):
                Dx_arr["dx"] = self.FC_obj.restrict_signal(deriv_array[i])
            elif deriv_tuple == (2, 0, 0):
                Dx_arr["dxx"] = self.FC_obj.restrict_signal(deriv_array[i])
            elif deriv_tuple == (0, 2, 0):
                Dx_arr["dyy"] = self.FC_obj.restrict_signal(deriv_array[i])
            elif deriv_tuple == (0, 0, 2):
                Dx_arr["dzz"] = self.FC_obj.restrict_signal(deriv_array[i])



        print("computed derivatives")


        Q1 = self.projection.fcs[0]
        Q2 = self.projection.fcs[-1]



        x = self.FC_obj.restrict_signal(x)


        X1 = Q1(x.permute(0,2,3,4,1))
        Dx_arr = self.dQ_3D(X1, Dx_arr, Q1, Q2, derivs_to_compute)

        

      
        # ==================== PROJECTION OPERATION ===============================

        
        x = self.projection(x.permute(0, 2, 3, 4, 1))
        x = x.permute(0, 4, 1, 2, 3)


        return x, Dx_arr
        
        # ==================== PROJECTION OPERATION ===============================

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


