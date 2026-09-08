import torch
import matplotlib.pyplot as plt
import numpy
from matplotlib.lines import Line2D
from neuralop.layers.fourier_continuation import FCLegendre, FCGram


class FC_Legendre():

    def __init__(self) -> None:
        self.no_ext_points = 14 #defines number of points to add on extension
        self.extend_legendre = FCLegendre(d=3, n_additional_pts=self.no_ext_points)
   

    def extend_signal(self, data: torch.tensor):
        # periodic_signal = self.extend_legendre(data, dim=2)

        periodic_signal_xy = self.extend_legendre(data, dim = 3)

        return periodic_signal_xy
    
    def restrict_signal(self, ext_data: torch.tensor):

        signal_xy = self.extend_legendre.restrict(ext_data, dim=3)

        return signal_xy
    
    def get_n_additional_points(self):
        if self.extend_legendre:
            return self.extend_legendre.n_additional_pts or self.no_ext_points

        
    def plot_results(self, original_data, new_data):
        c = self.no_ext_points // 2 

        od = original_data[0, 0, 0].cpu().numpy()
        nd = new_data[0, 0, c].cpu().numpy()
        H, W = od.shape

        vmin, vmax = od.min(), od.max()


        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(od, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title("Original")


        axs[1].imshow(nd, cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].add_patch(plt.Rectangle((c, c), W, H, edgecolor='red', facecolor='none', lw=1))
        axs[1].set_title("Continuation (red box = original region)")
        plt.show()