def visualise_psuedo(model_path, AutoEncoder,model_descrip,execution_profile):
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from numpy.linalg import pinv
    import matplotlib.colors as mcolors
    if execution_profile == "default":
        state_dict = torch.load(model_path)

        # Instantiate a new model of the same architecture
        model_loaded = AutoEncoder()

        # Load the state dictionary into the model
        model_loaded.load_state_dict(state_dict)

        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs = axs.flatten()

    
        weights_1 = model_loaded.layer1.weight.data.numpy()
        weights_2 = model_loaded.LNL_model.weight.data.numpy()
        weights_3 = model_loaded.layer3.weight.data.numpy()

        forward_model = np.dot(weights_2, weights_1)
        reverse_model = pinv(weights_3)
        cmap_custom = mcolors.LinearSegmentedColormap.from_list('red_blue', ['blue', 'white', 'red'])
        norm_custom_forw = mcolors.TwoSlopeNorm(vmin=forward_model.min(), vcenter=0, vmax=forward_model.max())
        norm_custom_rev = mcolors.TwoSlopeNorm(vmin=reverse_model.min(), vcenter=0, vmax=reverse_model.max())


        img = axs[0].imshow(forward_model, aspect='equal',cmap=cmap_custom,norm=norm_custom_forw)
        axs[0].set_title('Forward Model Weights')
        axs[0].set_xlabel('Pixels')
        axs[0].set_ylabel('Neurons')
        divider = make_axes_locatable(axs[0])

        # Append an axes to the right of the current axes with the same height
        cax = divider.append_axes("right", size="2%", pad=0.05)
            
            # Create the colorbar in the new axes
        fig.colorbar(img, cax=cax)  

        axs[1].imshow(reverse_model, aspect='equal',cmap=cmap_custom,norm=norm_custom_rev)
        axs[1].set_title('Psuedo Inverse Model Weights')
        axs[1].set_xlabel('Pixels')
        axs[1].set_ylabel('Neurons')
        divider = make_axes_locatable(axs[1])

        # Append an axes to the right of the current axes with the same height
        cax = divider.append_axes("right", size="2%", pad=0.05)
            
            # Create the colorbar in the new axes
        fig.colorbar(img, cax=cax)  

        

        plt.tight_layout()  # Add space for the title
        fig.savefig(f'{model_descrip}_forw_vs_inv.png', format='png',dpi=1200)
        plt.show()