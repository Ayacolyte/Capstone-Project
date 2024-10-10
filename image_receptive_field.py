
def visualize_img_recep(model_path, AutoEncoder, img_side_dim, elec_side_dim,model_descrip,show_fft,execution_profile):
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    state_dict = torch.load(model_path)

    # Instantiate a new model of the same architecture
    model_loaded = AutoEncoder()

    # Load the state dictionary into the model
    model_loaded.load_state_dict(state_dict)

    fig1, axs1 = plt.subplots(elec_side_dim, elec_side_dim,figsize=(25, 25))
    axs1 = axs1.flatten()
    if show_fft:
        fig2, axs2 = plt.subplots(elec_side_dim, elec_side_dim,figsize=(25, 25))
        axs2 = axs2.flatten()

    #recep_mem = []

    for i in range(elec_side_dim**2):
        if execution_profile == "CNN": 
            show_fft = False
            weights = model_loaded.layer1[0].weight.data.cpu().numpy()
            curr_recep = weights
            curr_recep = np.squeeze(curr_recep)
        elif execution_profile == "CNN_pool":
            #show_fft = False
            
            weights = model_loaded.layer1[0].weight.data.cpu().numpy()
            if i < weights.shape[0]:
                curr_recep = weights[i,:,:]
            else:
                curr_recep = np.zeros((8,8))
            curr_recep = np.squeeze(curr_recep)
        else:
            weights = model_loaded.layer1.weight.data.numpy()
            curr_recep = np.squeeze(weights[i,:])
            curr_recep = curr_recep.reshape(img_side_dim, img_side_dim)
        if show_fft:
            # Perform the 2D FFT
            img_fft = np.fft.fft2(curr_recep)

            # Shift the zero-frequency component to the center
            img_fft = np.fft.fftshift(img_fft)
            magnitude_spectrum = np.log(np.abs(img_fft) + 1)
            img = axs2[i].imshow(magnitude_spectrum, aspect='equal', cmap='gray')
            axs2[i].set_title(f'Electrode {i + 1}')
            #axs1[i].set_xlabel('Neurons')
            #axs1[i].set_ylabel('Input Features')
            axs2[i].set_xticks([])
            axs2[i].set_yticks([])
            divider = make_axes_locatable(axs2[i])

            # Append an axes to the right of the current axes with the same height
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            # Create the colorbar in the new axes
            fig2.colorbar(img, cax=cax)

        img = axs1[i].imshow(curr_recep, aspect='equal', cmap='gray')
        axs1[i].set_title(f'Electrode {i + 1}')
        #axs1[i].set_xlabel('Neurons')
        #axs1[i].set_ylabel('Input Features')
        axs1[i].set_xticks([])
        axs1[i].set_yticks([])
        divider = make_axes_locatable(axs1[i])
        #recep_mem.append(curr_recep)
        # Append an axes to the right of the current axes with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Create the colorbar in the new axes
        fig1.colorbar(img, cax=cax)

    fig1.suptitle(f'Image Receptive Fields: {model_descrip}', fontsize=16)
    fig1.savefig(f'{model_descrip}_recept.png', format='png')
    
    if show_fft:
        fig2.suptitle(f'Image Receptive Fields FFT: {model_descrip}', fontsize=16)
        fig2.savefig(f'{model_descrip}_recept_FFT.png', format='png')

    plt.tight_layout()  # Add space for the title
    
    plt.show()

# Visualize the weights of the model
    #visualize_img_recep(model_loaded)


        

def visualize_img_recon_recep(model_path, AutoEncoder, neu_side_dim, model_descrip,show_fft):
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    state_dict = torch.load(model_path)

    # Instantiate a new model of the same architecture
    model_loaded = AutoEncoder()

    # Load the state dictionary into the model
    model_loaded.load_state_dict(state_dict)

    fig1, axs1 = plt.subplots(2, 2,figsize=(5, 5))
    axs1 = axs1.flatten()
    if show_fft:
        fig2, axs2 = plt.subplots(2, 2,figsize=(5, 5))
        axs2 = axs2.flatten()

    #recep_mem = []
    idcs = [0,31,992,1023]
    titles = ['Top-Left ', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    for i in range(4):
        # if execution_profile == "CNN": 
        #     show_fft = False
        #     weights = model_loaded.layer1[0].weight.data.cpu().numpy()
        #     curr_recep = weights
        #     curr_recep = np.squeeze(curr_recep)
        # elif execution_profile == "CNN_pool":
        #     show_fft = False
            
        #     weights = model_loaded.layer1[0].weight.data.cpu().numpy()
        #     if i < weights.shape[0]:
        #         curr_recep = weights[i,:,:]
        #     else:
        #         curr_recep = np.zeros((8,8))
        #     curr_recep = np.squeeze(curr_recep)
        # else:
        weights = model_loaded.layer3.weight.data.numpy()
        curr_recep = np.squeeze(weights[idcs[i],:])
        curr_recep = curr_recep.reshape(neu_side_dim, neu_side_dim)
        if show_fft:
            # Perform the 2D FFT
            img_fft = np.fft.fft2(curr_recep)

            # Shift the zero-frequency component to the center
            img_fft = np.fft.fftshift(img_fft)
            magnitude_spectrum = np.log(np.abs(img_fft) + 1)
            img = axs2[i].imshow(magnitude_spectrum, aspect='equal', cmap='gray')
            axs2[i].set_title((f"{titles[i]} Pixel"))
            #axs1[i].set_xlabel('Neurons')
            #axs1[i].set_ylabel('Input Features')
            axs2[i].set_xticks([])
            axs2[i].set_yticks([])
            divider = make_axes_locatable(axs2[i])
            
            # Append an axes to the right of the current axes with the same height
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            # Create the colorbar in the new axes
            fig2.colorbar(img, cax=cax)
        cmap_custom = mcolors.LinearSegmentedColormap.from_list('red_blue', ['blue', 'white', 'red'])
        norm_custom = mcolors.TwoSlopeNorm(vmin=curr_recep.min(), vcenter=0, vmax=curr_recep.max())
        img = axs1[i].imshow(curr_recep, aspect='equal', cmap=cmap_custom,norm=norm_custom)
        axs1[i].set_title(f"{titles[i]} Pixel")
        #axs1[i].set_xlabel('Neurons')
        #axs1[i].set_ylabel('Input Features')
        axs1[i].set_xticks([])
        axs1[i].set_yticks([])
        divider = make_axes_locatable(axs1[i])
        #recep_mem.append(curr_recep)
        # Append an axes to the right of the current axes with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Create the colorbar in the new axes
        fig1.colorbar(img, cax=cax)

    fig1.suptitle(f'Reconstruction Receptive Fields from Neural Map: {model_descrip}', fontsize=16)
    fig1.savefig(f'{model_descrip}_recon_recept.png', format='png')
    
    if show_fft:
        fig2.suptitle(f'Reconstruction Receptive Fields FFT: {model_descrip}', fontsize=16)
        fig2.savefig(f'{model_descrip}_recon_recept_FFT.png', format='png')

    plt.tight_layout()  # Add space for the title
    
    plt.show()

# Visualize the weights of the model
    #visualize_img_recep(model_loaded)


        
    