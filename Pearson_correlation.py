def Pearson_corr_comp(neu_maps,model_descrip):
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    print("confirming")
    
    length = int(neu_maps.shape[0])
    print(0.02*length)
    random_idx = random.sample(range(length), int(0.02*length))
    sel_img = neu_maps[random_idx,:,:]
    from itertools import combinations
    combs = list(combinations(range(int(0.02*length)),2))
    random_vec = 1.15*np.random.uniform(0, 1, (length,int(neu_maps.shape[1]),int(neu_maps.shape[1])))

    def pearson_corr(x,y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

        return numerator/denominator
    corr_img_mat = []
    corr_random_mat = []
    for comb in combs:
        idx1 = comb[0]
        idx2 = comb[1]

        # calculate pearson correlation coefficient
        corr_img = pearson_corr(sel_img[idx1,:,:],sel_img[idx2,:,:])
        corr_img_mat.append(corr_img)
        corr_random = pearson_corr(random_vec[idx1,:,:],random_vec[idx2,:,:])
        corr_random_mat.append(corr_random)

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].hist(corr_img_mat, density = True, bins=50, color='blue', edgecolor='black', label='Between Image Correlation')
    axs[0].set_title('Between Image Correlation Distribution')
    axs[0].set_xlabel('Pearson Correlation Coefficient')
    axs[0].set_ylabel('Normalised Frequency')
    axs[0].set_xlim(-1 ,1)

    axs[1].hist(corr_random_mat, density = True, bins=50, color='blue', edgecolor='black', label='Random Vector Correlation')
    axs[1].set_title('Random Vector Correlation Distribution')
    axs[1].set_xlabel('Pearson Correlation Coefficient')
    axs[1].set_ylabel('Normalised Frequency')
    axs[1].set_xlim(-1 ,1)


    plt.suptitle(f"Correlation between Images compared to Random Vectors:{model_descrip}")
    plt.savefig(f"{model_descrip}_Corr.png", format='png')
    plt.show()


    