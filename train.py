import argparse
import logging
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.preprocessing import *
from Full_model import *
from LNL import define_save_LNL_model
from tqdm import tqdm
from evaluate import evaluate
from models import AutoEncoder_fc

def get_args():
    parser = argparse.ArgumentParser(description='Train a model with LNL fixed layers and CIFAR100')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--amp',action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--aug', action='store_true', default=False, help='Use augmentation of rotation and mirroring')
    parser.add_argument('Model', type=str, help='Choose model "FC","default","sparse","CNN","local_patch","CNN_pool"')
    parser.add_argument('model_title',type=str, help='Set file name of saved model')
    parser.add_argument('--elc',type=int, dest='elec_side_dim',default=8,help='Set side dimension of electrode grid')
    parser.add_argument('--neu',type=int, dest='neu_side_dim',default=16,help='Set side dimension of neuron grid')
    parser.add_argument('--d',type=float, dest='drop_rate',default=0.2,help='Set drop rate')
    
    parser.add_argument('--s',type=float, dest='shift',default=2,help='Set shift of sigmoid')
    parser.add_argument('--m',type=float, dest='magnitude',default=2,help='Set magnitude of sigmoid')
    parser.add_argument('--n',type=float, dest='noise',default=0.1,help='Set weight noise')

    return parser.parse_args()

def train_model(
        curr_model,
        device,
        title: str,
        n_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        augment:bool=False
):
    
    input_dir = os.getcwd() + '/data'
    

    # 1. Download Input data: CIFAR100
    if augment:
        cifar100_train_og = datasets.CIFAR100(root=input_dir, train=True, download=True,transform=transform)
        cifar100_train_mir = datasets.CIFAR100(root=input_dir, train=True, download=True,transform=mirrored_transform)
        #cifar100_train_rot = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=rotation_transform)
        cifar100_train = ConcatDataset([cifar100_train_og, cifar100_train_mir])
        # Load CIFAR-10 test data
        cifar100_val_og = datasets.CIFAR100(root=input_dir, train=False, download=True,transform=transform)
        cifar100_val_mir = datasets.CIFAR100(root=input_dir, train=False, download=True,transform=mirrored_transform)
        #cifar100_val_rot = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=rotation_transform)
        cifar100_val = ConcatDataset([cifar100_val_og, cifar100_val_mir])
    else:
        cifar100_train = datasets.CIFAR100(root=input_dir, train=True, download=True,transform=transform)
        # Load CIFAR-10 test data
        cifar100_val = datasets.CIFAR100(root=input_dir, train=False, download=True,transform=transform)

    # 2. Prepare DataLoaders
    train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    #test_dataset = TensorDataset(cifar100_test, cifar100_test)
    val_loader = DataLoader(cifar100_val, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    
    # 3. Save tensor version of input for demonstrations later
    cifar100_train_tsr = cached_loader2tensor(cifar100_train,cache_path=input_dir + '/train.pt')
    cifar100_val_tsr = cached_loader2tensor(cifar100_val,cache_path=input_dir+'/val.pt')

    cifar100_train_tsr_flat = cifar100_train_tsr.view(cifar100_train_tsr.size(0), -1)
    cifar100_val_tsr_flat = cifar100_val_tsr.view(cifar100_val_tsr.size(0), -1)

    n_train = cifar100_train_tsr_flat.shape[0]
    n_val = cifar100_val_tsr_flat.shape[0]
    print(f'Convesion to tensor on {cifar100_train_tsr_flat.device} completed')


    train_err = torch.zeros(n_epochs + 1)
    val_err = torch.zeros(n_epochs + 1)
    criterion = nn.MSELoss()  # mean squared loss, eucledian

    logging.info(f'''Starting training:
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Define an optimizer, loss and scheduler
    optimizer = torch.optim.Adam([
        {'params': curr_model.layer1.parameters()},
        {'params': curr_model.layer3.parameters()}
    ], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler('cuda',enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin Training
    for epoch in range(1, n_epochs + 1):
        curr_model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epochs}', unit='img') as pbar:
            for batch , _ in train_loader:
                batch = batch.view(batch.size(0), -1).to(device)
                #print(f'input is on {input.device}')
                # Zero the parameter gradients
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    outputs,_, _ = curr_model(batch)
                    loss = criterion(outputs,batch)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(curr_model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                #loss.backward()
                #optimizer.step()
                pbar.update(batch.shape[0])
                global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                
                # Evaluation round, 5 rounds/epoch
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_loss, train_loss = evaluate(
                            curr_model, 
                            cifar100_train_tsr_flat, 
                            cifar100_val_tsr_flat,
                            criterion,  
                            device, 
                            amp
                            )
                        scheduler.step(val_loss)
        train_err[epoch] = train_loss.item()
        val_err[epoch] = val_loss.item()   

    file_path = os.path.join(input_dir, f'{title}_err.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump((train_err, val_err), f)
    
    model_path = input_dir + f'/model_{title}.pth'
    torch.save(curr_model.state_dict(), model_path)







    # # # Number of epochs
    

    # #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device


    # # Define Model
    # 
        

    #     # Define a loss function
    #     criterion = nn.MSELoss()  # mean squared loss, eucledian

    #     # Define an optimizer, specifying only the parameters of fc2 and fc3
    #     optimizer = torch.optim.Adam([
    #         {'params': basic_autoencoder.layer1.parameters()},
    #         {'params': basic_autoencoder.layer3.parameters()}
    #     ], lr=learning_rate)
    #     for epoch in range(n_epochs):  # Number of epochs
    #         basic_autoencoder.train() 
    #         for input, _ in train_loader:
                
    #             # flatten
    #             input = input.view(input.size(0), -1).to(device)
    #             #print(f'input is on {input.device}')
    #             # Zero the parameter gradients
    #             optimizer.zero_grad()

    #             # Forward pass
    #             outputs,_, _ = basic_autoencoder(input)

    #             # Compute loss
    #             loss = criterion(outputs, input)

    #             # Backward pass and optimize
    #             loss.backward()
    #             optimizer.step()

    #         # Print loss
    #         if (epoch) % 10 == 0:  # Print every 10 epochs
    #             print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')
            
    #         with torch.no_grad():
    #             # training data
    #             output_train,_,_ = basic_autoencoder(cifar100_train_tsr_flat)
    #             train_loss = criterion(output_train, cifar100_train_tsr_flat)
    #             train_err[epoch + 1,i] = train_loss.item()  

    #             basic_autoencoder.eval()  # Set model to evaluation mode     
    #             # validation data
    #             output_test,_,_ = basic_autoencoder(cifar100_test_tsr_flat)
    #             val_loss = criterion(output_test, cifar100_test_tsr_flat)
    #             val_err[epoch + 1,i] = val_loss.item()
    #     Autoencoders.append(basic_autoencoder)

    # train_err[0], val_err [0] = cifar100_train_tsr_flat.mean(),cifar100_test_tsr_flat.mean()
    # # save the result in the data folder
    # cwd = os.getcwd()
    # data_dir = cwd + '/data'

    # # Create the directory if it doesn't exist
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    
    # file_path = os.path.join(data_dir, f'NN_{model_title}_output.pkl')
    # with open(file_path, 'wb') as f:
    #     pickle.dump((train_err, val_err), f)
    
    # labels = ['lr = 0.01', 'lr = 0.001', 'lr = 0.0001', 'lr = 0.00001']
    # for idx, model in enumerate(Autoencoders):
    #     if len(Autoencoders) == 1:
    #         model_path = cwd+f'/data/model_{model_title}_lr = 0.0001.pth'
    #     else:
    #         model_path = cwd+f'/data/model_{model_title}_{labels[idx]}.pth'
    #     torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    
    LNL_model_path = os.getcwd() + '/data/LNL_model.pth'
    W_d = define_save_LNL_model(LNL_model_path,8, 16, 0, 1,FHWM=False)
    af_array = ['linear','sig','ReLU']
    
    train_model(
        AutoEncoder_fc(device,args.elec_side_dim,args.neu_side_dim, LNL_model_path, args.drop_rate,af_array,args.shift,args.magnitude,args.noise).to(device),
        device,
        args.model_title,
        amp=args.amp)