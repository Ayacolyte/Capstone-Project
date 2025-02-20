import torch

@torch.inference_mode()
def evaluate(curr_model, train_tsr, val_tsr,criterion,  device, amp):
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        curr_model.eval()  # Set model to evaluation mode    
        # training data
        output_train,_,_ = curr_model(train_tsr)
        train_loss = criterion(output_train, train_tsr)
        
        # validation data
        output_val,_,_ = curr_model(val_tsr)
        val_loss = criterion(output_val, val_tsr)

        curr_model.train()
        return val_loss, train_loss