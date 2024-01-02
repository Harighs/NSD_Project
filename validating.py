# usr/bin/python3

import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test(model, test_loader, criterion, optimizer, config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('Gpu not avialble for testing')
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0
    with torch.no_grad():  # No need to track gradients for testing
        with tqdm(test_loader, unit="batch") as tepoch:
            for images, fmri in tepoch:
                images, fmri = images.to(device), fmri.to(device)
                tepoch.set_description(f"Testing Started")

                # Forward pass
                outputs = model(fmri)
                loss = criterion(outputs, images)

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

                # Clear memory
                del images, fmri, outputs, loss
                torch.cuda.empty_cache()

    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    if config.Save_Model:
        # Create name convention for model including epoch, loss, time, date, etc
        base_name = f"{model.__class__.__name__}_{time.strftime('%Y-%m-%d-%H:%M')}_epoch{config.num_epochs+1}_loss-{avg_loss:.2f}"
        # Saving the model
        model_name_for_saving = base_name + '.pth''
        torch.save(mode, os.path.join(os.getcwd(),config.Saving_Model_Path, model_name_for_saving))

        # also saving other parameter details as text file
        description_name_for_saving = base_name + '.txt''
        
        with open(os.path.join(os.getcwd(),config.Saving_Model_Path, description_name_for_saving), 'w') as f:
        # Model Details
            f.write(f"%%%%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%%%")
            f.write(f"Model Summary : \n{model}\n")
            f.write(f"Number of Epochs : {config.num_epochs}\n")
            f.write(f"Batch Size : {config.batch_size}\n")
            f.write(f"Learning Rate : {config.learning_rate}\n")
            f.write(f"Loss Function : {criterion}\n")
            f.write(f"Optimizer : {optimizer}\n")
            f.write(f"DataLoader : {test_loader}\n")
            f.write(f"Device in Use : {device}\n")
            f.write(f"Model in Use : {model.__class__.__name__}\n")
            f.write(f"Model Parameters : {sum(p.numel() for p in model.parameters())}\n")
            f.write(f"Model State Dict : {model.state_dict}\n")
            f.write(f"Model Optimizer : {optimizer}\n")
            f.write(f"Epoch [{config.num_epochs+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}\n")
        # Subjects used for training
            f.write(f"Subjects used for training : {config.subject_to_use}\n")
            f.write(f"Model Saving Path : {os.path.join(os.getcwd(),config.Saving_Model_Path, model_name_for_saving)}\n")
        # Time details
            diff = time.mktime(time.strptime(config.training_end_time)) - time.mktime(time.strptime(config.training_start_time)) / 60
            f.write(f"Traning_Time : {diff} minutes" )
            f.close()


