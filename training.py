# usr/bin/python3
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(model, train_loader, criterion, optimizer, config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('GPU not available')

    # Initialize the model, loss function, and optimizer
    model.to(device)
    num_epochs = config.num_epochs
    
    # Prinitng out details and model summary for sanity
    print('Model Summary : ')
    print(model)
    print('Number of Epochs : ', num_epochs)
    print('Batch Size : ', config.batch_size)
    print('Learning Rate : ', config.learning_rate)
    print('Loss Function : ', criterion)
    print('Optimizer : ', optimizer)
    print('DataLoader : ', train_loader)
    print('Device in Use : ', device)
    print('Model in Use : ', model)
    print('Model Parameters : ', model.parameters)
    print('Model State Dict : ', model.state_dict)
    print('Model Optimizer : ', optimizer)
    
    for epoch in range(num_epochs):
        # start time current time
        config.training_start_time = time.strftime("%c")
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, fmri in tepoch:
                images, fmri = images.to(device), fmri.to(device)
                tepoch.set_description(f"Epoch {epoch+1}")

            # Forward pass
                optimizer.zero_grad()
                outputs = model(fmri)
            # Backward and optimize
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            
            # Clear memory
                del images, fmri, outputs, loss
                torch.cuda.empty_cache()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()
        config.training_end_time = time.strftime("%c")

    # if config.Save_Model:        
    #     # create name convention for model includeing epoch and loss time date etc
    #     base_name = f"{model.__class__.__name__}_{time.strftime('%Y-%m-%d-%H:%M')}_epoch{epoch+1}_loss-{avg_loss:.2f}"
    #     # Saving the model
    #     model_name_for_saving = base_name + '.pth'
    #     torch.save(model, os.path.join(os.getcwd(),config.Saving_Model_Path, model_name_for_saving))
        
    #     # also save other details as a text file
    #     desciption_name_for_saving = base_name + '.txt'
    #     with open(os.path.join(os.getcwd(),config.Saving_Model_Path, desciption_name_for_saving), 'w') as f:
    #     # Model details
    #         f.write(f"%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%")
    #         f.write(f"Model Summary : \n{model}\n")
    #         f.write(f"Number of Epochs : {num_epochs}\n")
    #         f.write(f"Batch Size : {config.batch_size}\n")
    #         f.write(f"Learning Rate : {config.learning_rate}\n")
    #         f.write(f"Loss Function : {criterion}\n")
    #         f.write(f"Optimizer : {optimizer}\n")
    #         f.write(f"Device in Use : {device}\n")
    #         f.write(f"DataLoader : {train_loader}\n")
    #         f.write(f"Model in Use : {model.__class__.__name__}\n")
    #         f.write(f"Model Parameters : {sum(p.numel() for p in model.parameters())}\n")
    #         f.write(f"Model State Dict : {model.state_dict}\n")
    #         f.write(f"Model Optimizer : {optimizer}\n")
    #         f.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}\n")
    #     # Subjects used for training
    #         f.write(f"Subjects used for training : {config.subject_to_use}\n")
    #         f.write(f"Model Saving Path : {os.path.join(os.getcwd(),config.Saving_Model_Path, model_name_for_saving)}\n")
    #     # Time details
    #         f.write(f"Start Time : {start_time}\n")
    #         end_time = time.strftime("%c")
    #         diff = time.mktime(time.strptime(end_time)) - time.mktime(time.strptime(start_time)) / 60
    #         f.write(f"End Time : {end_time}\n")
    #         f.write(f"Total Training Time : {diff} minutes\n")
    #         f.close()
        
        # print('Model Saved Successfully')