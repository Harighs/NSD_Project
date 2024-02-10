# /home/hari/projects/NSD_Project/config.py

class Config:

# Model Configuration
    input_dim = 16000
    output_dim = (256, 256)  # Height, Width
    subject_to_use = (1,)
    data_path = '/media/hari/2TB_T7/DATASET/FMRI/NSD-Furkan_Ozcelik/brain-diffuser/data/processed_data/'

# Hyperparameters
    learning_rate = 0.001
    batch_size = 1
    num_epochs = 10
   
# Training Configuration
    data_splitting_ratio = 0.8
    normalize_fmri = True
    Save_Model = True
    Saving_Model_Path = 'Saved_Models/'
    
