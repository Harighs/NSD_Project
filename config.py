# /home/hari/projects/NSD_Project/config.py

class Config:
    input_dim = 16000
    output_dim = (256, 256)  # Height, Width
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    subject_to_use = (1,)
    data_path = '/media/hari/2TB_T7/DATASET/FMRI/NSD-Furkan_Ozcelik/brain-diffuser/data/processed_data/'
    Save_Model = True
    Saving_Model_Path = 'Saved_Models/'
    normalize_fmri = True
    data_splitting_ratio = 0.8
    
    # Normal details (Not Changable)
    training_start_time = None
    training_end_time = None
    total_time_taken = None