
# # Soft Gated Skip Connections
config = {
    # architecture in ['SGSC', 'SHG'] for SoftGatedSkipConnections and StackedHourGlass
    'architecture': 'SGSC',

    'inference': {
        'nstack': 4,
        'inp_dim': 144,
        'oup_dim': 16,
        'num_parts': 16,
        'checkpoint_path': 'checkpoints/SGSC_model_weights.tar',
        'presentation_dir' : 'presentation/',
    },

    'train': {
        'epochs': 200,
        'batchsize': 24,
        'input_res': 256,
        'output_res': 64,
        'learning_rate': 1e-3,
        'decay_epochs': [75, 100, 150],
        'start_learning_rate': 2.5e-4,
        'end_learning_rate': 1e-5,
    },

}


# # Stacked Hourglass Network
# config = {
#     # architecture in ['SGSC', 'SHG'] for SoftGatedSkipConnections and StackedHourGlass
#     'architecture': 'SHG',

#     'inference': {
#         'nstack': 8,
#         'inp_dim': 256, # channels 256
#         'oup_dim': 16,
#         'num_parts': 16,
#         'checkpoint_path': 'checkpoints/SHG_model_weights.tar',
#         'presentation_dir' : '/presentation',
#     },

#     'train': {
#         'epochs': 200,
#         'batchsize': 24,
#         'input_res': 256,
#         'output_res': 64,
#         'learning_rate': 1e-3,
#         'decay_epochs': [75, 100, 150],
#         'start_learning_rate': 2.5e-4,
#         'end_learning_rate': 1e-5,
#     },

# }
