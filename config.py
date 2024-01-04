config = {
    'num_epochs': 1,
    'learning_rate': 0.0001,
    'train_ds': 'data/mams/mams_acsa/train.xml',
    'val_ds': 'data/mams/mams_acsa/val.xml',
    'batch_size': 32,
    'max_length': 100,
    # embedding
    'embedding_pretrained_path': 'facebook/bart-base',
    'vocab_size': 8000, #7700
    'embedding_dim': 768,
    # 'padding_idx': 1,
    # cnn
    'cnn_hidden_dim': 10,
    'kernel_size': 3,
    'padding': 1,
    # bilstm
    'lstm_units': 256,
    'lstm_layers': 4,
    # linear
    'linear_hidden_dim': 64,
    'num_category': 8,
    'num_polarity': 3,
    'dropout': 0.25
}

def get_config():
    return config