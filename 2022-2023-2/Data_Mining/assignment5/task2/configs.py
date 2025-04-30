def get_config():
    conf = {
        # dataset
        'train_data_path': './data/train.json',
        'valid_data_path': './data/valid.json',
        'name_len': 6,
        'tokens_len': 50,
        'desc_len': 30,

        # vocabulary info
        'n_words': 10000,
        'vocab_name': 'vocab.name.json',
        'vocab_tokens': 'vocab.tokens.json',
        'vocab_desc': 'vocab.desc.json',

        # training_params
        'batch_size': 64,
        'chunk_size': 200000,
        'nb_epoch': 20,
        # 'optimizer': 'adam',
        'learning_rate': 1.34e-4,  # 2.08e-4,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,

        'emb_size': 512,
        'n_hidden': 512,  # number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 1024,  # 256, # * 2
        'margin': 0.413,  # 0.3986,
        'sim_measure': 'cos',

    }
    return conf
