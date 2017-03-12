import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config['input_output_paths']={}
    file_paths_config = config['input_output_paths']
    file_paths_config['its_file_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat.xls"
    file_paths_config['project_dir_path'] = "C:/Users/dell/tomcat"
    file_paths_config['code_index_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_index"
    file_paths_config['code_content_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_code_content"
    file_paths_config['bug_content_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_bug_content"
    file_paths_config['oracle_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/Tomcat_oracle"
    file_paths_config['evaluation_path'] = "C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/evaluation"

    config['oracle_generator']={}
    oracle_generator_config = config['oracle_generator']
    oracle_generator_config['issue_field'] = 'both'
    oracle_generator_config['preprocess_options'] = '111111'
    oracle_generator_config['min_word_length'] = '2'
    oracle_generator_config['max_word_length'] = '20'
    oracle_generator_config['parse_options'] = '11'
    oracle_generator_config['neg_sample_number'] = '20'

    config['oracle_reader']={}
    oracle_reader_config = config['oracle_reader']
    oracle_reader_config['vocabulary_size'] = '50'
    oracle_reader_config['split_ratio'] = '0.8'
    oracle_reader_config['max_lstm_length'] = '100'

    config['network_structure']={}
    network_structure_config = config['network_structure']
    network_structure_config['lstm_core_length'] = '32'
    network_structure_config['activation_function'] = 'tanh'
    network_structure_config['inner_activation_function'] = 'hard_sigmoid'
    network_structure_config['distance_function'] = 'cos'
    network_structure_config['initializer'] = 'glorot_uniform'
    network_structure_config['inner_initializer'] = 'orthogonal'
    #network_structure_config['regularizer'] = ''

    config['training_options']={}
    training_options_config = config['training_options']
    training_options_config['optimizer'] = 'RMSprop'
    training_options_config['learning_rate'] = '0.0001'
    training_options_config['epsilon'] = '1e-8'
    training_options_config['decay'] = '0.0'
    training_options_config['rho_1'] = '0.9'
    training_options_config['dropout'] = '0.8'
    training_options_config['nb_epoch'] = '100'
    training_options_config['batch_size'] = '32'

    config['evaluation_options'] = {}
    evaluation_options_config = config['evaluation_options']
    evaluation_options_config['k_value'] = '10'
    evaluation_options_config['rel_threshold'] = '0.65'
    #config['DEFAULT'] = {'ServerAliveInterval': '45','Compression': 'yes','CompressionLevel': '9'}
    #config['bitbucket.org'] = {}
    #config['bitbucket.org']['User'] = 'hg'
    #config['topsecret.server.com'] = {}
    #topsecret = config['topsecret.server.com']
    #topsecret['Port'] = '50022'
    #topsecret['ForwardX11'] = 'no'
    #config['DEFAULT']['ForwardX11'] = 'yes'
    with open('NeeDLes.ini', 'w') as configfile:
        config.write(configfile)

