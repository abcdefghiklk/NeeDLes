import configparser
import os
from main import main_siamese_lstm

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('NeeDLes.ini')
    file_paths_config = config['input_output_paths']

    its_file_path = file_paths_config['its_file_path']
    project_dir_path = file_paths_config['project_dir_path']
    code_index_path = file_paths_config['code_index_path']
    code_content_path = file_paths_config['code_content_path']
    bug_content_path = file_paths_config['bug_content_path']
    oracle_path = file_paths_config['oracle_path']
    evaluation_path = file_paths_config['evaluation_path']

    oracle_generator_config = config['oracle_generator']


    issue_field = 'description'
    if 'issue_field' in oracle_generator_config:
        issue_field = oracle_generator_config['issue_field']

    preprocess_options = '000000'
    if 'preprocess_options' in oracle_generator_config:
        preprocess_options = oracle_generator_config['preprocess_options']

    min_word_length = 0
    if 'min_word_length' in oracle_generator_config:
        min_word_length = oracle_generator_config['min_word_length']

    max_word_length = 20
    if 'max_word_length' in oracle_generator_config:
        max_word_length = oracle_generator_config['max_word_length']

    parse_options = '00'
    if 'parse_options' in oracle_generator_config:
        parse_options = oracle_generator_config['parse_options']

    neg_sample_number = 10
    if 'neg_sample_number' in oracle_generator_config:
        neg_sample_number = oracle_generator_config['neg_sample_number']


    run_python_str = 'python main.py -b {} -c {} -o {} -e {}'.format(bug_content_path, code_content_path, oracle_path, evaluation_path)


    oracle_reader_config = config['oracle_reader']
    if 'vocabulary_size' in oracle_reader_config:
        vocabulary_size = int(oracle_reader_config['vocabulary_size'])
        run_python_str = run_python_str + ' --v {}'.format(vocabulary_size)

    if 'split_ratio' in oracle_reader_config:
        split_ratio = float(oracle_reader_config['split_ratio'])
        run_python_str = run_python_str + ' --sr {}'.format(split_ratio)

    if 'max_lstm_length' in oracle_reader_config:
        max_lstm_length = int(oracle_reader_config['max_lstm_length'])
        run_python_str = run_python_str + ' --m {}'.format(max_lstm_length)

    network_structure_config = config['network_structure']
    if 'lstm_core_length' in network_structure_config:
        lstm_core_length = int(network_structure_config['lstm_core_length'])
        run_python_str = run_python_str + ' --l {}'.format(lstm_core_length)

    if 'activation_function' in network_structure_config:
        activation_function = network_structure_config['activation_function']
        run_python_str = run_python_str + ' --af {}'.format(activation_function)

    if 'inner_activation_function' in network_structure_config:
        inner_activation_function = network_structure_config['inner_activation_function']
        run_python_str = run_python_str + ' --iaf {}'.format(inner_activation_function)

    if 'distance_function' in network_structure_config:
        distance_function = network_structure_config['distance_function']
        run_python_str = run_python_str + ' --df {}'.format(distance_function)

    if 'initializer' in network_structure_config:
        initializer = network_structure_config['initializer']
        run_python_str = run_python_str + ' --i {}'.format(initializer)

    if 'inner_initializer' in network_structure_config:
        inner_initializer = network_structure_config['inner_initializer']
        run_python_str = run_python_str + ' --ii {}'.format(inner_initializer)

    if 'regularizer' in network_structure_config:
        if network_structure_config['regularizer'] != '':
            regularizer = network_structure_config['regularizer']
            run_python_str = run_python_str + ' --r {}'.format(regularizer)

    training_options_config = config['training_options']
    if 'optimizer' in training_options_config:
        optimizer = training_options_config['optimizer']
        run_python_str = run_python_str + ' --op {}'.format(optimizer)
        if 'learning_rate' in training_options_config:
            lr = training_options_config['learning_rate']
            run_python_str = run_python_str + ' --lr {}'.format(lr)
        if 'epsilon' in training_options_config:
            epsilon = training_options_config['epsilon']
            run_python_str = run_python_str + ' --ep {}'.format(epsilon)
        if 'decay' in training_options_config:
            decay = training_options_config['decay']
            run_python_str = run_python_str + ' --dc {}'.format(decay)
        if 'rho_1' in training_options_config:
            rho_1 = training_options_config['rho_1']
            run_python_str = run_python_str + ' --dc {}'.format(rho_1)
            if 'rho_2' in training_options_config:
                rho_2 = training_options_config['rho_2']
                run_python_str = run_python_str + ' {}'.format(rho_2)

    if 'dropout' in training_options_config:
        dropout = float(training_options_config['dropout'])
        run_python_str = run_python_str + ' --d {}'.format(dropout)

    if 'nb_epoch' in training_options_config:
        nb_epoch = int(training_options_config['nb_epoch'])
        run_python_str = run_python_str + ' --en {}'.format(nb_epoch)

    if 'batch_size' in training_options_config:
        batch_size = int(training_options_config['batch_size'])
        run_python_str = run_python_str + ' --bs {}'.format(batch_size)

    evaluation_options_config = config['evaluation_options']
    if 'k_value' in evaluation_options_config:
        k_value = int(evaluation_options_config['k_value'])
        run_python_str = run_python_str + ' --k {}'.format(k_value)

    if 'rel_threshold' in evaluation_options_config:
        rel_threshold = float(evaluation_options_config['rel_threshold'])
        run_python_str = run_python_str + ' --th {}'.format(rel_threshold)

   # run_jar_str ='java -jar C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/OracleGenerator.jar -i {} -d {} -x {} -c {} -b {} -o {}'.format(its_file_path,project_dir_path,code_index_path,code_content_path,bug_content_path,oracle_path)
   # os.system(run_jar_str)

    print(run_python_str)
    os.system(run_python_str)





