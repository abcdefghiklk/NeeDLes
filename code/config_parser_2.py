import ConfigParser
import os
from main import main_siamese_lstm

def parseArgs():

    #required arguments:
    parser = argparse.ArgumentParser(description='running the lstm siamese network from configuration file')
    parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
    args = parser.parse_args()
    return(args)

def parse_config(config_file_path):
    config = ConfigParser.RawConfigParser()
    config.read('NeeDLes_linux.ini')

    its_file_path = config.get('input_output_paths','its_file_path')
    project_dir_path = config.get('input_output_paths','project_dir_path')
    code_index_path = config.get('input_output_paths','code_index_path')
    code_contents_path = config.get('input_output_paths','code_contents_path')
    bug_contents_path = config.get('input_output_paths','bug_contents_path')
    file_oracle_path = config.get('input_output_paths','file_oracle_path')
    method_oracle_path = config.get('input_output_paths','method_oracle_path')
    model_dir_path = config.get('input_output_paths','model_dir_path')
    evaluation_path = config.get('input_output_paths','evaluation_path')

    issue_field = 'description'
    if config.has_option('options','issue_field'):
        issue_field =  config.get('options','issue_field')

    preprocess_options = '000000'
    if config.has_option('options','preprocess_options'):
        preprocess_options = config.get('options','preprocess_options')

    parse_options = '00'
    if config.has_option('options','parse_options'):
        parse_options = config.get('options','parse_options')

    min_word_length = 0
    if config.has_option('options','min_word_length'):
        min_word_length = config.get('options','min_word_length')

    max_word_length = 20
    if config.has_option('options','max_word_length'):
        max_word_length = config.get('options','max_word_length')

    neg_sample_number = 10
    if config.has_option('options','neg_sample_number'):
        neg_sample_number =  config.get('options','neg_sample_number')

    run_python_str = '/usr/local/python27/bin/python main.py -b {} -c {} -f {} -m {} -d {} -e {}'.format(bug_contents_path, code_contents_path, file_oracle_path, method_oracle_path, model_dir_path, evaluation_path)


    if config.has_option('oracle_reader,','vocabulary_size'):
        vocabulary_size = config.getint('oracle_reader','vocabulary_size')
        run_python_str = run_python_str + ' --v {}'.format(vocabulary_size)

    if config.has_option('oracle_reader','lstm_seq_length'):
        lstm_seq_length = config.getint('oracle_reader','lstm_seq_length')
        run_python_str = run_python_str + ' --s {}'.format(lstm_seq_length)

    if config.has_option('oracle_reader','neg_method_num'):
        neg_method_num = config.getint('oracle_reader','neg_method_num')
        run_python_str = run_python_str + ' --s {}'.format(neg_method_num)

    if config.has_option('oracle_reader','split_ratio'):
        split_ratio = config.getfloat('oracle_reader','split_ratio')
        run_python_str = run_python_str + ' --sr {}'.format(split_ratio)

    if config.has_option('network_structure','lstm_core_length'):
        lstm_core_length = config.getint('network_structure','lstm_core_length')
        run_python_str = run_python_str + ' --l {}'.format(lstm_core_length)

    if config.has_option('network_structure','activation_function'):
        activation_function = config.get('network_structure','activation_function')
        run_python_str = run_python_str + ' --af {}'.format(activation_function)

    if config.has_option('network_structure','inner_activation_function'):
        inner_activation_function = config.get('network_structure','inner_activation_function')
        run_python_str = run_python_str + ' --iaf {}'.format(inner_activation_function)

    if config.has_option('network_structure','distance_function'):
        distance_function = config.get('network_structure','distance_function')
        run_python_str = run_python_str + ' --df {}'.format(distance_function)

    if config.has_option('network_structure','initializer'):
        initializer = config.get('network_structure','initializer')
        run_python_str = run_python_str + ' --i {}'.format(initializer)

    if config.has_option('network_structure','inner_initializer'):
        inner_initializer = config.get('network_structure','inner_initializer')
        run_python_str = run_python_str + ' --ii {}'.format(inner_initializer)

    if config.has_option('network_structure','regularizer'):
        regularizer = config.get('network_structure','regularizer')
        run_python_str = run_python_str + ' --r {}'.format(regularizer)

    if config.has_option('training_options','optimizer'):
        optimizer = config.get('training_options','optimizer')
        run_python_str = run_python_str + ' --op {}'.format(optimizer)
        if config.has_option('training_options','learning_rate'):
            lr = config.getfloat('training_options','learning_rate')
            run_python_str = run_python_str + ' --lr {}'.format(lr)
        if config.has_option('training_options','epsilon'):
            epsilon = config.getfloat('training_options','epsilon')
            run_python_str = run_python_str + ' --ep {}'.format(epsilon)
        if config.has_option('training_options','decay'):
            decay = config.getfloat('training_options','decay')
            run_python_str = run_python_str + ' --dc {}'.format(decay)
        if config.has_option('training_options','rho_1'):
            rho_1 = config.getfloat('training_options','rho_1')
            run_python_str = run_python_str + ' --dc {}'.format(rho_1)
            if config.has_option('training_options','rho_2'):
                rho_2 = config.getfloat('training_options','rho_2')
                run_python_str = run_python_str + ' {}'.format(rho_2)

    if config.has_option('training_options','dropout'):
        dropout = config.getfloat('training_options','dropout')
        run_python_str = run_python_str + ' --d {}'.format(dropout)

    if config.has_option('training_options','nb_epoch'):
        nb_epoch = config.getint('training_options','nb_epoch')
        run_python_str = run_python_str + ' --en {}'.format(nb_epoch)

    if config.has_option('evaluation_options','k_value'):
        k_value = config.getint('evaluation_options','k_value')
        run_python_str = run_python_str + ' --k {}'.format(k_value)

    if config.has_option('evaluation_options','rel_threshold'):
        rel_threshold = config.getfloat('evaluation_options','rel_threshold')
        run_python_str = run_python_str + ' --th {}'.format(rel_threshold)

   # run_jar_str ='java -jar C:/Users/dell/Dropbox/NeeDLes/data/Hyloc_data/OracleGenerator.jar -i {} -d {} -x {} -c {} -b {} -o {}'.format(its_file_path,project_dir_path,code_index_path,code_content_path,bug_content_path,oracle_path)
   # os.system(run_jar_str)

    print(run_python_str)
    os.system(run_python_str)

if __name__ == '__main__':
    args = parseArgs()
    parse_config(args.config_file_path)







