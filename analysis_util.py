import re
import json
from os import listdir
from os.path import join, isdir, exists, basename
from subprocess import run, PIPE

import pandas as pd
from sklearn.preprocessing import label_binarize


def read_log_file(experiment_dir, log_file_name='logs.jsonl'):
    logs = []
    with open(join(experiment_dir, log_file_name), 'r') as f:
        for epoch, log in enumerate(f, start=1):
            epoch_log = json.loads(log)

            epoch_log['epoch'] = epoch

            dev_micro_f1 = epoch_log['dev_micro_f1']
            epoch_log['dev_micro_f1'] = dev_micro_f1 * 100.

            dev_macro_f1 = epoch_log['dev_macro_f1']
            epoch_log['dev_macro_f1'] = dev_macro_f1 * 100.

            logs.append(epoch_log)

    return logs


def read_config_file(experiment_dir, config_file_name='config.jsonl'):
    with open(join(experiment_dir, config_file_name), 'r') as f:
        return json.loads(next(f))


def read_experiment_logs(experiments_dir, filter_empty_logs=True):
    def list_experiment_dirs(path):
        dirs = [join(path, d) for d in listdir(path) if isdir(join(path, d))]
        return [d for d in dirs if exists(join(d, 'logs.jsonl'))]

    experiment_dirs = list_experiment_dirs(experiments_dir)

    experiments = {}
    for experiment_dir in experiment_dirs:
        experiment_name = basename(experiment_dir)

        config = read_config_file(experiment_dir)
        logs = read_log_file(experiment_dir)

        if filter_empty_logs and not logs:
            continue

        experiments[experiment_name] = {
            'experiment_dir': experiment_dir,
            'config': config,
            'logs': logs
        }

    return experiments


def add_official_scorer_metrics(experiments, path_to_eval_script, path_to_test_answers):
    for experiment_name, experiment in experiments.items():
        experiment_dir = experiment['experiment_dir']
        config = experiment['config']
        logs = experiment['logs']
        
        if config['dataset'] == 'semeval_2010_task8':
            dev_id_labels_true_file = join(experiment_dir, 'dev_labels.txt')
            test_id_labels_true_file = path_to_test_answers
            
            for log in logs:
                epoch = log['epoch']
                
                dev_id_labels_pred_file = join(experiment_dir, f'predictions/dev/predictions_epoch_{epoch}.txt')
                test_id_labels_pred_file = join(experiment_dir, f'predictions/test/predictions_epoch_{epoch}.txt')
                
                dev_precision_official, dev_recall_official, dev_f1_official = \
                    evaluate_semeval2010_task8(id_labels_true_file=dev_id_labels_true_file,
                                               id_labels_pred_file=dev_id_labels_pred_file,
                                               eval_script=path_to_eval_script)
                
                test_precision_official, test_recall_official, test_f1_official = \
                    evaluate_semeval2010_task8(id_labels_true_file=test_id_labels_true_file,
                                               id_labels_pred_file=test_id_labels_pred_file,
                                               eval_script=path_to_eval_script)
                
                log['dev_precision_official'] = dev_precision_official
                log['dev_recall_official'] = dev_recall_official
                log['dev_f1_official'] = dev_f1_official
                
                log['test_precision_official'] = test_precision_official
                log['test_recall_official'] = test_recall_official
                log['test_f1_official'] = test_f1_official
    
    return experiments


PRECISION_REGEX = r'P =\s*([0-9]{1,2}\.[0-9]{2})%'
RECALL_REGEX = r'R =\s*([0-9]{1,2}\.[0-9]{2})%'
F1_REGEX = r'F1 =\s*([0-9]{1,2}\.[0-9]{2})%'

OFFICIAL_RESULT_REGEX = r'\(9\+1\)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL'
RESULT_LINE_REGEX = r'MACRO-averaged result \(excluding Other\):\n((.*\n){1})'

def evaluate_semeval2010_task8(id_labels_true_file, id_labels_pred_file, eval_script):
    p = run([eval_script, id_labels_true_file, id_labels_pred_file], stdout=PIPE, encoding='utf-8')
    report = p.stdout

    official_result_match = re.search(OFFICIAL_RESULT_REGEX, report)

    if official_result_match:
        result_start = official_result_match.span(0)[1]
        match = re.search(RESULT_LINE_REGEX, report[result_start:])

        precision = None
        recall = None
        f1 = None
        if match:
            result_line = match.group(1)
            precision_match = re.search(PRECISION_REGEX, result_line)
            recall_match = re.search(RECALL_REGEX, result_line)
            f1_match = re.search(F1_REGEX, result_line)
            
            if precision_match:
                precision = float(precision_match.group(1))
            if recall_match:
                recall = float(recall_match.group(1))
            if f1_match:
                f1 = float(f1_match.group(1))
        
    return precision, recall, f1


def experiments_to_dataframe(experiments):
    all_logs = []
    all_configs = []
    for experiment_name, experiment in experiments.items():
        config = experiment['config']
        logs = experiment['logs']

        config['experiment'] = experiment_name
        all_configs.append(config)

        for log in logs:
            log['experiment'] = experiment_name
        all_logs.extend(logs)

    return pd.DataFrame(all_logs), pd.DataFrame(all_configs)


def load_experiments_df(log_dir):
    experiment_logs = read_experiment_logs(log_dir)
    df_logs, df_configs = experiments_to_dataframe(experiment_logs)
    experiments_df = df_configs.set_index('time').join(df_logs.set_index('experiment')).reset_index(drop=True)
    return experiments_df
