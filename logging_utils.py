import time
import json
from collections import namedtuple, defaultdict
from datetime import datetime
from operator import attrgetter

from os import makedirs
from os.path import join

import numpy as np
from sklearn.metrics import auc

from utils import make_path


class ResultLogger(object):
    def __init__(self, path_to_log_dir, *args, **kwargs):
        self.start_time = datetime.now().strftime("%Y-%m-%d__%H-%M__%f")
        self._base_path = join(path_to_log_dir, self.start_time)

        print()
        print("Logging to", self._base_path)
        print()
        makedirs(self._base_path, exist_ok=False)

        if 'time' not in kwargs:
            kwargs['time'] = self.start_time

        config_file = join(self._base_path, 'config.jsonl')
        with open(make_path(config_file), 'w') as f:
            f.write(json.dumps(kwargs) + '\n')

        log_file = join(self._base_path, 'logs.jsonl')
        self._log_file = open(make_path(log_file), 'w')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = datetime.now().strftime("%Y-%m-%d__%H-%M__%f")
        self._log_file.write(json.dumps(kwargs) + '\n')
        self._log_file.flush()

    def get_base_dir(self):
        return self._base_path

    @staticmethod
    def _write_pred_file(path_to_file, labels_pred, ids, log_with_id=True):
        with open(make_path(path_to_file), 'w') as pred_f:
            if log_with_id:
                for id_, prediction in zip(ids, labels_pred):
                    pred_f.write('{}\t{}\n'.format(id_, prediction))
            else:
                for prediction in labels_pred:
                    pred_f.write('{}\n'.format(prediction))

    def log_dev_labels(self, labels_dev, ids):
        dev_labels_file = join(self._base_path, 'dev_labels.txt')
        self._write_pred_file(dev_labels_file, labels_dev, ids)

    def log_dev_predictions(self, epoch, labels_pred, ids, log_with_id=True):
        dev_pred_file = join(self._base_path, 'predictions', 'dev', 'predictions_epoch_{}.txt'.format(epoch))
        self._write_pred_file(dev_pred_file, labels_pred, ids, log_with_id=log_with_id)

    def log_test_predictions(self, epoch, labels_pred, ids, log_with_id=True):
        test_pred_file = join(self._base_path, 'predictions', 'test', 'predictions_epoch_{}.txt'.format(epoch))
        self._write_pred_file(test_pred_file, labels_pred, ids, log_with_id=log_with_id)

    def log_test_pr_curve(self, epoch, entity_ids_test, labels_test, probs_test, negative_label_idx, label_encoder=None):
        bag_ids = [e1 + '_' + e2 for e1, e2 in entity_ids_test]

        bag_to_mention_mapping = defaultdict(set)
        for idx, bag_id in enumerate(bag_ids):
            bag_to_mention_mapping[bag_id].add(idx)

        num_relation_facts = 0
        Prediction = namedtuple('Prediction', ['score', 'is_correct', 'bag_id', 'predicted_label_idx', 'bag_label_idxs',
                                               'predicted_label', 'bag_labels', 'bag_size'])
        predictions = []
        for bag_id, mention_idxs in bag_to_mention_mapping.items():
            # Aggregate and count the labels per bag without the negative label
            bag_labels = set(labels_test[list(mention_idxs)])
            bag_labels.discard(negative_label_idx)
            num_relation_facts += len(bag_labels)
            bag_size = len(mention_idxs)

            # Use max to aggregate the mention probabilities in the bag
            mention_probs = probs_test[list(mention_idxs)]
            bag_probs = np.max(mention_probs, axis=0)

            # For each bag and positive relation create a prediction
            for relation_idx, relation_prob in enumerate(bag_probs):
                if relation_idx == negative_label_idx:
                    continue

                if len(bag_labels) == 0:
                    bag_labels_str = 'NA'
                    bag_label_idxs_str = negative_label_idx
                else:
                    if label_encoder:
                        decoded_bag_labels = [label_encoder.get_item_for_index(idx) for idx in bag_labels]
                        bag_labels_str = ', '.join(decoded_bag_labels)
                    else:
                        bag_labels_str = ''

                    bag_label_idxs_str = ', '.join([str(lbl) for lbl in bag_labels])

                if label_encoder:
                    predicted_label_str = label_encoder.get_item_for_index(relation_idx)
                else:
                    predicted_label_str = ""
                predicted_label_idx_str = str(relation_idx)

                is_correct = relation_idx in bag_labels
                predictions.append(Prediction(score=relation_prob,
                                              is_correct=is_correct,
                                              bag_id=bag_id,
                                              predicted_label_idx=predicted_label_idx_str,
                                              bag_label_idxs=bag_label_idxs_str,
                                              predicted_label=predicted_label_str,
                                              bag_labels=bag_labels_str,
                                              bag_size=bag_size))

        predictions = sorted(predictions, key=attrgetter('score'), reverse=True)

        correct = 0
        precision_values = []
        recall_values = []
        for idx, prediction in enumerate(predictions):
            if prediction.is_correct:
                correct += 1
            precision_values.append(correct / (idx+1))
            recall_values.append(correct / num_relation_facts)

        def precision_at(n):
            return (sum([prediction.is_correct for prediction in predictions[:n]]) / n) * 100

        pr_metrics = {
            'P/R AUC': auc(x=recall_values, y=precision_values),
            'Precision@100': precision_at(100),
            'Precision@200': precision_at(200),
            'Precision@500': precision_at(500)
        }

        predictions_dir = join(self._base_path, 'predictions', 'test')
        pr_metrics_file_path = join(predictions_dir, 'pr_metrics_epoch_{}.jsonl'.format(epoch))
        with open(make_path(pr_metrics_file_path), 'w', encoding='utf-8') as pr_metrics_file:
            pr_metrics_file.write(json.dumps(pr_metrics) + '\n')

        pr_predictions_file = join(predictions_dir, 'predictions_pr_curve_epoch_{}.tsv'.format(epoch))
        with open(make_path(pr_predictions_file), 'w') as pr_pred_file:
            tuple_attrs = ['score', 'is_correct', 'bag_id', 'predicted_label_idx',
                           'bag_label_idxs', 'predicted_label', 'bag_labels', 'bag_size']
            pr_pred_file.write("\t".join(tuple_attrs) + "\n")
            for prediction in predictions:
                pred_values = attrgetter(*tuple_attrs)(prediction)
                pred_values = [str(val) for val in pred_values]
                pr_pred_file.write("\t".join(pred_values) + "\n")

        np.save(join(predictions_dir, 'pr_curve_y_epoch_{}.npy'.format(epoch)), precision_values)
        np.save(join(predictions_dir, 'pr_curve_x_epoch_{}.npy'.format(epoch)), recall_values)

    def close(self):
        self._log_file.close()
