"""
Converts various datasets into a jsonl format.
The following datasets can be converted:
    Semeval 2010 Task 8:
        Paper: http://www.aclweb.org/anthology/S10-1006
        Download: http://www.kozareva.com/downloads.html
    KBP37:
        Paper: https://arxiv.org/abs/1508.01006
        Download: https://github.com/zhangdongxu/kbp37
    TACRED:
        Paper: https://nlp.stanford.edu/pubs/zhang2017tacred.pdf
        Download: LDC publication pending


Exemplary conversion for the Semeval 2010 Task 8 Format:
9       "The <e1>lawsonite</e1> was contained in a <e2>platinum crucible</e2> and the counter-weight was a plastic crucible with metal pieces."
Content-Container(e1,e2)
Comment: prototypical example

JSONL output Format:
{
  "id": "9",
  "tokens": ["The", "lawsonite", "was", "contained", "in", "a", "platinum", "crucible", "and", "the", "counter-weight", "was", "a", "plastic", "crucible", "with", "metal", "pieces", "."],
  "label": "Content-Container(e1,e2)",
  "entities": [[1, 2], [6, 8]]
}
"""

import argparse
import json
import os
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split

from datasets import SemEval2010Task8
from utils import make_path

SUPPORTED_DATASETS = ['semeval', 'kbp37', 'tacred']


class DatasetConverter:

    def __init__(self, dataset, dataset_dir, output_dir, subsample):

        self.dataset = dataset
        self.subsample = subsample

        if dataset == "semeval":
            self.input_train_file = os.path.join(dataset_dir, "SemEval2010_task8_training", "TRAIN_FILE.TXT")
            self.input_test_file = os.path.join(dataset_dir, "SemEval2010_task8_testing_keys", "TEST_FILE_FULL.TXT")
            self.input_dev_file = None
        elif dataset == "kbp37":
            self.input_train_file = os.path.join(args.dataset_dir, "train.txt")
            self.input_test_file = os.path.join(args.dataset_dir, "test.txt")
            self.input_dev_file = os.path.join(args.dataset_dir, "dev.txt")
        elif dataset == "tacred":
            path_to_json_files = os.path.join(dataset_dir, "data", "json")
            self.input_train_file = os.path.join(path_to_json_files, "train.json")
            self.input_test_file = os.path.join(path_to_json_files, "test.json")
            self.input_dev_file = os.path.join(path_to_json_files, "dev.json")
        else:
            raise RuntimeError("Only the following datasets are supported: " + ", ".join(SUPPORTED_DATASETS))

        self.output_dir = output_dir

        assert os.path.exists(self.input_train_file), "Train file not found: {}".format(self.input_train_file)
        if not subsample:
            self.output_train_file = os.path.join(output_dir, "train.jsonl")
        else:
            self.masking_modes = [None, 'grammar', 'ner', 'grammar_and_ner', 'unk', 'unk_w_position']

        assert os.path.exists(self.input_test_file), "Test file not found: {}".format(self.input_test_file)
        self.output_test_file = os.path.join(output_dir, "test.jsonl")

        if self.input_dev_file:
            assert os.path.exists(self.input_dev_file), "Test file not found: {}".format(self.input_dev_file)
            self.output_dev_file = os.path.join(output_dir, "dev.jsonl")
        else:
            self.output_dev_file = None

        self.glove_mapping = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}'
        }

    def run(self):
        print("Converting dataset to jsonl")
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.subsample:
            self._run_normally()
        else:
            self._run_subsampling()

    def _run_normally(self):
        # Convert the dev and test set
        if self.dataset in ['semeval', 'kbp37']:
            self._convert_semeval_format_file(self.input_test_file, self.output_test_file)
            if self.output_dev_file:
                self._convert_semeval_format_file(self.input_dev_file, self.output_dev_file)
        elif self.dataset == 'tacred':
            self._convert_tacred_format_file(self.input_test_file, self.output_test_file)
            self._convert_tacred_format_file(self.input_dev_file, self.output_dev_file)
        else:
            raise RuntimeError("Unexpected dataset: " + self.dataset)

        if self.dataset in ['semeval', 'kbp37']:
            self._convert_semeval_format_file(self.input_train_file, self.output_train_file)
        elif self.dataset == 'tacred':
            self._convert_tacred_format_file(self.input_train_file, self.output_train_file)
        else:
            raise RuntimeError("Unexpected dataset: " + self.dataset)

    def _run_subsampling(self):
        train_examples = list(self._read_tacred_file(self.input_train_file))
        train_labels = list(map(itemgetter('label'), train_examples))
        dev_examples = list(self._read_tacred_file(self.input_dev_file))
        test_examples = list(self._read_tacred_file(self.input_test_file))

        for sample_ratio in np.linspace(.1, 1.0, 10):
            sampling_dir = os.path.join(self.output_dir, str(int(sample_ratio * 100)))
            subsampled_ids_file = os.path.join(sampling_dir, "sentence_ids")

            if self.dataset == 'tacred':
                if sample_ratio == 1.0:
                    subsampled_examples = train_examples
                else:
                    subsampled_examples, _ = train_test_split(train_examples,
                                                              train_size=sample_ratio,
                                                              stratify=train_labels)
            else:
                raise RuntimeError("Unsupported dataset: " + self.dataset)

            with open(make_path(subsampled_ids_file), 'w') as ids_file:
                for example in subsampled_examples:
                    ids_file.write(str(example['id']) + "\n")

            for masking_mode in self.masking_modes:
                masking_mode_name = 'unmasked' if masking_mode is None else masking_mode
                masking_dir = os.path.join(sampling_dir, masking_mode_name)

                print("Creating train set with sampling ratio {:.1f} and masking mode {}"
                      .format(sample_ratio, masking_mode_name))
                output_train_file = os.path.join(masking_dir, "train.jsonl")

                if masking_mode is None:
                    masked_examples = subsampled_examples
                else:
                    masked_examples = [SemEval2010Task8.apply_masking_mode(example, masking_mode)
                                       for example in subsampled_examples]

                with open(make_path(output_train_file), 'w') as output_file:
                    for example in masked_examples:
                        output_file.write(json.dumps(example) + "\n")

        # Write dev set with different masking modes
        for masking_mode in self.masking_modes:
            masking_mode_name = 'unmasked' if masking_mode is None else masking_mode
            masking_dir = os.path.join(self.output_dir, masking_mode_name)

            print("Creating dev and test set with masking mode {}".format(masking_mode_name))
            output_dev_file = os.path.join(masking_dir, "dev.jsonl")
            output_test_file = os.path.join(masking_dir, "test.jsonl")

            if masking_mode is None:
                masked_dev_examples = dev_examples
                masked_test_examples = test_examples
            else:
                masked_dev_examples = [SemEval2010Task8.apply_masking_mode(example, masking_mode)
                                       for example in dev_examples]
                masked_test_examples = [SemEval2010Task8.apply_masking_mode(example, masking_mode)
                                        for example in test_examples]

            with open(make_path(output_dev_file), 'w') as output_file:
                for example in masked_dev_examples:
                    output_file.write(json.dumps(example) + "\n")

            with open(make_path(output_test_file), 'w') as output_file:
                for example in masked_test_examples:
                    output_file.write(json.dumps(example) + "\n")

    def _convert_semeval_format_file(self, input_path, output_path, sample_ratio=None):
        with open(input_path, mode="r") as input_file, open(output_path, mode="w") as output_file:
            while True:
                tokens_line = input_file.readline()
                if not tokens_line:
                    break

                (index, tokens_string) = tokens_line.split('\t', maxsplit=1)  # separate index and tokens
                tokens_string = tokens_string.strip()[1:-1]  # remove quotation marks
                tokens = self._split_tokens(tokens_string)

                tokens, first_args, second_args = self._parse_args(tokens)

                relation_label = input_file.readline().strip()  # Remove trailing newline
                _ = input_file.readline()  # Comment string
                _ = input_file.readline()  # Empty line separator

                example = {
                    "id": index,
                    "tokens": tokens,
                    "label": relation_label,
                    "entities": [first_args, second_args]
                }

                output_file.write(json.dumps(example) + "\n")

    @staticmethod
    def _split_tokens(tokens_string):
        prepared_string = tokens_string \
            .replace(".", " . ") \
            .replace("<e1>", " <e1>") \
            .replace("</e1>", "</e1> ") \
            .replace("<e2>", " <e2>") \
            .replace("</e2>", "</e2> ") \
            .replace(",", " , ") \
            .replace("'", " ' ") \
            .replace("!", " ! ") \
            .replace("?", " ? ")
        return [token.strip() for token in prepared_string.split(" ") if len(token.strip()) > 0]

    def _parse_args(self, tokens):
        tokens, first_args = self._parse_arg(tokens, 'e1')
        tokens, second_args = self._parse_arg(tokens, 'e2')
        return tokens, first_args, second_args

    @staticmethod
    def _parse_arg(tokens, arg_label):
        """
        Parses a relation argument with the given xml entity label.
        Returns the tokens without the xml entity label and the token offsets of the argument.
        """
        start_tag = '<' + arg_label + '>'
        end_tag = '</' + arg_label + '>'
        cleaned_tokens = []

        arg_start_idx = None
        arg_end_idx = None

        # track the index difference due to removed empty tokens
        cleaned_tokens_offset = 0

        for index, token in enumerate(tokens):

            if token.startswith(start_tag):
                arg_start_idx = index - cleaned_tokens_offset
                token = token[len(start_tag):]  # clean the tag from the token

            if token.endswith(end_tag):
                token = token[:-len(end_tag)]  # clean the tag from the token

                # If the current token is now empty, it is going to be removed
                # and the end offset will be a token earlier
                if DatasetConverter._is_empty_token(token):
                    arg_end_idx = index - cleaned_tokens_offset
                else:
                    arg_end_idx = index - cleaned_tokens_offset + 1

            if DatasetConverter._is_empty_token(token):
                cleaned_tokens_offset += 1
            else:
                cleaned_tokens.append(token)

        assert arg_start_idx is not None and arg_end_idx is not None, "Argument offsets could not be found"

        # argument_offsets = []
        # argument_offsets += list(range(-arg_start_idx, 0))  # Add negative offsets up to the argument
        # argument_offsets += [0] * (arg_end_idx-arg_start_idx)  # within the argument, all offsets are 0
        # argument_offsets += list(range(0, len(tokens) - arg_end_idx))  # add positive offsets after the argument

        return cleaned_tokens, (arg_start_idx, arg_end_idx)

    def _convert_tacred_format_file(self, input_file, output_file):
        with open(output_file, 'w') as output_file:
            for example in self._read_tacred_file(input_file):
                output_file.write(json.dumps(example) + "\n")

    def _read_tacred_file(self, input_file):
        with open(input_file, 'r') as input_file:
            input_examples = json.loads(input_file.readline())
            for input_example in input_examples:
                tokens = input_example['token']
                subj_offsets = (input_example['subj_start'], input_example['subj_end'] + 1)
                obj_offsets = (input_example['obj_start'], input_example['obj_end'] + 1)

                tokens = self.normalize_glove_tokens(tokens)

                output_example = {
                    "id": input_example['id'],
                    "tokens": tokens,
                    "label": input_example['relation'],
                    "entities": (subj_offsets, obj_offsets),
                    "grammar": ('SUBJ', 'OBJ'),
                    "type": (input_example['subj_type'], input_example['obj_type'])
                }

                yield output_example

    def normalize_glove_tokens(self, tokens):
        return [self.glove_mapping[token]
                if token in self.glove_mapping
                else token
                for token in tokens]

    @staticmethod
    def _is_empty_token(token):
        return len(token.strip()) == 0


def main(args):
    assert os.path.exists(args.dataset_dir), "Input directory does not exist"
    converter = DatasetConverter(args.dataset, args.dataset_dir, args.output_dir, args.subsample)
    converter.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help="The root directory of the dataset")
    parser.add_argument('output_dir', type=str, help="An output directory of jsonl files")
    parser.add_argument('--dataset', type=str, default="semeval", help="Either semeval, kbp37 or tacred")
    parser.add_argument('--subsample', action='store_true', help="Generate subsampled versions of the dataset with"
                                                                 " splits from 10% to 100% in 10% steps")

    args = parser.parse_args()
    print(args)
    main(args)
