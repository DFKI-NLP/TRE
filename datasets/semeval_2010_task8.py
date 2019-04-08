import json
from collections import Counter
from random import random
from itertools import product

import numpy as np

from os.path import join

from sklearn.model_selection import train_test_split


class SemEval2010Task8:
    TACRED_GRAMMAR_TYPES = ['obj', 'subj']
    TACRED_NER_TYPES = [
        'title', 'criminal_charge', 'url', 'ideology', 'location', 'cause_of_death', 'duration',
        'date', 'religion', 'person', 'nationality', 'city', 'country', 'organization', 'misc',
        'state_or_province', 'number'
    ]
    UNK_POS_TYPES = ['_first_entity_', '_second_entity_']
    UNK_TYPES = ['_entity_']

    TACRED_MASKED_TOKENS_COMBINED = [f'{grammar}-{type_}'.lower() for grammar, type_ in 
        product(TACRED_GRAMMAR_TYPES, TACRED_NER_TYPES)]

    TACRED_MASKED_TOKENS_NER = [f'_{type_}_'.lower() for type_ in TACRED_NER_TYPES]
    TACRED_MASKED_TOKENS_GRAMMAR = [f'_{type_}_'.lower() for type_ in TACRED_GRAMMAR_TYPES]

    MASKED_ENTITY_TOKENS = (
        TACRED_MASKED_TOKENS_COMBINED
        + TACRED_MASKED_TOKENS_NER
        + TACRED_MASKED_TOKENS_GRAMMAR
        + UNK_POS_TYPES
        + UNK_TYPES)


    @staticmethod
    def _subsample(sentences, entities, labels, ids, negative_label, subsampling_rate):
        subsampled_dataset = []
        dataset = zip(sentences, entities, labels, ids)

        for example in dataset:
            label = example[2]
            if label == negative_label:
                if random() < subsampling_rate:
                    subsampled_dataset.append(example)
            else:
                subsampled_dataset.append(example)

        return zip(*subsampled_dataset)

    @staticmethod
    def _mask_entities(tokens, entity_offsets, first_entity_replace, second_entity_replace):
        first_entity, second_entity = entity_offsets

        if first_entity[0] > second_entity[0]:
            tokens, first_entity, token_diff = SemEval2010Task8._replace_tokens(tokens, first_entity, first_entity_replace)
            tokens, second_entity, token_diff = SemEval2010Task8._replace_tokens(tokens, second_entity, second_entity_replace)

            first_entity = (first_entity[0] - token_diff, first_entity[1] - token_diff)
        else:
            tokens, second_entity, token_diff = SemEval2010Task8._replace_tokens(tokens, second_entity, second_entity_replace)
            tokens, first_entity, token_diff = SemEval2010Task8._replace_tokens(tokens, first_entity, first_entity_replace)

            second_entity = (second_entity[0] - token_diff, second_entity[1] - token_diff)

        return tokens, (first_entity, second_entity)

    @staticmethod
    def _replace_tokens(tokens, token_offsets, token):
        token_diff = token_offsets[1] - token_offsets[0] - 1
        tokens = tokens[:token_offsets[0]] + [token] + tokens[token_offsets[1]:]
        token_offsets = (token_offsets[0], token_offsets[0] + 1)

        return tokens, token_offsets, token_diff

    @staticmethod
    def _load_from_jsonl(path_to_file, is_test=True, masking_mode=None):
        sentences = []
        entities = []
        labels = []
        ids = []
        with open(path_to_file) as f:
            for line in f.readlines():
                example = json.loads(line)

                if masking_mode is not None:
                    example = SemEval2010Task8.apply_masking_mode(example, masking_mode)
                    
                sentences.append(example['tokens'])
                entities.append(example['entities'])
                if not is_test:
                    labels.append(example['label'])
                ids.append(example['id'])
        
        return sentences, entities, labels, ids

    @staticmethod
    def fetch(path_to_data, dev_size, seed, train_file='train.jsonl', test_file='test.jsonl', negative_label=None,
              subsampling_rate=1.0, train_set_limit=None, dev_set_limit=None, verbose=False, skip_test_set=False,
              predefined_dev_set=False, dev_file=None, masking_mode=None):

        if predefined_dev_set:
            if not dev_file:
                dev_file = 'dev.jsonl'

            sentences_train, entities_train, labels_train, ids_train = \
                SemEval2010Task8._load_from_jsonl(join(path_to_data, train_file), is_test=False, masking_mode=masking_mode)
            sentences_dev, entities_dev, labels_dev, ids_dev = \
                SemEval2010Task8._load_from_jsonl(join(path_to_data, dev_file), is_test=False, masking_mode=masking_mode)
        else:
            sentences_train_dev, entities_train_dev, labels_train_dev, ids_train_dev = \
                SemEval2010Task8._load_from_jsonl(join(path_to_data, train_file), is_test=False, masking_mode=masking_mode)
            sentences_train, sentences_dev, entities_train, entities_dev, labels_train, labels_dev, ids_train, ids_dev = \
                train_test_split(sentences_train_dev, entities_train_dev, labels_train_dev, ids_train_dev, test_size=dev_size, random_state=seed)

        if subsampling_rate < 1.0:
            assert negative_label is not None, "Negative class label required for subsampling"
            sentences_train, entities_train, labels_train, ids_train =\
                SemEval2010Task8._subsample(sentences_train, entities_train, labels_train, ids_train, negative_label, subsampling_rate)

        if train_set_limit:
            train_set = list(zip(sentences_train, entities_train, labels_train, ids_train))[:train_set_limit]
            sentences_train, entities_train, labels_train, ids_train = zip(*train_set)

        if dev_set_limit:
            dev_set = list(zip(sentences_dev, entities_dev, labels_dev, ids_dev))[:dev_set_limit]
            sentences_dev, entities_dev, labels_dev, ids_dev = zip(*dev_set)

        if verbose:
            train_label_counter = Counter(labels_train)
            print()
            print("Train set size: {}".format(len(ids_train)))
            print("Train set distribution:")
            for (label, count) in train_label_counter.items():
                print("{}: {}".format(label, count))
            print()

            if dev_set_limit:
                print()
                print("Dev set size: {}".format(len(ids_dev)))
                print()

        if not skip_test_set:
            sentences_test, entities_test, labels_test, ids_test = \
                SemEval2010Task8._load_from_jsonl(join(path_to_data, test_file), is_test=True, masking_mode=masking_mode)

            return (sentences_train, entities_train, labels_train, ids_train),\
                   (sentences_dev, entities_dev, labels_dev, ids_dev),\
                   (sentences_test, entities_test, labels_test, ids_test)
        else:
            return (sentences_train, entities_train, labels_train, ids_train), \
                   (sentences_dev, entities_dev, labels_dev, ids_dev)

        
    @staticmethod
    def encode(*splits, text_encoder, label_encoder):
        encoded_splits = []
        for split in splits:
            fields = []
            # encode sentence tokens
            fields.append(text_encoder.encode(split[0], special_tokens=SemEval2010Task8.MASKED_ENTITY_TOKENS))

            # encode entities
            encoded_entities = []
            for sentence, entities in zip(split[0], split[1]):
                encoded_entity = []
                for start, end in entities:
                    encoded_entity.append(text_encoder.encode([sentence[start: end]], special_tokens=SemEval2010Task8.MASKED_ENTITY_TOKENS)[0])
                encoded_entities.append(encoded_entity)
            fields.append(encoded_entities)

            # encode labels, if present
            encoded_labels = []
            for label in split[2]:
                if isinstance(label, str):
                    encoded_labels.append(label_encoder.add_item(label))
                else:
                    encoded_labels.append(label)
            fields.append(np.asarray(encoded_labels, dtype=np.int32))

            # pass through ids
            fields.append(split[3])

            # Add a none value for entity ids of datasets, that are not evaluated on a bag-level
            fields.append(None)

            encoded_splits.append(fields)
        return encoded_splits

    def transform(*splits, text_encoder, max_length, n_ctx, format='entities_first'):
        # TODO: add different input format
        # TODO: maybe max_length should be different for sentence and entities
        
        def transform(sentences, entities):
            batch_size = len(sentences)

            batch_indices = np.zeros((batch_size, 1, n_ctx, 2), dtype=np.int32)
            batch_mask = np.zeros((batch_size, 1, n_ctx), dtype=np.float32)

            encoder = text_encoder.encoder
            start = encoder['_start_']
            delimiter = encoder['_delimiter_']
            delimiter2 = encoder['_delimiter2_']
            clf_token = encoder['_classify_']

            n_vocab = len(encoder)
            
            for i, (sentence, entities), in enumerate(zip(sentences, entities)):
                input_sentence = [start]

                for entity in entities:
                    input_sentence.extend(entity[:max_length])
                    input_sentence.append(delimiter)

                input_sentence[-1] = delimiter2

                input_sentence = input_sentence + sentence[:max_length] + [clf_token]
                input_sentence_length = len(input_sentence)    

                batch_indices[i, 0, :input_sentence_length, 0] = input_sentence
                batch_mask[i, 0, :input_sentence_length] = 1
            
            # Position information that is added to the input embeddings in the TransformerModel
            batch_indices[:, :, :, 1] = np.arange(n_vocab, n_vocab + n_ctx)
            return batch_indices, batch_mask

        transformed_splits = []
        for sentences, entities, labels, ids, _ in splits:
            batch_indices, batch_mask = transform(sentences, entities)
            transformed_splits.append((batch_indices, batch_mask, labels, ids, None))

        return tuple(transformed_splits)

    @staticmethod
    def max_length(*splits, max_len):
        # TODO: do not clip the sentences to max_len, if the entities are smaller than max_len
        return max([
            len(sentence[:max_len]) + len(entities[0][:max_len]) + len(entities[1][:max_len])
            for split in splits
            for sentence, entities in zip(*split[0:2])
        ])

    @staticmethod
    def apply_masking_mode(example, masking_mode):
        masking_mode = masking_mode.lower()

        # TODO: that's kind of unsafe
        if 'grammar' in example:
            grammar_type = example['grammar']
        if 'type' in example:
            ner_type = example['type']

        if masking_mode == 'grammar':
            first_entity_replace, second_entity_replace = [f'_{g}_' for g in grammar_type]
        elif masking_mode == 'ner':
            first_entity_replace, second_entity_replace = [f'_{n}_' for n in ner_type]
        elif masking_mode == 'grammar_and_ner':
            first_entity_replace, second_entity_replace = [f'{g}-{n}' for g, n in zip(grammar_type, ner_type)]
        elif masking_mode == 'unk':
            first_entity_replace, second_entity_replace = SemEval2010Task8.UNK_TYPES[0], SemEval2010Task8.UNK_TYPES[0]
        elif masking_mode == 'unk_w_position':
            first_entity_replace, second_entity_replace = SemEval2010Task8.UNK_POS_TYPES
        else:
            raise ValueError(f"Masking mode '{masking_mode}' not supported.")

        example = example.copy()
        example['tokens'], example['entities'] = SemEval2010Task8._mask_entities(
            example['tokens'], example['entities'], first_entity_replace, second_entity_replace)

        return example
