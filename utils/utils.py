import argparse
import re
import json
from torch import cuda

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
ENT_token = 4
SYS_token = 5
USR_token = 6

MAX_GPU_SAMPLES = 4
BINARY_SLOTS = ['hotel-parking', 'hotel-internet']
# CATEGORICAL_SLOTS = ['hotel-pricerange', 'hotel-book day', 'train-day', 'hotel-stars', 'restaurant-food', 'restaurant-pricerange', 'restaurant-book day']
CATEGORICAL_SLOTS = ['hotel-pricerange', 'hotel-book day', 'hotel-stars', 'hotel-area',
                     'train-day',
                     'attraction-area',
                     'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'restaurant-book day']
ALL_SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people',
             'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure',
             'train-arriveby', 'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food',
             'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name',
             'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure',
             'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ID", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("--MAX_GPU_SAMPLES", type=int, default=MAX_GPU_SAMPLES)
    parser.add_argument("--parallel_decode", type=bool, default=True)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-dr", "--dropout", type=float, default=0.2)
    parser.add_argument('-clip', '--clip', help='gradient clipping', default=10, type=int)
    parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--load_embedding', type=bool, default=True)
    parser.add_argument('--model_path', type=str, help="Use model_path if you want to load a pre-trained model")
    parser.add_argument('--lang_path', type=str, default="lang_data")
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--dataset', type=str, default='multiwoz')
    parser.add_argument('--task', type=str, default='DST')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--eval_patience', type=int, default=1)
    parser.add_argument('--gen_sample', action='store_true')
    parser.add_argument('--train_data_ratio', type=int, default=100)
    parser.add_argument('--dev_data_ratio', type=int, default=100)
    parser.add_argument('--test_data_ratio', type=int, default=100)
    parser.add_argument('--percent_ground_truth', type=int, default=100)
    parser.add_argument('--no_binary_slots', action='store_true')
    parser.add_argument('--only_binary_slots', action='store_true')
    parser.add_argument('--no_categorical_slots', action='store_true')
    parser.add_argument('--no_binary_evaluation', action='store_true', help="remove binary slots from test/evaluate")
    parser.add_argument('--only_binary_evaluation', action='store_true')
    parser.add_argument('--no_categorical_evaluation', action='store_true')
    parser.add_argument('--only_categorical_evaluation', action='store_true')
    parser.add_argument('--appended_values', type=str, default=None,
                        choices=['NER', 'ground_truth', 'boosted_NER', 'BERT_VE', 'DB'])
    parser.add_argument('--USR_SYS_tokens', action='store_true')
    parser.add_argument('--append_SYS_values', action='store_true')

    args = parser.parse_args()

    setattr(args, 'device', 'cuda' if cuda.is_available() else 'cpu')
    setattr(args, 'UNK_token', UNK_token)
    setattr(args, 'PAD_token', PAD_token)
    setattr(args, 'SOS_token', SOS_token)
    setattr(args, 'EOS_token', EOS_token)
    setattr(args, 'ENT_token', ENT_token)
    setattr(args, 'SYS_token', SYS_token)
    setattr(args, 'USR_token', USR_token)
    setattr(args, 'unk_mask', True)
    setattr(args, 'early_stopping', None)

    # if not using all slots, add them to drop slots
    setattr(args, 'drop_slots', list())
    if args.no_binary_slots:
        args.drop_slots.extend(BINARY_SLOTS)
    if args.no_categorical_slots:
        args.drop_slots.extend(CATEGORICAL_SLOTS)
    if args.only_binary_slots:
        args.drop_slots = ALL_SLOTS
        for slot in BINARY_SLOTS:
            args.drop_slots.remove(slot)

    setattr(args, "eval_slots", ALL_SLOTS)
    if args.only_binary_evaluation:
        args.eval_slots = BINARY_SLOTS
    if args.only_categorical_evaluation:
        args.eval_slots = CATEGORICAL_SLOTS
    if args.no_binary_evaluation:
        for slot in BINARY_SLOTS:
            args.eval_slots.remove(slot)
    if args.no_categorical_evaluation:
        for slot in CATEGORICAL_SLOTS:
            args.eval_slots.remove(slot)
    # print(f"Evaluating on {args.eval_slots}")
    if args.dataset == 'multiwoz_22':
        assert(args.lang_path == 'lang_data_multiwoz_22')

    assert(not(getattr(args, "appended_values") == "ground_truth" and getattr(args, "append_SYS_values"))),\
        "Ground truth values are not determined by the speaker, appending these values to the system utterance\
                will result in doubly appending values to the system utterance and user utterance"

    return vars(args)


# def find_database_value_in_utterance(utterance, database):
#     found_values = {}
#     for domain_slot, values in database.items():
#         # for val in values:
#         #     if val in utterance:
#         #         if domain_slot not in found_values.keys():
#         #             found_values[domain_slot] = []
#         #         found_values[domain_slot].append(val)

#         matches = re.findall(r"(?=("+'|'.join(values)+r"))", utterance)
#         if matches:
#             found_values[domain_slot] = matches
#     return found_values

def find_database_value_in_utterance(utterance, database):
    found_values = {}
    for domain_slot, values in database.items():

        matches = re.findall(r"(?=("+'|'.join(values)+r"))", utterance, re.IGNORECASE)
        if matches:
            found_values[domain_slot] = [m.strip() for m in matches]
    return found_values

def load_multiwoz_database(database_file="data/multi-woz/MULTIWOZ2 2/ontology.json"):  # "edited_ontology.json"
    # Returns all possible values from the database, as a dict
    ontology = json.load(open(database_file))
    return ontology


def load_multiwoz_22_database():
    # Returns all possible values from the database, as a dataset


    files_train = ["MultiWOZ_2.2/train/dialogues_001.json", "MultiWOZ_2.2/train/dialogues_002.json",
                "MultiWOZ_2.2/train/dialogues_003.json", "MultiWOZ_2.2/train/dialogues_004.json",
                "MultiWOZ_2.2/train/dialogues_005.json", "MultiWOZ_2.2/train/dialogues_006.json",
                "MultiWOZ_2.2/train/dialogues_007.json", "MultiWOZ_2.2/train/dialogues_008.json",
                "MultiWOZ_2.2/train/dialogues_009.json", "MultiWOZ_2.2/train/dialogues_010.json",
                "MultiWOZ_2.2/train/dialogues_011.json", "MultiWOZ_2.2/train/dialogues_012.json",
                "MultiWOZ_2.2/train/dialogues_013.json", "MultiWOZ_2.2/train/dialogues_014.json",
                "MultiWOZ_2.2/train/dialogues_015.json", "MultiWOZ_2.2/train/dialogues_016.json",
                "MultiWOZ_2.2/train/dialogues_017.json"]
    files_dev = ["MultiWOZ_2.2/dev/dialogues_001.json",
                "MultiWOZ_2.2/dev/dialogues_002.json"]
    files_test = ["MultiWOZ_2.2/test/dialogues_001.json",
                "MultiWOZ_2.2/test/dialogues_002.json"]

    noncat_slot_names = ["restaurant-food", "restaurant-name", "restaurant-booktime",
                        "attraction-name", "hotel-name", "taxi-destination",
                        "taxi-departure", "taxi-arriveby", "taxi-leaveat",
                        "train-arriveby", "train-leaveat"]
    cat_slot_names = ["restaurant-pricerange", "restaurant-area", "restaurant-bookday", "restaurant-bookpeople",
                    "attraction-area", "attraction-type", "hotel-pricerange", "hotel-parking",
                    "hotel-internet", "hotel-stars", "hotel-area", "hotel-type", "hotel-bookpeople",
                    "hotel-bookday", "hotel-bookstay", "train-destination", "train-departure",
                    "train-day", "train-bookpeople"]

    ontology = {k: set() for k in noncat_slot_names+cat_slot_names}

    for f in files_train+files_dev+files_test:
        dialogues = json.load(open(f))
        for dialogue_dict in dialogues:
            for turn in dialogue_dict['turns']:
                for frame in turn['frames']:
                    for slot in frame['slots']:
                        # check to make sure this domain-slot is in the domains that we care about
                        if slot['slot'] not in noncat_slot_names+cat_slot_names:
                            continue

                        if type(slot['value']) == list:
                            for v in slot['value']:
                                ontology[slot['slot']].add(f" {v} ")
                        else:
                            ontology[slot['slot']].add(f" {slot['value']} ")

                    # belief state only comes attached with user turns
                    if turn['speaker'] == "USER":
                        for ds, values in frame['state']['slot_values'].items():
                            # check to make sure this domain-slot is in the domains that we care about
                            if ds not in noncat_slot_names+cat_slot_names:
                                continue
                            for v in values:
                                ontology[ds].add(f" {v} ")
    return ontology
