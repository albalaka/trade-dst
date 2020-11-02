import argparse
from torch import cuda

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
ENT_token = 4

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
    parser.add_argument('--ground_truth_labels', action="store_true")
    parser.add_argument('--NER_labels', action="store_true")
    parser.add_argument('--percent_ground_truth', type=int, default=100)
    parser.add_argument('--no_binary_slots', action='store_true')
    parser.add_argument('--only_binary_slots', action='store_true')
    parser.add_argument('--no_categorical_slots', action='store_true')
    parser.add_argument('--no_binary_evaluation', action='store_true', help="remove binary slots from test/evaluate")
    parser.add_argument('--only_binary_evaluation', action='store_true')
    parser.add_argument('--no_categorical_evaluation', action='store_true')
    parser.add_argument('--only_categorical_evaluation', action='store_true')
    parser.add_argument('--boosted_NER_labels',action='store_true')

    args = parser.parse_args()

    assert(not (args.ground_truth_labels and args.NER_labels)), "Select only one of either ground truth, or NER labels"

    setattr(args, 'device', 'cuda' if cuda.is_available() else 'cpu')
    setattr(args, 'UNK_token', UNK_token)
    setattr(args, 'PAD_token', PAD_token)
    setattr(args, 'SOS_token', SOS_token)
    setattr(args, 'EOS_token', EOS_token)
    setattr(args, 'ENT_token', ENT_token)
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

    return vars(args)
