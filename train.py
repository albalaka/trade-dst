from tqdm import tqdm
import argparse
from torch import cuda

from models.TRADE import TRADE
from utils.multiwoz import prepare_data
from utils.logger import simple_logger

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
ENT_token = 4

MAX_GPU_SAMPLES = 4


def main(**kwargs):
    logger = simple_logger(kwargs) if kwargs['log_path'] else None

    avg_best, count, accuracy = 0.0, 0, 0.0
    train, dev, _, lang, slot_list, gating_dict, vocab_size_train = prepare_data(training=True, **kwargs)

    model = TRADE(lang, slot_list, gating_dict, **kwargs)
    model.train()

    for epoch in range(200):
        print(f"Epoch {epoch}")
        if logger:
            logger.save()

        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data, slot_list[1], logger, reset=True if i == 0 else False)
            model.optimize(kwargs['clip'])
            pbar.set_description(model.print_loss())

        if ((epoch+1) % kwargs['eval_patience']) == 0:
            model.eval()
            accuracy = model.evaluate(dev, slot_list[2], avg_best, logger, kwargs['early_stopping'])
            model.train()
            model.scheduler.step(accuracy)

            if accuracy >= avg_best:
                avg_best = accuracy
                count = 0
                best_model = model
            else:
                count += 1

            if count == kwargs['patience'] or (accuracy == 1.0 and kwargs['early_stopping'] == None):
                if logger:
                    logger.save()
                print("ran out of patience, stopping early")
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=MAX_GPU_SAMPLES)
    parser.add_argument("--parallel_decode", type=bool, default=True)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-dr", "--dropout", type=float, default=0.2)
    parser.add_argument('-clip', '--clip', help='gradient clipping',
                        default=10, type=int)
    parser.add_argument('-tfr', '--teacher_forcing_ratio',
                        help='teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--load_embedding', type=bool, default=True)
    parser.add_argument('--model_path', type=str,
                        help="Use model_path if you want to load a pre-trained model")
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

    args = parser.parse_args()

    setattr(args, 'device', 'cuda' if cuda.is_available() else 'cpu')
    setattr(args, 'UNK_token', 0)
    setattr(args, 'PAD_token', 1)
    setattr(args, 'SOS_token', 2)
    setattr(args, 'EOS_token', 3)
    setattr(args, 'ENT_token', 4)
    setattr(args, 'unk_mask', True)
    setattr(args, 'early_stopping', None)

    main(**vars(args))
