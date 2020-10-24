from tqdm import tqdm
import argparse
from torch import cuda
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

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

    optimizer = Adam(model.parameters(), lr=kwargs['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=kwargs['learning_rate']/100, verbose=True)

    gradient_accumulation_steps = kwargs['batch_size']/kwargs['MAX_GPU_SAMPLES']

    for epoch in range(200):
        print(f"Epoch {epoch}")
        if logger:
            logger.save()

        optimizer.zero_grad()

        # Initialize vars for std output
        total_loss = 0
        total_loss_pointer = 0
        total_loss_gate = 0

        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:

            # model.calculate_loss_batch(data, slot_list[1], logger, reset=True if i == 0 else False)

            # Calculate outputs
            outputs_pointer, outputs_gate, _ = model(data, slot_list[1])

            # Compute losses
            loss_pointer = model.calculate_loss_pointer(outputs_pointer, data['generate_y'], data['y_lengths'])
            loss_gate = model.calculate_loss_gate(outputs_gate, data['gating_label'])
            loss = loss_pointer + loss_gate

            # Calculate gradient
            loss.backward()

            # update vars for std output
            total_loss += loss.item()
            total_loss_pointer += loss_pointer.item()
            total_loss_gate += loss_gate.item()

            # update model weights
            if ((i+1) % gradient_accumulation_steps) == 0:
                clip_norm = clip_grad_norm_(model.parameters(), kwargs['clip'])
                optimizer.step()
                optimizer.zero_grad()

                # update logger
                if logger:
                    logger.training_update([
                        "training_batch",
                        {"loss": loss.item(),
                         "loss_pointer": loss_pointer.item(),
                         "loss_gate": loss_gate.item()}])

                # Update std output
                batch_num = ((i+1)/gradient_accumulation_steps)
                pbar.set_description(f"Loss: {total_loss/batch_num:.4f},Pointer loss: {total_loss_pointer/batch_num:.4f},Gate loss: {total_loss_gate/batch_num:.4f}")

        if ((epoch+1) % kwargs['eval_patience']) == 0:
            model.eval()
            accuracy = model.evaluate(dev, slot_list[2], avg_best, logger, kwargs['early_stopping'])
            model.train()
            scheduler.step(accuracy)

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
    parser.add_argument("--MAX_GPU_SAMPLES", type=int, default=MAX_GPU_SAMPLES)
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
