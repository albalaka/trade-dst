import argparse
from torch import cuda

from models.TRADE import TRADE
from utils.multiwoz import prepare_data
from utils.logger import simple_logger
import utils.utils


def main(**kwargs):

    logger = simple_logger(kwargs) if kwargs['log_path'] else None

    _, _, test, lang, slot_list, gating_dict, _ = prepare_data(training=False, **kwargs)

    model = TRADE(lang, slot_list, gating_dict, **kwargs)
    model.eval()

    model.test(test, slot_list[3], logger)

    if logger:
        logger.save()


if __name__ == "__main__":

    main(**utils.utils.parse_args())
