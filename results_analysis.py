import json
import os

# Functions for loading data
def load_log(log_name):
    return json.load(open(os.path.join("logs", log_name)))


def get_metadata(experiment_ID):
    return load_log(experiment_ID)['metadata']


def get_training_evaluation_data(experiment_ID):
    log = load_log(experiment_ID)['training']
    training_data = []
    eval_data = []
    for batch in log:
        if batch[0] == 'evaluation':
            eval_data.append(batch[1])
        else:
            training_data.append(batch[1])
    return training_data, eval_data


def get_testing_data(experiment_ID):
    return load_log(experiment_ID)['testing']


def get_training_data(experiment_ID):
    training_data, _ = get_training_evaluation_data(experiment_ID)
    return training_data

# Function for getting training loss
def get_training_loss(experiment_ID):
    training_data = get_training_data(experiment_ID)
    loss = []
    loss_ptr = []
    loss_gate = []
    for batch in training_data:
        loss.append(batch["loss"])
        loss_ptr.append(batch["loss_pointer"])
        loss_gate.append(batch["loss_gate"])
    return loss, loss_ptr, loss_gate


def get_evaluation_data(experiment_ID):
    _, eval_data = get_training_evaluation_data(experiment_ID)
    return eval_data

# Functions for evaluation metrics - Joint accuracy, Turn accuracy, Joint F1 score
def get_single_eval_metric(eval_dict):
    """
    Takes as input a single evaluation dict
    Returns just the evaluation_metrics portion
    """
    eval_metrics = eval_dict['evaluation_metrics']
    return eval_metrics['Joint_accuracy'], eval_metrics['Turn accuracy'], eval_metrics['Joint F1']


def get_all_evaluation_eval_metrics(experiment_ID):
    """
    :param experiment_ID: path of log.json file
    Returns joint accuracies, turn accuracies, and joint F1 scores for all evaluations done during training 
    """
    eval_data = get_evaluation_data(experiment_ID)
    joint_accs, turn_accs, joint_F1s = [], [], []
    for e in eval_data:
        res = get_single_eval_metric(e)
        joint_accs.append(res[0])
        turn_accs.append(res[1])
        joint_F1s.append(res[2])

    return joint_accs, turn_accs, joint_F1s

# Functions for getting TP, FP, FN for each domain-slot pair
def get_single_slot_scores(eval_dict):
    """
    Slot scores are the number of TP, FP, and FN for an individual slot (eg. hotel-pricerange)
    Returns the slot scores (eg. 
            {'hotel-pricerange': {'TP': 10,'FP':5,'FN':4}, 'hotel-parking': {'TP':0,'FP':10,'FN':5}, etc.)
    """
    return eval_dict['individual_slot_scores']

def get_testing_slot_scores(experiment_ID):
    test_data = get_testing_data(experiment_ID)
    return get_single_slot_scores(test_data)


def get_all_evaluation_slot_scores(experiment_ID):
    """
    Reorganizes slot scores, instead of a single dict entry representing the results of an epoch (TP,FP,FN),
        we return a dict such that each entry is a list containing all scores over training epochs
    :param experiment_ID: path of log.json file
    Returns a dict of lists of all slot scores (eg. 
                        'hotel-pricerange': {"TP":[all TP scores], "FP":[all FP scores], "FN":[all FN scores]})
    """
    eval_data = get_evaluation_data(experiment_ID)
    # convert raw evaluation metrics into a list of just slot scores
    slot_scores = [get_single_slot_scores(e) for e in eval_data]
    reorganized_slot_scores = {e: {"TP": [], "FP": [], "FN": [], "F1": []} for e in slot_scores[0]}
    # reorganize slot scores from a per-epoch basis to a per-score basis
    for scores in slot_scores:
        for slot in scores:
            reorganized_slot_scores[slot]["TP"].append(scores[slot]["TP"])
            reorganized_slot_scores[slot]["FP"].append(scores[slot]["FP"])
            reorganized_slot_scores[slot]["FN"].append(scores[slot]["FN"])
            reorganized_slot_scores[slot]["F1"].append(
                scores[slot]["TP"]/(scores[slot]["TP"]+(scores[slot]["FP"]+scores[slot]["FN"])/2)
            )
    return reorganized_slot_scores

# Functions for getting (possibly multiple) successful joint domain-slot-value pairs
# These are pairs that all must be correct to count towards joint accuracy
def get_single_unique_joint_slot_success(eval_dict):
    """
    A single joint success is when all slots were correctly labeled
    Returns the number of unique joint slots that were correctly labelled
    """
    return eval_dict['unique_joint_slots_success']


def get_all_evaluation_unique_joint_slot_successes(experiment_ID):
    eval_data = get_evaluation_data(experiment_ID)
    return [get_single_unique_joint_slot_success(e) for e in eval_data]


def get_single_top_k_joint_slot_success(eval_dict, k):
    """
    Takes as input a single evaluation dict
    A single joint success is when all slots were correctly labeled
    Returns the top k joint slots which were correctly labeled and frequencies
    """
    js = eval_dict['joint_success']
    return {k: js[k] for k in list(js.keys())[:k]}


def get_first_top_k_joint_slot_success(experiment_ID, k):
    """
    :param experiment_ID: path of log.json file
    Returns a dict of all joint slots successfully labeled and their frequencies
        for the first evaluation
    """
    eval_data = get_evaluation_data(experiment_ID)
    return get_single_top_k_joint_slot_success(eval_data[0], k)


def get_final_top_k_joint_slot_success(experiment_ID, k):
    """
    :param experiment_ID: path of log.json file
    Returns a dict of all joint slots successfully labeled and their frequencies
        for the final evaluation
    """
    eval_data = get_evaluation_data(experiment_ID)
    return get_single_top_k_joint_slot_success(eval_data[-1], k)

def get_testing_joint_slot_succes(experiment_ID):
    test_data = get_testing_data(experiment_ID)



def get_all_evaluation_top_k_joint_slot_successes(experiment_ID, k):
    """
    Reorganizes joint slot successes, instead of a single dict entry representing the successes of an epoch,
            we return a dict, whose keys are the joint-slot names and each entry is a list of successes per epoch
    :param experiment_ID: path of log.json file
    Returns a dict of lists for the top k most successful joint-slot names, eg.
            {"['attraction-area-centre']": [1,1,2,3,4,5], "['attraction-type-museum'],['attraction-area-centre']: [2,2,2,0,4,5,6], ...}
            Note that a "joint-slot" may actually only contain a single slot-value
            Note also that each joint slot also includes its value
    """

    eval_data = get_evaluation_data(experiment_ID)
    all_top_k_joint_slot_successes = [get_single_top_k_joint_slot_success(e, k) for e in eval_data]
    # set comprehension, get all unique joint-slot names
    all_joint_slots = {k for slots in all_top_k_joint_slot_successes for k in slots.keys()}
    reorganized_top_k_joint_slot_successes = {slot: [] for slot in all_joint_slots}
    for slot in all_joint_slots:
        for single_top_k_joint_slot_success in all_top_k_joint_slot_successes:
            successes = 0 if slot not in single_top_k_joint_slot_success else single_top_k_joint_slot_success[slot]
            reorganized_top_k_joint_slot_successes[slot].append(successes)
    return reorganized_top_k_joint_slot_successes


def get_all_evaluation_individual_joint_slot_success(experiment_ID, slot):
    """
    :param experiment_ID: path of log.json file
    Returns a list of the slot success for an individual slot-value
    """
    eval_data = get_evaluation_data(experiment_ID)
    individual_joint_slot_successes = [e['joint_success'][slot] if slot in e['joint_success'] else 0 for e in eval_data]
    return individual_joint_slot_successes

# Functions for getting FN slots
def get_single_unique_FN_slots(eval_dict):
    """
    An FN slot is a slot value that existed in the ground truth, but was not correctly labeled
    Returns the number of unique slot-values that were missed by the DST (usually just the number of unique slot-values)
    """
    return eval_dict['unique_FN_slots']


def get_all_evaluation_unique_FN_slots(experiment_ID):
    """
    Returns the number of slots with FNs per epoch
    """
    eval_data = get_evaluation_data(experiment_ID)
    return [get_single_unique_FN_slots(e) for e in eval_data]


def get_single_top_k_FN_slots(eval_dict, k):
    """
    An FN slot is a slot value that existed in the ground truth,
        but was not correctly labeled
    Returns the top k FN slots and frequencies
    """
    fn = eval_dict['FN_slots']
    return {k: fn[k] for k in list(fn.keys())[:k]}


def get_all_evaluation_top_k_FN_slots(experiment_ID, k):
    """
    Reorganizes FN slots, instead of a single dict entry representing the slots with FNs of an epoch,
            we return a dict, whose keys are the joint-slot names and each entry is a list of FNs per epoch
    :param experiment_ID: path of log.json file
    Returns a dict of lists of the top k FN slots
        eg. {'hotel-pricerange':[10,5,4,2,2,2,1], etc}
    """
    eval_data = get_evaluation_data(experiment_ID)
    all_top_k_FN_slots = [get_single_top_k_FN_slots(e, k) for e in eval_data]
    # set comprehension, get all FN slot names in the top k of any epoch
    all_FN_slots = {k for slots in all_top_k_FN_slots for k in slots.keys()}
    reorganized_top_k_FN_slots = {slot: [] for slot in all_FN_slots}
    for slot in all_FN_slots:
        for single_top_k_FN_slot in all_top_k_FN_slots:
            FN = 0 if slot not in single_top_k_FN_slot else single_top_k_FN_slot[slot]
            reorganized_top_k_FN_slots[slot].append(FN)
    return reorganized_top_k_FN_slots


def get_all_evaluation_individual_FN_slot(experiment_ID, slot):
    """
    :param experiment_ID: path of log.json file
    Returns a list of the slot success for an individual slot-value
    """
    eval_data = get_evaluation_data(experiment_ID)
    individual_FN_slot = [e['FN_slots'][slot] if slot in e['FN_slots'] else 0 for e in eval_data]
    return individual_FN_slot

# Functions for getting FP slots
def get_single_unique_FP_slots(eval_dict):
    """
    An FP slot is a slot value that was labelled by the DST, but not the same label as the ground truth
    Returns the number of unique FP slot-values (usually just the number of unique slot-values) 
    """
    return eval_dict['unique_FP_slots']


def get_all_evaluation_unique_FP_slots(experiment_ID):
    """
    Returns the number of slots with FPs per epoch
    """
    eval_data = get_evaluation_data(experiment_ID)
    return [get_single_unique_FP_slots(e) for e in eval_data]


def get_single_top_k_FP_slots(eval_dict, k):
    """
    An FP slot is a slot value that was labelled by the DST, but not the same label as the ground truth
    Returns the top k FP slots and frequencies
    """
    fp = eval_dict['FP_slots']
    return {k: fp[k] for k in list(fp.keys())[:k]}


def get_all_evaluation_top_k_FP_slots(experiment_ID, k):
    """
    Reorganizes FP slots, instead of a single dict entry representing the slots with FPs of an epoch,
            we return a dict, whose keys are the joint-slot names and each entry is a list of FPs per epoch
    :param experiment_ID: path of log.json file
    Returns a dict of lists of the top k FP slots
        eg. {'hotel-pricerange':[10,5,4,2,2,2,1], etc}
    """
    eval_data = get_evaluation_data(experiment_ID)
    all_top_k_FP_slots = [get_single_top_k_FP_slots(e, k) for e in eval_data]
    # set comprehension, get all FP slot names in the top k of any epoch
    all_FP_slots = {k for slots in all_top_k_FP_slots for k in slots.keys()}
    reorganized_top_k_FP_slots = {slot: [] for slot in all_FP_slots}
    for slot in all_FP_slots:
        for single_top_k_FP_slot in all_top_k_FP_slots:
            FP = 0 if slot not in single_top_k_FP_slot else single_top_k_FP_slot[slot]
            reorganized_top_k_FP_slots[slot].append(FP)
    return reorganized_top_k_FP_slots


def get_all_evaluation_individual_FP_slot(experiment_ID, slot):
    """
    :param experiment_ID: path of log.json file
    Returns a list of the slot success for an individual slot-value
    """
    eval_data = get_evaluation_data(experiment_ID)
    individual_FP_slot = [e['FP_slots'][slot] if slot in e['FP_slots'] else 0 for e in eval_data]
    return individual_FP_slot
