import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm

from utils.masked_cross_entropy import masked_cross_entropy_for_value


class TRADE(torch.nn.Module):
    def __init__(self, lang, slots, gating_dict, **kwargs):
        super(TRADE, self).__init__()
        self.kwargs = kwargs
        self.hidden_size = kwargs['hidden']
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = kwargs['learning_rate']
        self.dropout = kwargs['dropout']
        # combined slots from all of train, dev, and test
        self.slots = slots[0]
        # self.slot_temp = slots[2] # slots from dev
        self.gating_dict = gating_dict
        self.num_gates = len(self.gating_dict)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.encoder = EncoderRNN(
            self.lang.n_words, self.hidden_size, self.dropout, self.kwargs['PAD_token'],
            self.kwargs['device'], self.kwargs['load_embedding'])
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words,
                                 self.hidden_size, self.dropout, self.slots, self.num_gates, self.kwargs['device'])

        if kwargs['model_path'] and 'enc.pt' in os.listdir(kwargs['model_path']):
            if self.kwargs['device'] == 'cuda':
                print("MODEL {} LOADED".format(kwargs['model_path']))
                trained_encoder = torch.load(kwargs['model_path']+'/enc.pt')
                trained_decoder = torch.load(kwargs['model_path']+'/dec.pt')
            else:
                print("MODEL {} LOADED".format(kwargs['model_path']))
                trained_encoder = torch.load(
                    kwargs['model_path']+'/enc.pt', lambda storage, loc: storage)
                trained_decoder = torch.load(
                    kwargs['model_path']+'/dec.pt', lambda storage, loc: storage)

            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
        #                                                             factor=0.5, patience=1, min_lr=self.lr/100, verbose=True)

        # self.reset()

        self.encoder.to(kwargs['device'])
        self.decoder.to(kwargs['device'])

    # def print_loss(self):
    #     print_loss_avg = self.loss / self.print_every
    #     print_loss_pointer = self.loss_pointer / self.print_every
    #     print_loss_gate = self.loss_gate / self.print_every
    #     # print_loss_class = self.loss_class / self.print_every
    #     # print_loss_domain = self.loss_domain / self.print_every
    #     self.print_every += 1
    #     return 'Average Loss:{:.2f},Average Pointer Loss:{:.2f},Average Gating Loss:{:.2f}'.format(print_loss_avg, print_loss_pointer, print_loss_gate)

    def save_model(self, score):
        directory = f"save/{self.kwargs['experiment_ID']}-TRADE-{self.kwargs['dataset']}{self.kwargs['task']}/HDD{self.hidden_size}-BSZ{self.kwargs['batch_size']}-DR{self.dropout}-{score}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.pt')
        torch.save(self.decoder, directory + '/dec.pt')
        print("MODEL SAVED")

    # def reset(self):
    #     self.loss, self.print_every, self.loss_pointer, self.loss_gate = 0, 1, 0, 0

    # def train_batch(self, data, slots, logger=None, reset=False):
    #     """
    #     Train a single batch
    #     :param data: a single batch of data from the dataloader
    #     :param slots: domain-value slots to include
    #     :param reset: reset overall loss, ptr loss, gating loss. Done at the first batch of every epoch
    #     """
    #     if reset:
    #         self.reset()
    #     self.optimizer.zero_grad()

    #     # Encode and Decode
    #     use_teacher_forcing = random.random() < self.kwargs['teacher_forcing_ratio']
    #     all_pointer_outputs, all_gate_outputs, _ = self.encode_and_decode(data, use_teacher_forcing, slots)

    #     loss_pointer = masked_cross_entropy_for_value(all_pointer_outputs.transpose(0, 1).contiguous(),
    #                                                   data['generate_y'].contiguous(), data['y_lengths'])
    #     loss_gate = self.cross_entropy(all_gate_outputs.transpose(0, 1).contiguous().view(-1, all_gate_outputs.shape[-1]), data['gating_label'].contiguous().view(-1))

    #     # keep loss for gradient separately
    #     loss = loss_pointer + loss_gate
    #     self.loss_grad = loss

    #     # log loss per batch
    #     if logger:
    #         logger.logger['training'].append(['training_batch', {
    #             'loss': loss.item(),
    #             'loss_pointer': loss_pointer.item(),
    #             'loss_gates': loss_gate.item()
    #         }])

    #     # track loss for each component just for plotting purposes
    #     self.loss += loss.data
    #     self.loss_pointer += loss_pointer.item()
    #     self.loss_gate += loss_gate.item()

    def forward(self, data, slots):
        """Forward method for TRADE model, encode a single batch of data, then decode
        :param data: a single batch of data from the dataloader
        :param slots: domain-value slots to include
        """

        # Encode and Decode
        use_teacher_forcing = random.random() < self.kwargs['teacher_forcing_ratio']
        all_pointer_outputs, all_gate_outputs, words = self.encode_and_decode(data, use_teacher_forcing, slots)

        return all_pointer_outputs, all_gate_outputs, words

    def calculate_loss_pointer(self, pointer_outputs, pointer_targets, target_lengths):
        # pointer outputs has shape (# slots, batch size, max target length, vocab size)
        # pointer targets has shape (batch size, # slots, max target length)
        return masked_cross_entropy_for_value(pointer_outputs.transpose(0, 1).contiguous(), pointer_targets.contiguous(), target_lengths)

    def calculate_loss_gate(self, gate_outputs, gate_targets):
        # gate outputs has shape (# slots, batch size, # gates)
        # gate targets has shape (batch size, # slots)
        return self.cross_entropy(gate_outputs.transpose(0, 1).contiguous().view(-1, gate_outputs.shape[-1]), gate_targets.contiguous().view(-1))

    # def calculate_loss_batch(self, data, slots, logger=None, reset=False):
    #     """Calculate loss, but not gradients, on a single batch
    #     :param data: a single batch of data from the dataloader
    #     :param slots: domain-value slots to include
    #     :param reset: reset overall loss, ptr loss, gating loss. Done at the first batch of every epoch
    #     """
    #     if reset:
    #         self.reset()

    #     # Encode and Decode
    #     use_teacher_forcing = random.random() < self.kwargs['teacher_forcing_ratio']
    #     all_pointer_outputs, all_gate_outputs, _ = self.encode_and_decode(data, use_teacher_forcing, slots)

    #     # loss_pointer = masked_cross_entropy_for_value(all_pointer_outputs.transpose(0, 1).contiguous(),
    #     #                                               data['generate_y'].contiguous(), data['y_lengths'])
    #     # loss_gate = self.cross_entropy(all_gate_outputs.transpose(0, 1).contiguous().view(-1, all_gate_outputs.shape[-1]), data['gating_label'].contiguous().view(-1))
    #     loss_pointer = self.calculate_loss_pointer(all_pointer_outputs, data['generate_y'], data['y_lengths'])
    #     loss_gate = self.calculate_loss_gate(all_gate_outputs, data['gating_label'])

    #     # keep loss for gradient separately
    #     loss = loss_pointer + loss_gate
    #     self.loss_grad = loss
    #     self.loss_grad.backward()

    #     # log loss per batch
    #     if logger:
    #         logger.logger['training'].append(['training_batch', {
    #             'loss': loss.item(),
    #             'loss_pointer': loss_pointer.item(),
    #             'loss_gates': loss_gate.item()
    #         }])

    #     # track loss for each component just for plotting purposes
    #     self.loss += loss.data
    #     self.loss_pointer += loss_pointer.item()
    #     self.loss_gate += loss_gate.item()

    # def optimize(self, clip):
    #     """
    #     :param clip: value to clip gradients at
    #     """
    #     clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
    #     self.optimizer.step()

    # def optimize(self, clip):
    #     """
    #     :param clip: value to clip gradients at
    #     """
    #     self.loss_grad.backward()
    #     clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
    #     self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slots):
        # if training, randomly mask tokens to encourage generalization
        if self.kwargs['unk_mask'] and self.decoder.training:
            # is the random mask required????
            # why not just go straight to binomial mask?
            story_shape = data['context'].shape
            random_mask = np.ones(story_shape)
            binomial_mask = np.random.binomial([np.ones((story_shape[0], story_shape[1]))], 1-self.dropout)[0]
            random_mask = random_mask * binomial_mask
            random_mask = torch.Tensor(random_mask).to(self.kwargs['device'])
            story = data['context'] * random_mask.long()

        else:
            story = data['context']

        # Encode the dialogue history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get list of words that can be copied
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_pointers = data['generate_y'].shape[2] if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_pointer_output = self.decoder.forward(batch_size,
                                                                                         encoded_hidden, encoded_outputs, data[
                                                                                             'context_len'], story, max_pointers, data['generate_y'],
                                                                                         use_teacher_forcing, slots)

        return all_point_outputs, all_gate_outputs, words_pointer_output

    def evaluate(self, dev, slots, metric_best=None, logger=None, early_stopping=True):
        print("EVALUATING ON DEV")
        all_predictions = {}
        inverse_gating_dict = dict([(v, k) for k, v in self.gating_dict.items()])
        for j, data_dev in enumerate(tqdm(dev)):
            batch_size = len(data_dev['context_len'])
            _, gates, words = self.encode_and_decode(data_dev, False, slots)

            for batch_idx in range(batch_size):
                if data_dev["ID"][batch_idx] not in all_predictions.keys():
                    all_predictions[data_dev['ID'][batch_idx]] = {}
                all_predictions[data_dev["ID"][batch_idx]][data_dev["turn_id"][batch_idx]] = {
                    "turn_belief": data_dev["turn_belief"][batch_idx]}
                predict_belief_bsz_ptr = []
                predicted_gates = torch.argmax(
                    gates.transpose(0, 1)[batch_idx], dim=1)

                for slot_idx, gate in enumerate(predicted_gates):
                    if gate == self.gating_dict['none']:
                        continue
                    elif gate == self.gating_dict['ptr']:
                        pred = np.transpose(words[slot_idx])[batch_idx]
                        st = []
                        for token in pred:
                            if token == 'EOS':
                                break
                            else:
                                st.append(token)
                        st = " ".join(st)
                        if st == 'none':
                            continue
                        else:
                            predict_belief_bsz_ptr.append(
                                f"{slots[slot_idx]}-{st}")
                    else:
                        predict_belief_bsz_ptr.append(
                            f"{slots[slot_idx]}-{inverse_gating_dict[gate.item()]}")

                all_predictions[data_dev["ID"][batch_idx]][data_dev["turn_id"]
                                                           [batch_idx]]["pred_beliefstate_ptr"] = predict_belief_bsz_ptr

                if set(data_dev['turn_belief'][batch_idx]) != set(predict_belief_bsz_ptr) and self.kwargs['gen_sample']:
                    print("True", set(data_dev["turn_belief"][batch_idx]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if self.kwargs['gen_sample']:
            json.dump(all_predictions, open(
                "all_prediction_{}.json".format(self.name), 'w'), indent=2)

        joint_acc_score, turn_acc_score, joint_F1_score, individual_slot_scores, joint_success, missing_slots, incorrect_slots = self.evaluate_metrics(
            all_predictions, "pred_beliefstate_ptr", slots)

        evaluation_metrics = {
            "Joint_accuracy": joint_acc_score,
            "Turn accuracy": turn_acc_score,
            "Joint F1": joint_F1_score
        }
        if logger:
            logger.logger['training'].append([['evaluation', evaluation_metrics],
                                              ['individual_slot_scores', individual_slot_scores],
                                              ['unique_joint_slots_success', len(joint_success)],
                                              ['top_100_joint_success', joint_success],
                                              ['unique_missing_slots', len(missing_slots)],
                                              ['top_50_missing_slots', missing_slots],
                                              ['unique_incorrect_slots', len(incorrect_slots)],
                                              ['top_50_incorrect_slots', incorrect_slots]
                                              ])
        print(evaluation_metrics)

        if (early_stopping == "F1"):
            if joint_F1_score >= metric_best:
                self.save_model('F1-{:.4f}'.format(joint_F1_score))
            return joint_F1_score
        else:
            if joint_acc_score >= metric_best:
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
            return joint_acc_score

    def test(self, test, slots, logger=None):
        print("EVALUATING ON TEST")
        all_predictions = {}
        inverse_gating_dict = dict([(v, k) for k, v in self.gating_dict.items()])
        for j, data_test in enumerate(tqdm(test)):
            batch_size = len(data_test['context_len'])
            _, gates, words = self.encode_and_decode(data_test, False, slots)

            for batch_idx in range(batch_size):
                if data_test["ID"][batch_idx] not in all_predictions.keys():
                    all_predictions[data_test['ID'][batch_idx]] = {}
                all_predictions[data_test["ID"][batch_idx]][data_test["turn_id"][batch_idx]] = {"turn_belief": data_test["turn_belief"][batch_idx]}
                predict_belief_bsz_ptr = []
                predicted_gates = torch.argmax(
                    gates.transpose(0, 1)[batch_idx], dim=1)

                for slot_idx, gate in enumerate(predicted_gates):
                    if gate == self.gating_dict['none']:
                        continue
                    elif gate == self.gating_dict['ptr']:
                        pred = np.transpose(words[slot_idx])[batch_idx]
                        st = []
                        for token in pred:
                            if token == 'EOS':
                                break
                            else:
                                st.append(token)
                        st = " ".join(st)
                        if st == 'none':
                            continue
                        else:
                            predict_belief_bsz_ptr.append(f"{slots[slot_idx]}-{st}")
                    else:
                        predict_belief_bsz_ptr.append(f"{slots[slot_idx]}-{inverse_gating_dict[gate.item()]}")

                all_predictions[data_test["ID"][batch_idx]][data_test["turn_id"]
                                                            [batch_idx]]["pred_beliefstate_ptr"] = predict_belief_bsz_ptr

                if self.kwargs['gen_sample'] and set(data_test['turn_belief'][batch_idx]) != set(predict_belief_bsz_ptr):
                    print("True", set(data_test["turn_belief"][batch_idx]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if self.kwargs['gen_sample']:
            json.dump(all_predictions, open(
                "all_prediction_{}.json".format(self.name), 'w'), indent=2)

        joint_acc_score, turn_acc_score, joint_F1_score, individual_slot_scores, joint_success, missing_slots, incorrect_slots = self.evaluate_metrics(
            all_predictions, "pred_beliefstate_ptr", slots)

        # ANALYSIS: print out results from individual slot analysis, and joint analysis
        # print("JOINT SUCCESS:")
        # for k in list(joint_success.keys())[:20]:
        #     print(f"{k} - {joint_success[k]}")
        # print(f"\nJOINT FAILURE - MISSING SLOTS:")
        # for k in list(missing_slots.keys())[:20]:
        #     print(f"{k} - {missing_slots[k]}")
        # print(f"\nJOINT FAILURE - INCORRECT SLOTS:")
        # for k in list(incorrect_slots.keys())[:20]:
        #     print(f"{k} - {incorrect_slots[k]}")

        evaluation_metrics = {
            "Joint_accuracy": joint_acc_score,
            "Turn accuracy": turn_acc_score,
            "Joint F1": joint_F1_score
        }
        if logger:
            logger.logger['testing'].append([['evaluation', evaluation_metrics],
                                             ['individual_slot_scores', individual_slot_scores],
                                             ['unique_joint_slots_success', len(joint_success)],
                                             ['top_100_joint_success', joint_success],
                                             ['unique_missing_slots', len(missing_slots)],
                                             ['top_50_missing_slots', missing_slots],
                                             ['unique_incorrect_slots', len(incorrect_slots)],
                                             ['top_50_incorrect_slots', incorrect_slots]
                                             ])
        print(evaluation_metrics)

    def evaluate_metrics(self, all_predictions, from_which, slots):
        """
        :param all_predictions: dict of dialogues, each dialogue contains turns with ground truth turn beliefs and predicted beliefs
        :param from_which: which prediction method are we comparing
        :param slots: domain-slot names
        """
        # ANALYSIS
        individual_slot_scores = {
            slot: {
                "TP": 0,
                "FP": 0,
                "FN": 0}
            for slot in slots}
        joint_success = {}
        joint_failure = {"missing_slots": {}, "incorrect_slots": {}}

        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0

        for datum_ID, turns in all_predictions.items():
            for turn_idx, turn in turns.items():

                # ANALYSIS: compute score for each slot individually
                for slot in turn['turn_belief']:
                    if slot in turn[from_which]:
                        individual_slot_scores[slot.rsplit("-", 1)[0]]["TP"] += 1
                    else:
                        individual_slot_scores[slot.rsplit("-", 1)[0]]["FN"] += 1
                for slot in turn[from_which]:
                    if slot not in turn['turn_belief']:
                        individual_slot_scores[slot.rsplit("-", 1)[0]]["FP"] += 1

                # compute joint goal accuracy per turn
                if set(turn['turn_belief']) == set(turn[from_which]):
                    joint_acc += 1
                    # ANALYSIS on succesful joint accuracy
                    if str(turn['turn_belief']) not in joint_success.keys():
                        joint_success[str(turn['turn_belief'])] = 1
                    else:
                        joint_success[str(turn['turn_belief'])] += 1
                # ANALYSIS on failed joint accuracy
                else:
                    missing_slots = set(turn['turn_belief'])-set(turn[from_which])
                    for slot in missing_slots:
                        if slot not in joint_failure['missing_slots'].keys():
                            joint_failure["missing_slots"][slot] = 1
                        else:
                            joint_failure["missing_slots"][slot] += 1

                    incorrect_slots = set(turn[from_which])-set(turn['turn_belief'])
                    for slot in incorrect_slots:
                        if slot not in joint_failure['incorrect_slots'].keys():
                            joint_failure["incorrect_slots"][slot] = 1
                        else:
                            joint_failure["incorrect_slots"][slot] += 1

                total += 1

                # compute slot accuracy
                temp_acc = self.compute_slot_acc(set(turn['turn_belief']), set(turn[from_which]), slots)
                turn_acc += temp_acc

                # compute joint F1 score
                temp_F1, temp_recall, temp_precision = self.compute_precision_recall_F1(set(turn['turn_belief']), set(turn[from_which]))
                F1_pred += temp_F1

        # joint accuracy requires the each slot within a turn to be correct to get a point
        joint_accuracy = joint_acc/total if total > 0 else 0
        # turn accuracy considers each slot separately within a single turn
        turn_accuracy = turn_acc/total if total > 0 else 0
        # F1 is calculated jointly across all slots in a single turn
        F1_score = F1_pred/total if total > 0 else 0

        joint_success = {k: v for k, v in sorted(joint_success.items(), key=lambda item: item[1], reverse=True)}
        missing_slots = {k: v for k, v in sorted(joint_failure['missing_slots'].items(), key=lambda item: item[1], reverse=True)}
        incorrect_slots = {k: v for k, v in sorted(joint_failure['incorrect_slots'].items(), key=lambda item: item[1], reverse=True)}

        return joint_accuracy, turn_accuracy, F1_score, individual_slot_scores, joint_success, missing_slots, incorrect_slots

    def compute_slot_acc(self, gold, pred, slots):
        FN = 0
        missed_slot = []
        for slot in gold:
            if slot not in pred:
                FN += 1
                missed_slot.append(slot.rsplit("-", 1)[0])
        FP = 0
        for slot in pred:
            if slot not in gold and slot.rsplit("-", 1)[0] not in missed_slot:
                FP += 1

        # Their version, maybe incorrect
        total = len(slots)
        accuracy = (total-FN-FP)/total

        # my version of accuracy
        acc = (total-FN)/(total+FP)

        return accuracy

    def compute_precision_recall_F1(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) > 0:
            for turn in gold:
                if turn in pred:
                    TP += 1
                else:
                    FN += 1
            for turn in pred:
                if turn not in gold:
                    FP += 1
            precision = TP/(TP+FP) if (TP+FP) > 0 else 0
            recall = TP/(TP+FN) if (TP+FN) > 0 else 0
            F1 = (2 * precision * recall) / \
                (precision + recall) if (precision + recall) > 0 else 0

        else:
            if len(pred) == 0:
                F1, recall, precision = 1, 1, 1
            else:
                F1, recall, precision = 0, 0, 0
        return F1, recall, precision


class EncoderRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, PAD_token, device, load_embedding, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        if load_embedding:
            with open(os.path.join('data', f'emb{self.vocab_size}.json')) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

    def get_state(self, batch_size):
        """Get cell states and hidden states"""
        return torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device)

    def forward(self, input_sequences, input_lengths=None):
        embedded = self.embedding(input_sequences)
        embedded = self.dropout_layer(embedded)

        hidden = self.get_state(input_sequences.size(1))
        if input_lengths:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(torch.nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, num_gates, device):
        super(Generator, self).__init__()
        self.lang = lang
        self.embedding = shared_emb
        self.vocab_size = vocab_size
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.num_gates = num_gates
        self.hidden_size = hidden_size
        self.W_ratio = torch.nn.Linear(3*self.hidden_size, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.slots = slots
        self.device = device

        self.W_gate = torch.nn.Linear(hidden_size, num_gates)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        # initialize slot embeddings
        self.Slot_emb = torch.nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs,
                encoded_lengths, story, max_pointers, target_batches, use_teacher_forcing, slots):

        # initialize tensors for pointers and gates
        all_pointer_outputs = torch.zeros([len(slots), batch_size, max_pointers, self.vocab_size], device=self.device)
        all_gate_outputs = torch.zeros([len(slots), batch_size, self.num_gates], device=self.device)

        # Get slot embeddings
        slot_emb_dict = {}
        for i, slot in enumerate(slots):
            # Domain embedding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx, device=self.device)
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx, device=self.device)
                slot_emb = self.Slot_emb(slot_w2idx)

            # combine domain and slot embeddings by addition
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            # Duplicate/expand the domain+slot embeddings for each datum in the batch
            slot_emb_expanded = combined_emb.expand_as(encoded_hidden)
            # Duplicate/expand the (domain+slot)*batch_size embeddings for each slot
            if i == 0:
                slot_emb_arr = slot_emb_expanded.clone()
            else:
                slot_emb_arr = torch.cat(
                    (slot_emb_arr, slot_emb_expanded), dim=0)

        # In this implementation we only use parallel decode
        # if self.kwargs['parallel_decode']:

        # Compute pointer-generator output, with all (domain, slot) pairs in a single batch
        decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size)  # (batch*|slot|) * emb
        hidden = encoded_hidden.repeat(1, len(slots), 1)  # 1 * (batch*|slot|) * emb
        words_point_out = [[] for i in range(len(slots))]

        for word_idx in range(max_pointers):
            dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

            enc_out = encoded_outputs.repeat(len(slots), 1, 1)
            enc_len = encoded_lengths * len(slots)
            context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

            if word_idx == 0:
                all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

            p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
            p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
            vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size(), device=self.device)

            p_context_ptr.scatter_add_(1, story.repeat(len(slots), 1), prob)

            final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
            pred_word = torch.argmax(final_p_vocab, dim=1)
            words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]

            for si in range(len(slots)):
                words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])

            all_pointer_outputs[:, :, word_idx, :] = torch.reshape(final_p_vocab, (len(slots), batch_size, self.vocab_size))

            if use_teacher_forcing:
                decoder_input = self.embedding(torch.flatten(target_batches[:, :, word_idx].transpose(1, 0)))
            else:
                decoder_input = self.embedding(pred_word)

        return all_pointer_outputs, all_gate_outputs, words_point_out

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = torch.nn.functional.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = torch.nn.functional.softmax(scores_, dim=1)
        return scores
