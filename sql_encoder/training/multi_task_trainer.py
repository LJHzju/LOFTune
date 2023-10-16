from sql_encoder.data_utils.threaded_iterator import ThreadedIterator
from sql_encoder.training.multi_task_model import PretrainSQLEncoder
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .base_trainer import BaseTrainer
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MulticlassF1Score
from config.encoder_config import *


class MultiTaskTrainer(BaseTrainer):
    def __init__(self):
        super(MultiTaskTrainer, self).__init__()

        self.subtree_vocab = Tokenizer.from_file(data_path['subtree_vocab_model_path'])

        subtree_prediction_config['num_objects'] = self.subtree_vocab.get_vocab_size()

        self.model = PretrainSQLEncoder(encoder_config, subtree_prediction_config, sequence_tagging_config).to(
            self.device)
        self.num_labels = sequence_tagging_config['num_labels']
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': train_config['lr'], 'weight_decay': train_config['weight_decay']},
        ])
        lr_step = lambda epoch: 1 if epoch < 10 else (0.2 if epoch < 20 else 0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_step, verbose=True)

        self.f1 = MultilabelF1Score(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(self.device)
        self.recall = MultilabelRecall(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(
            self.device)
        self.precision = MultilabelPrecision(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(
            self.device)

    def augment_data(self, node_type, node_tokens, children_index, task_data):
        node_tag = task_data['node_tag'].unsqueeze(-1)
        bs, n_node, n_token = node_tokens.shape
        node_type_repeat = node_type.unsqueeze(-1)
        node_tokens_tmp = node_tokens != 0

        pos_num, neg_num_ran = 1, 6
        pos_mask = (node_tag <= 0) & ((node_type_repeat == 314) | (node_type_repeat == 387)) & node_tokens_tmp
        pos_mask = pos_mask.unsqueeze(1).expand(-1, pos_num, -1, -1)
        pos_pairs = node_tokens.unsqueeze(1).repeat(1, pos_num, 1, 1)
        pos_pairs[pos_mask] = torch.randint(8, 9999, pos_pairs[pos_mask].shape, device=device, dtype=pos_pairs.dtype)
        pos_pairs = pos_pairs.view(-1, n_node, n_token)

        neg_mask = (node_tag == 1) & node_tokens_tmp

        # Repeating the tensor in the batch dimension to get A B C D A B C D ...
        neg_pairs_ran = node_tokens.repeat(neg_num_ran, 1, 1)
        # Create a repeated mask for the new neg_pairs_ran
        repeated_mask = neg_mask.repeat(neg_num_ran, 1, 1)
        neg_pairs_ran[repeated_mask] = torch.randint(8, 9999, neg_pairs_ran[repeated_mask].shape, device=device, dtype=neg_pairs_ran.dtype)
        neg_pairs_ran = neg_pairs_ran.view(-1, n_node, n_token)

        node_type = node_type.repeat(1 + pos_num + neg_num_ran, 1)
        node_tokens = torch.cat([node_tokens, pos_pairs, neg_pairs_ran])
        children_index = children_index.repeat(1 + pos_num + neg_num_ran, 1, 1)

        task_data['node_tag'] = task_data['node_tag'].repeat(1 + pos_num + neg_num_ran, 1)
        if 'subtree' in task_data.keys():
            # batch_size × subtree_vocab_size
            task_data['subtree_labels'] = task_data['subtree_labels'].repeat(1 + pos_num + neg_num_ran, 1)

        return node_type, node_tokens, children_index, task_data

    def load_batch_task_data(self, batch_data, TASK_WEIGHTS):
        batch_task_data = {}
        if TASK_WEIGHTS[0] != 0.0:
            batch_subtree_id = batch_data["batch_subtree_id"]
            # dtype=torch.float32 if device = torch.device('cpu')
            batch_subtree_labels = torch.zeros((self.batch_size, self.subtree_vocab.get_vocab_size()),
                                               dtype=torch.float16, device=self.device)
            for index, _1 in enumerate(batch_subtree_id):
                batch_subtree_labels[index][_1] = 1.0
            batch_task_data['subtree_labels'] = batch_subtree_labels
            batch_task_data['subtree'] = True

        if TASK_WEIGHTS[1] != 0.0:
            batch_task_data['contrastive'] = True

        if TASK_WEIGHTS[1] != 0.0 or TASK_WEIGHTS[2] != 0.0:
            # dtype=torch.float32 if device = torch.device('cpu')
            batch_node_tag = batch_data['batch_node_tag']
            batch_node_tag = [torch.tensor(node_tag, dtype=torch.int8) for node_tag in batch_node_tag]
            padded_node_tag = pad_sequence(batch_node_tag, batch_first=True, padding_value=-1).to(self.device)

            children_index = batch_data['batch_children_index']
            child_index_sum = np.sum(children_index, axis=2)
            not_leaf_index = [np.where(row != 0)[0].tolist() for row in child_index_sum]
            for i in range(len(padded_node_tag)):
                padded_node_tag[i][not_leaf_index[i]] = -1
            batch_task_data['node_tag'] = padded_node_tag
            if TASK_WEIGHTS[2] != 0.0:
                batch_task_data['tagging'] = True
        return batch_task_data

    def eval(self):
        eval_batch_iterator = ThreadedIterator(self.eval_data_loader.make_minibatch_iterator(self.evaluation_buckets),
                                               max_queue_size=2 * self.batch_size)
        overall_eval_loss_all = 0.0
        task_eval_loss_all = [0.0, 0.0, 0.0]
        tagging_recall = [0.0 for _ in range(self.num_labels)]
        contrastive_f1_value = 0.0
        TASK_WEIGHTS = [1.0, 1.0, 1.0]

        steps = 0
        for _, eval_batch_data in enumerate(eval_batch_iterator):
            self.model.eval()
            with torch.no_grad():
                node_type, node_tokens, children_index = self.load_batch_tree_data(eval_batch_data)
                task_data = self.load_batch_task_data(eval_batch_data, TASK_WEIGHTS)
                if TASK_WEIGHTS[1] != 0.0:
                    node_type, node_tokens, children_index, task_data = self.augment_data(node_type, node_tokens, children_index, task_data)

                prediction, contrastive, tagging = self.model(node_type, node_tokens, children_index, task_data)
                loss = TASK_WEIGHTS[0] * prediction['loss'].item() + \
                       TASK_WEIGHTS[1] * contrastive['loss'].item() + \
                       TASK_WEIGHTS[2] * tagging['loss'].item()

                self.f1.update(prediction['logit'], task_data['subtree_labels'])
                self.recall.update(prediction['logit'], task_data['subtree_labels'])
                self.precision.update(prediction['logit'], task_data['subtree_labels'])

                if TASK_WEIGHTS[1] != 0.0:
                    contrastive_f1 = MulticlassF1Score(num_classes=contrastive['logit'].shape[1], average=None).to(self.device)
                    contrastive_f1_value += contrastive_f1(contrastive['logit'], torch.zeros(self.batch_size, device=device))[0]

            overall_eval_loss_all += loss
            task_eval_loss_all[0] += prediction['loss'].item()
            task_eval_loss_all[1] += contrastive['loss'].item()
            task_eval_loss_all[2] += tagging['loss'].item()

            for i in range(0, self.num_labels):
                tagging_recall[i] += tagging['recall'][i]

            steps += 1

        f1_score = self.f1.compute()
        precision_score = self.precision.compute()
        recall_score = self.recall.compute()
        overall_metrics = TASK_WEIGHTS[0] * 2 * f1_score + TASK_WEIGHTS[2] * tagging_recall[1] / steps + TASK_WEIGHTS[2] * tagging_recall[2] / steps

        self.logger.info(f"Evaluation finished. Overall eval loss = {overall_eval_loss_all / steps},")
        self.logger.info(
            f"  subtree prediction eval loss = {task_eval_loss_all[0] / steps} , f1 = {f1_score}, precision = {precision_score}, recall = {recall_score}")
        self.logger.info(f"  contrastive learning eval loss = {task_eval_loss_all[1] / steps}, f1 = {contrastive_f1_value / steps}")
        self.logger.info(
            f"  sequence tagging eval loss = {task_eval_loss_all[2] / steps}, table recall = {tagging_recall[1] / steps}, column recall = {tagging_recall[2] / steps}")
        self.logger.info(f"  overall metrics = {overall_metrics}")

        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

        return -overall_metrics

    def train(self):
        self.logger.info("————————————————————————————————————————————————")
        self.logger.info(f"The param size of encoder = {self.model.get_encoder_param_size()}, "
                         f"The param size of model = {sum(p.numel() for p in self.model.parameters())}")
        best_eval_loss = 1000
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            train_batch_iterator = ThreadedIterator(
                self.train_data_loader.make_minibatch_iterator(self.training_buckets),
                max_queue_size=2 * self.batch_size)
            task_train_loss_all = [0.0, 0.0, 0.0]
            overall_train_loss_all = 0.0
            tagging_recall = [0.0 for _ in range(self.num_labels)]
            steps = 0

            for train_step, train_batch_data in enumerate(train_batch_iterator):
                self.model.train()
                self.optimizer.zero_grad()

                if train_step % 11 == 0:
                    if epoch <= 20:
                        TASK_WEIGHTS = [1.0, 0.25, 0.0]
                    else:
                        TASK_WEIGHTS = [1.0, 0.25, 0.1]
                else:
                    TASK_WEIGHTS = [1.0, 0.0, 0.0]

                node_type, node_tokens, children_index = self.load_batch_tree_data(train_batch_data)
                task_data = self.load_batch_task_data(train_batch_data, TASK_WEIGHTS)
                if TASK_WEIGHTS[1] != 0:
                    node_type, node_tokens, children_index, task_data = self.augment_data(node_type, node_tokens, children_index, task_data)
                prediction, contrastive, tagging = self.model(node_type, node_tokens, children_index, task_data)

                loss = TASK_WEIGHTS[0] * prediction['loss'] + TASK_WEIGHTS[1] * contrastive['loss'] + TASK_WEIGHTS[2] * \
                       tagging['loss']
                loss.backward()
                self.optimizer.step()

                overall_train_loss_all += loss.item()
                task_train_loss_all[0] += prediction['loss'].item()
                task_train_loss_all[1] += contrastive['loss'].item()
                task_train_loss_all[2] += tagging['loss'].item()

                for i in range(0, self.num_labels):
                    tagging_recall[i] += tagging['recall'][i]
                steps += 1

                if train_step % train_config['log_step_interval'] == 0:
                    self.logger.info(
                        f"Training at epoch {epoch} and step {train_step} with overall train loss {loss.item()}, "
                        f"subtree prediction train loss = {prediction['loss'].item()}, "
                        f"contrastive learning train loss = {contrastive['loss'].item()}, "
                        f"sequence tagging train loss = {tagging['loss'].item()}, "
                        f"table recall = {tagging['recall'][1]}, column recall = {tagging['recall'][2]}")

            tagging_recall = [recall / steps for recall in tagging_recall]
            self.logger.info("*************************")
            self.logger.info(f"Epoch {epoch}, overall train loss = {overall_train_loss_all / steps}")
            self.logger.info(f"  subtree prediction train loss = {task_train_loss_all[0] / steps}")
            self.logger.info(f"  contrastive learning train loss = {task_train_loss_all[1] / steps} ")
            self.logger.info(
                f"  sequence tagging train loss = {task_train_loss_all[2] / steps}, table recall = {tagging_recall[1]}, column recall = {tagging_recall[2]}")
            self.lr_scheduler.step()
            eval_loss = self.eval()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model(epoch)
                self.logger.info(f"Best Model update at epoch {epoch} with best loss {best_eval_loss}")
            self.logger.info("*************************")

        self.logger.info("————————————————————————————————————————————————")
