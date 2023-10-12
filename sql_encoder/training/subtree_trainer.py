from sql_encoder.data_utils.threaded_iterator import ThreadedIterator
from sql_encoder.training.pre_training import PretrainSQLEncoder
from config.encoder_config import *
from sql_encoder.training.base_trainer import BaseTrainer
from tokenizers import Tokenizer
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
import torch


class SubtreeTrainer(BaseTrainer):
    def __init__(self):
        super(SubtreeTrainer, self).__init__()

        self.subtree_vocab = Tokenizer.from_file(data_path['subtree_vocab_model_path'])
        subtree_prediction_config['num_subtrees'] = self.subtree_vocab.get_vocab_size()

        self.model = PretrainSQLEncoder(encoder_config, subtree_prediction_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config['lr'],
                                          weight_decay=train_config['weight_decay'])

        self.f1 = MultilabelF1Score(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(self.device)
        self.recall = MultilabelRecall(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(self.device)
        self.precision = MultilabelPrecision(num_labels=self.subtree_vocab.get_vocab_size(), average='weighted').to(self.device)

    def load_batch_task_data(self, batch_data):
        batch_subtree_id = batch_data["batch_subtree_id"]
        # 在CPU上只能使用32位float，不能使用16位
        batch_subtree_labels = torch.zeros((self.batch_size, self.subtree_vocab.get_vocab_size()), dtype=torch.float32, device=self.device)
        for index, _1 in enumerate(batch_subtree_id):
            batch_subtree_labels[index][_1] = 1.0
        return batch_subtree_labels

    def eval(self):
        eval_batch_iterator = ThreadedIterator(self.eval_data_loader.make_mini_batch_iterator(self.evaluation_buckets),
                                               max_queue_size=2 * self.batch_size)
        eval_loss_all = []
        for _, eval_batch_data in enumerate(eval_batch_iterator):
            self.model.eval()
            with torch.no_grad():
                node_type, node_tokens, children_index = self.load_batch_tree_data(eval_batch_data)
                task_data = self.load_batch_task_data(eval_batch_data)
                loss, logit = self.model(node_type, node_tokens, children_index, task_data)
                eval_loss_all.append(loss.item())

                self.f1.update(logit, task_data)
                self.recall.update(logit, task_data)
                self.precision.update(logit, task_data)

        eval_loss = sum(eval_loss_all) / len(eval_loss_all)
        f1_score = self.f1.compute()
        precision_score = self.precision.compute()
        recall_score = self.recall.compute()
        self.logger.info(
            f"Evaluation finished, eval loss {eval_loss}, F1 score {f1_score}, precision {precision_score}, recall {recall_score}")

        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

        return eval_loss, f1_score

    def train(self):
        self.logger.info("————————————————————————————————————————————————")
        self.logger.info(f"The param size of encoder = {self.model.get_encoder_param_size()}, "
                         f"The param size of model = {sum(p.numel() for p in self.model.parameters())}")
        best_f1_score = -1
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            train_batch_iterator = ThreadedIterator(
                self.train_data_loader.make_mini_batch_iterator(self.training_buckets),
                max_queue_size=2 * self.batch_size)
            train_loss_all = 0.0
            steps = 0
            for train_step, train_batch_data in enumerate(train_batch_iterator):
                self.model.train()
                self.optimizer.zero_grad()

                node_type, node_tokens, children_index = self.load_batch_tree_data(train_batch_data)
                task_data = self.load_batch_task_data(train_batch_data)
                loss, _ = self.model(node_type, node_tokens, children_index, task_data)
                loss.backward()
                self.optimizer.step()
                train_loss_all += loss.item()
                steps += 1

                if train_step % train_config['log_step_interval'] == 0:
                    self.logger.info(
                        f"Training at epoch {epoch} and step {train_step} with subtree prediction train loss {loss.item()}")

            self.logger.info("*************************")
            self.logger.info(f"Checkpoint saved at epoch {epoch}, overall train loss = {train_loss_all / steps}")
            eval_loss, f1_score = self.eval()
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                self.save_model(epoch)
                self.logger.info(f"Best Model update at epoch {epoch} with best F1 score {best_f1_score}")
            self.logger.info("*************************")

        self.logger.info("————————————————————————————————————————————————")
