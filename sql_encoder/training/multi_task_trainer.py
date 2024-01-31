from sql_encoder.data_utils.threaded_iterator import ThreadedIterator
from sql_encoder.training.multi_task_model import PretrainSQLEncoder
from sql_encoder.training.base_trainer import BaseTrainer
from config.encoder_config import *
from tokenizers import Tokenizer
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MulticlassF1Score


def gen_loss(task_weights, results):
    loss = task_weights[0] * results['subtree']['loss'] + \
           task_weights[1] * results['contrastive']['loss'] + \
           task_weights[2] * results['tagging']['loss']
    subtree_loss = results['subtree']['loss'].item() if task_weights[0] != 0 else 0
    contrastive_loss = results['contrastive']['loss'].item() if task_weights[1] != 0 else 0
    tagging_loss = results['tagging']['loss'].item() if task_weights[2] != 0 else 0
    return loss, subtree_loss, contrastive_loss, tagging_loss


class MultiTaskTrainer(BaseTrainer):
    def __init__(self):
        super(MultiTaskTrainer, self).__init__()

        self.subtree_vocab = Tokenizer.from_file(data_path['subtree_vocab_model_path'])
        self.num_subtrees = self.subtree_vocab.get_vocab_size()
        self.num_contrastive_pos = contrastive_config['num_pos']
        self.num_contrastive_neg = contrastive_config['num_neg_random']
        subtree_prediction_config['num_objects'] = self.num_subtrees

        self.model = PretrainSQLEncoder(encoder_config,
                                        subtree_prediction_config,
                                        contrastive_config,
                                        sequence_tagging_config).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(),
             'lr': train_config['lr'],
             'weight_decay': train_config['weight_decay']},
        ])
        lr_steps = lambda epoch: 1 if epoch < 5 else (0.2 if epoch < 15 else 0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_steps, verbose=True)

    def load_batch_task_data(self, batch_data, children_index):
        batch_task_data = {}

        num_samples = self.batch_size * (1 + self.num_contrastive_pos + self.num_contrastive_neg)
        batch_subtree_labels = torch.zeros((num_samples, self.subtree_vocab.get_vocab_size()),
                                           dtype=torch.float16, device=self.device)
        batch_subtree_id = torch.from_numpy(batch_data["batch_subtree_id"]).to(self.device)
        valid_indices_mask = batch_subtree_id != -1
        flat_subtree_ids = batch_subtree_id[valid_indices_mask]
        batch_indices = torch.arange(num_samples, device=self.device).unsqueeze(1).expand(-1, batch_subtree_id.shape[1]).flatten()
        batch_indices = batch_indices[valid_indices_mask.flatten()]
        batch_subtree_labels.index_put_((batch_indices, flat_subtree_ids),
                                        torch.tensor(1.0, device=self.device, dtype=torch.float16))
        batch_task_data['subtree_labels'] = batch_subtree_labels

        padded_node_tag = torch.from_numpy(batch_data["batch_node_tag"]).to(device)
        non_leaf_node_mask = torch.any(children_index != 0, dim=2)
        padded_node_tag[non_leaf_node_mask] = -1
        batch_task_data['node_tag'] = padded_node_tag

        return batch_task_data

    def eval(self):
        eval_batch_iterator = ThreadedIterator(self.eval_data_loader.make_minibatch_iterator(self.evaluation_buckets,
                                                                                             self.num_contrastive_pos,
                                                                                             self.num_contrastive_neg),
                                               max_queue_size=2 * self.batch_size)
        overall_eval_loss_all = 0.0
        task_eval_loss_all = [0.0, 0.0, 0.0]
        task_weights = [1.0, 1.0, 1.0]

        subtree_f1 = MultilabelF1Score(num_labels=self.num_subtrees, average='weighted').to(self.device)
        subtree_recall = MultilabelRecall(num_labels=self.num_subtrees, average='weighted').to(self.device)
        subtree_precision = MultilabelPrecision(num_labels=self.num_subtrees, average='weighted').to(self.device)

        contrastive_f1 = MulticlassF1Score(num_classes=sum(contrastive_config.values()), average=None).to(self.device)

        steps = 0
        for _, eval_batch_data in enumerate(eval_batch_iterator):
            self.model.eval()
            with torch.no_grad():
                node_type, node_tokens, children_index = self.load_batch_tree_data(eval_batch_data)
                task_data = self.load_batch_task_data(eval_batch_data, children_index)

                selected_node_tokens = node_tokens[2 * self.batch_size:]
                selected_node_types = node_type[2 * self.batch_size:].unsqueeze(-1)
                neg_mask = (selected_node_types == 367) & (selected_node_tokens != 0)
                selected_node_tokens[neg_mask] = torch.randint(8, 9999, selected_node_tokens[neg_mask].shape,
                                                               device=device, dtype=selected_node_tokens.dtype)
                node_tokens[2 * self.batch_size:] = selected_node_tokens

                results = self.model(node_type, node_tokens, children_index, task_data)
                loss, subtree_loss, contrastive_loss, tagging_loss = gen_loss(task_weights, results)

                subtree_labels = task_data['subtree_labels'][:self.batch_size]
                subtree_f1.update(results['subtree']['logit'], subtree_labels)
                subtree_recall.update(results['subtree']['logit'], subtree_labels)
                subtree_precision.update(results['subtree']['logit'], subtree_labels)

                contrastive_f1.update(results['contrastive']['logit'], torch.zeros(self.batch_size, device=device))

            overall_eval_loss_all += loss
            task_eval_loss_all[0] += subtree_loss
            task_eval_loss_all[1] += contrastive_loss
            task_eval_loss_all[2] += tagging_loss

            steps += 1

        subtree_f1_score = subtree_f1.compute()
        subtree_precision_score = subtree_precision.compute()
        subtree_recall_score = subtree_recall.compute()
        contrastive_f1_score = contrastive_f1.compute()[0]
        tagging_recall = self.model.get_tagging_recall()

        mean_recall = torch.mean(tagging_recall)
        overall_metrics = subtree_f1_score + mean_recall

        self.logger.info(f"Evaluation finished. Overall eval loss = {overall_eval_loss_all / steps},")
        self.logger.info(
            f"  subtree prediction eval loss = {task_eval_loss_all[0] / steps} , f1 = {subtree_f1_score}, precision = {subtree_precision_score}, recall = {subtree_recall_score}")
        self.logger.info(
            f"  contrastive learning eval loss = {task_eval_loss_all[1] / steps}, f1 = {contrastive_f1_score}")
        self.logger.info(
            f"  sequence tagging eval loss = {task_eval_loss_all[2] / steps}, recalls = {tagging_recall} with mean {mean_recall}")
        self.logger.info(f"  overall metrics = {overall_metrics}")

        return -overall_metrics

    def train(self):
        self.logger.info("————————————————————————————————————————————————")
        self.logger.info(f"The param size of encoder = {self.model.get_encoder_param_size()}, "
                         f"The param size of model = {sum(p.numel() for p in self.model.parameters())}")
        best_eval_loss = 1000
        early_stop = 0
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            train_batch_iterator = ThreadedIterator(
                self.train_data_loader.make_minibatch_iterator(self.training_buckets,
                                                               self.num_contrastive_pos,
                                                               self.num_contrastive_neg),
                max_queue_size=2 * self.batch_size)
            task_train_loss_all = [0.0, 0.0, 0.0]
            overall_train_loss_all = 0.0
            steps = 0

            if epoch <= 5:
                TASK_WEIGHTS = [1.0, 0.0, 0.0]
            elif epoch <= 15:
                TASK_WEIGHTS = [1.0, 0.5, 0.0]
            else:
                TASK_WEIGHTS = [1.0, 0.5, 0.1]

            for train_step, train_batch_data in enumerate(train_batch_iterator):
                self.model.train()
                self.optimizer.zero_grad()

                node_type, node_tokens, children_index = self.load_batch_tree_data(train_batch_data)
                task_data = self.load_batch_task_data(train_batch_data, children_index)

                selected_node_tokens = node_tokens[2 * self.batch_size:]
                selected_node_types = node_type[2 * self.batch_size:].unsqueeze(-1)
                neg_mask = (selected_node_types == 367) & (selected_node_tokens != 0)
                selected_node_tokens[neg_mask] = torch.randint(8, 9999, selected_node_tokens[neg_mask].shape,
                                                               device=device, dtype=selected_node_tokens.dtype)
                node_tokens[2 * self.batch_size:] = selected_node_tokens

                results = self.model(node_type, node_tokens, children_index, task_data)
                loss, subtree_loss, contrastive_loss, tagging_loss = gen_loss(TASK_WEIGHTS, results)

                loss.backward()
                self.optimizer.step()

                overall_train_loss_all += loss.item()
                task_train_loss_all[0] += subtree_loss
                task_train_loss_all[1] += contrastive_loss
                task_train_loss_all[2] += tagging_loss

                steps += 1

                if train_step % train_config['log_step_interval'] == 0:
                    self.logger.info(
                        f"Training at epoch {epoch} and step {train_step} with overall train loss {loss.item()}, "
                        f"subtree prediction train loss = {subtree_loss}, "
                        f"contrastive learning train loss = {contrastive_loss}, "
                        f"sequence tagging train loss = {tagging_loss}")

            self.logger.info("*************************")
            self.logger.info(f"Epoch {epoch}, overall train loss = {overall_train_loss_all / steps}")
            self.logger.info(f"  subtree prediction train loss = {task_train_loss_all[0] / steps}")
            self.logger.info(f"  contrastive learning train loss = {task_train_loss_all[1] / steps} ")
            self.logger.info(f"  sequence tagging train loss = {task_train_loss_all[2] / steps}")
            self.lr_scheduler.step()
            eval_loss = self.eval()

            if eval_loss < best_eval_loss:
                early_stop = 0
                best_eval_loss = eval_loss
                self.save_model(epoch)
                self.logger.info(f"Best Model update at epoch {epoch} with best loss {best_eval_loss}")
            else:
                early_stop += 1
                if early_stop >= 800:
                    self.logger.info("Early Stopping...")
                    break
            self.logger.info("*************************")

        self.logger.info("————————————————————————————————————————————————")
