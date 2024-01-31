import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from sql_encoder.model.tbcnn_encoder import TBCNNEncoder
from sql_encoder.training.learning_tasks import SequenceTaggingModel
from sql_encoder.training.learning_tasks import SubtreePredictionModel
from torchmetrics.classification import MulticlassRecall
from config.encoder_config import *


class PretrainSQLEncoder(nn.Module):

    def __init__(self, encoder_config, subtree_prediction_config, contrastive_config, sequence_tagging_config):
        super(PretrainSQLEncoder, self).__init__()

        self.encoder = TBCNNEncoder(encoder_config)

        self.subtree_predictor = SubtreePredictionModel(subtree_prediction_config)
        self.prediction_criterion = nn.BCEWithLogitsLoss()

        self.contrastive_config = contrastive_config

        self.sequence_tagger = SequenceTaggingModel(sequence_tagging_config)
        self.num_labels = sequence_tagging_config['num_labels']
        self.tagging_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.tagging_recall = MulticlassRecall(average=None, num_classes=self.num_labels, ignore_index=-1).to(device)

    def compute_contrastive_loss(self, code_vector, node_type, temp=0.05):
        num_neg_random = self.contrastive_config['num_neg_random']
        num_pos = self.contrastive_config['num_pos']
        bs = code_vector.shape[0] // (1 + num_pos + num_neg_random)
        device = code_vector.device

        # Slice code_vector into separate parts
        code_vector_raw, code_vector_pos, code_vector_neg_ran = torch.split(code_vector,
                                                                            [bs, bs, bs * num_neg_random])

        # Reshape the tensors
        code_vector_pos = code_vector_pos.unsqueeze(1)
        code_vector_neg_ran = code_vector_neg_ran.view(num_neg_random, bs, -1).transpose(0, 1)

        # node_type_raw: (bs, max_tree_size), node_type_neg_ran: (bs * num_neg_random, max_tree_size)
        node_type_raw, _, node_type_neg_ran = torch.split(node_type, [bs, bs, bs * num_neg_random])
        node_type_neg_ran = node_type_neg_ran.view(num_neg_random, bs, -1).transpose(0, 1)
        node_cnt_neg_ran = torch.count_nonzero(node_type_neg_ran, dim=2)
        node_cnt_raw = torch.count_nonzero(node_type_raw, dim=-1).unsqueeze(1).repeat(1, num_neg_random)
        alter_mask = node_cnt_neg_ran != node_cnt_raw
        if not self.training:
            print(alter_mask)

        # Create a range tensor
        range_tensor = torch.arange(bs, device=device)
        mask = range_tensor.unsqueeze(1) != range_tensor.unsqueeze(0)
        neg_indices = range_tensor.unsqueeze(0).repeat(bs, 1)[mask].view(bs, bs - 1)
        code_vector_oth = code_vector_raw[neg_indices]

        # Concatenate negative samples
        code_vector_samples = torch.cat([code_vector_pos, code_vector_neg_ran, code_vector_oth], dim=1)

        # Normalize the vectors
        normalized_code_vector_raw = F.normalize(code_vector_raw, p=2, dim=-1).unsqueeze(1)
        normalized_code_vector_samples = F.normalize(code_vector_samples, p=2, dim=-1).transpose(1, 2)

        # Compute logit scores (similarity)
        # (batch_size, 1, emb_dim) * (batch_size, emb_dim, sample_cnt)
        contrastive_logs = torch.bmm(normalized_code_vector_raw, normalized_code_vector_samples).squeeze(1)
        if not self.training:
            print(contrastive_logs)
        contrastive_logs = contrastive_logs / temp

        labels = torch.zeros(bs, device=device, dtype=torch.long)
        loss = nn.CrossEntropyLoss()
        return loss(contrastive_logs, labels), contrastive_logs

    # separate-pass
    def forward(self, node_type, node_tokens, children_index, task_data):
        # code_vector: batch_size × embedding_dim
        # node_vectors: batch_size × max_tree_size × embedding_dim
        batch_size = train_config['batch_size']
        anchor_node_type = node_type[:batch_size]
        anchor_node_tokens = node_tokens[:batch_size]
        anchor_children_index = children_index[:batch_size]

        code_vector, node_vectors = self.encoder(anchor_node_type, anchor_node_tokens, anchor_children_index)
        results = {}

        prediction_logit = self.subtree_predictor(code_vector)  # batch_size × num_subtrees
        prediction_loss = self.prediction_criterion(prediction_logit, task_data['subtree_labels'][:batch_size])
        results['subtree'] = {"loss": prediction_loss, "logit": prediction_logit}

        node_tag = task_data['node_tag'][:batch_size]
        tag_space = self.sequence_tagger(node_vectors)
        tagging_loss = self.tagging_criterion(tag_space.view(-1, self.num_labels), node_tag.view(-1))
        results['tagging'] = {"loss": tagging_loss}
        if not self.training:
            self.tagging_recall.update(tag_space.view(-1, self.num_labels), node_tag.view(-1))

        sample_node_type = node_type[batch_size:]
        sample_node_tokens = node_tokens[batch_size:]
        sample_children_index = children_index[batch_size:]
        sample_code_vector, _ = self.encoder(sample_node_type, sample_node_tokens, sample_children_index)
        code_vector = torch.cat((code_vector, sample_code_vector), dim=0)
        loss, logit = self.compute_contrastive_loss(code_vector, node_type)
        results['contrastive'] = {"loss": loss, "logit": logit}

        return results

    def get_encoder_param_size(self):
        return sum(p.numel() for p in self.encoder.parameters())

    def get_tagging_recall(self):
        recall = self.tagging_recall.compute()
        self.tagging_recall.reset()
        return recall
