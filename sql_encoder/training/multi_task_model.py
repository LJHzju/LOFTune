import torch.nn as nn
import torch.nn.functional as F
from sql_encoder.model.tbcnn_encoder import TBCNNEncoder
from sql_encoder.training.learning_tasks import SubtreePredictionModel, SequenceTaggingModel
from torchmetrics.classification import MulticlassRecall
from config.encoder_config import *


def compute_contrastive_loss(code_vector, temp=0.07):
    neg_num_oth = 15
    bs = train_config['batch_size']

    code_vector_raw = code_vector[:bs]
    code_vector_pos = code_vector[bs: 2 * bs]
    code_vector_neg_ran = code_vector[2 * bs:]
    code_vector_pos = code_vector_pos.view(bs, 1, -1)
    code_vector_neg_ran = code_vector_neg_ran.view(-1, bs, sql_embedding_dim).transpose(0, 1)

    # Create a tensor of probabilities with ones, we will use this to randomly sample indices
    probabilities = torch.ones((bs, bs), device=device)
    # Fill the diagonal with zeros to avoid sampling the same element
    probabilities.fill_diagonal_(0)
    # Sample the indices
    neg_indices = torch.multinomial(probabilities, neg_num_oth)
    code_vector_oth = code_vector_raw[neg_indices].view(bs, neg_num_oth, -1)

    code_vector_neg = torch.cat([code_vector_neg_ran, code_vector_oth], dim=1)

    normalized_code_vector_raw = F.normalize(code_vector_raw, p=2, dim=-1)
    normalized_code_vector_pos = F.normalize(code_vector_pos, p=2, dim=-1)
    normalized_code_vector_neg = F.normalize(code_vector_neg, p=2, dim=-1)

    log_pos = torch.bmm(normalized_code_vector_raw.unsqueeze(1), normalized_code_vector_pos.transpose(1, 2)).squeeze(1)
    log_neg = torch.bmm(normalized_code_vector_raw.unsqueeze(1), normalized_code_vector_neg.transpose(1, 2)).squeeze(1)
    log_pos, log_neg = log_pos / temp, log_neg / temp

    contrastive_logs = torch.cat([log_pos, log_neg], dim=-1)
    torch.set_printoptions(profile="full")

    labels = torch.zeros(bs, device=device, dtype=torch.long)
    loss = nn.CrossEntropyLoss()
    return loss(contrastive_logs, labels), contrastive_logs


class PretrainSQLEncoder(nn.Module):

    def __init__(self, encoder_config, subtree_prediction_config, sequence_tagging_config):
        super(PretrainSQLEncoder, self).__init__()

        self.encoder = TBCNNEncoder(encoder_config)
        
        self.subtree_predictor = SubtreePredictionModel(subtree_prediction_config)
        self.prediction_criterion = nn.BCEWithLogitsLoss()     
 
        self.sequence_tagger = SequenceTaggingModel(sequence_tagging_config)
        self.num_labels = sequence_tagging_config['num_labels']
        self.tagging_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.tagging_recall = MulticlassRecall(average=None, num_classes=self.num_labels, ignore_index=-1)

    def forward(self, node_type, node_tokens, children_index, task_data):
        # code_vector: batch_size × embedding_dim
        # node_vectors: batch_size × max_tree_size × embedding_dim
        # mask_nodes: batch_size × max_tree_size
        code_vector, node_vectors = self.encoder(node_type, node_tokens, children_index)
        mask_nodes = (node_type != 0)

        contrastive_result = {"loss": torch.tensor([0.0], requires_grad=True, device=device)}
        if 'contrastive' in task_data.keys():
            loss, logit = compute_contrastive_loss(code_vector)
            contrastive_result = {"loss": loss, "logit": logit}

        prediction_result = {"loss": torch.tensor([0.0], requires_grad=True, device=device)}
        if 'subtree' in task_data.keys():
            prediction_logit = self.subtree_predictor(code_vector)  # batch_size × num_subtrees
            prediction_loss = self.prediction_criterion(prediction_logit, task_data['subtree_labels'])
            prediction_result = {"loss": prediction_loss, "logit": prediction_logit}

        tagging_result = {"loss": torch.tensor([0.0], requires_grad=True, device=device), "recall": [0, 0, 0, 0]}
        if 'tagging' in task_data.keys():
            node_tag = task_data['node_tag']
            tag_space = self.sequence_tagger(node_vectors, code_vector, mask_nodes)
            tagging_loss = self.tagging_criterion(tag_space.view(-1, self.num_labels), node_tag.view(-1).long())
            _, pred_labels = torch.max(F.log_softmax(tag_space, dim=2), 2)
            pred_labels = torch.masked_select(pred_labels, mask_nodes)
            true_labels = torch.masked_select(node_tag, mask_nodes)
            tagging_recall = self.tagging_recall(pred_labels, true_labels)
            tagging_result = {"loss": tagging_loss, "recall": tagging_recall}

        return prediction_result, contrastive_result, tagging_result

    def get_encoder_param_size(self):
        return sum(p.numel() for p in self.encoder.parameters())

