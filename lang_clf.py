import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification

    
class Net(torch.nn.Module):
    def __init__(self, nb_label=2):
        super().__init__()
        self.device = "cpu"
        encoder_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=nb_label, output_attentions=True, output_hidden_states=True)
        encoder_model = list(encoder_model.children())[0] 
        self.encoder = encoder_model.to(self.device)
        self.encoder_size = list(self.encoder.parameters())[-1].shape[0]
        self.nb_label = nb_label
        self.state_attention = torch.nn.Linear(768, 1)
        self.layer_attention = torch.nn.Linear(768, 1)

        self.topic_head = torch.nn.Sequential(
            torch.nn.Dropout(0.4), 
            torch.nn.Linear(768, nb_label),
        ).to(self.device)
        

    def forward(self, input_ids, segment_ids, input_mask):
        model_output, hidden_state_7layers, attention = self.encoder(input_ids) 
        hidden_states = torch.stack([hidden_state_7layers[i] for i in range(7)], dim=1)
        state_weights = self.state_attention(hidden_states)
        normalized_state_weights = nn.functional.softmax(state_weights, dim = -2) 
        state_weighted_output = (hidden_states.transpose(-2, -1) @ normalized_state_weights).squeeze()
        layer_weights = self.state_attention(state_weighted_output)
        normalized_layer_weights = nn.functional.softmax(layer_weights, dim = -2) 
        weightedsum_output = (state_weighted_output.transpose(-2, -1) @ normalized_layer_weights).squeeze()
        outputs = self.topic_head(weightedsum_output)#.squeeze(-1))
        return outputs

    #finalModel = model_for_topic(model, nb_label=params.nb_classes, type_encoder = params.type_encode, smoothing = 0.03).to(params.device)
