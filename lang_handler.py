import io
import logging
import numpy as np
import os
import json
import torch
from PIL import Image
from torch.autograd import Variable
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

logger = logging.getLogger(__name__)


class ELDALangClassifier(object):
    """
    """
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = "cpu"
        self.initialized = False

        
    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "lang_model.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "lang_clf.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        from lang_clf import Net
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = Net(nb_label=3)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True

        
    def preprocess(self, jsondata):
        texts = jsondata[0]['body']
        print("!!!!!!!!!!!!!!!!!!!!!!!!INPUT", texts)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        outputs = []
        texts_ids = []
        segments_ids = []
        inputs_mask = []

        for text in texts: 
            text_tokenized = tokenizer.tokenize(text)
            text_ids = tokenizer.convert_tokens_to_ids([tokenizer.special_tokens_map["cls_token"]]+text_tokenized+[tokenizer.special_tokens_map["sep_token"]])
            texts_ids.append(text_ids)
            segments_ids.append([0] * len(text_ids))
            inputs_mask.append([1] * len(text_ids))

        lengths = [len(p) for p in texts_ids]
        max_length = max(lengths)
        padded_text_ids = np.ones((len(texts_ids), max_length))*tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map["pad_token"])

        padded_segments_ids = np.zeros((len(texts_ids), max_length))
        padded_inputs_mask = np.zeros((len(texts_ids), max_length))

        for i, l in enumerate(lengths):
            padded_text_ids[i, 0:l] = texts_ids[i][0:l]
            padded_segments_ids[i, 0:l] = segments_ids[i][0:l]
            padded_inputs_mask[i, 0:l] = inputs_mask[i][0:l]

        outputs = np.array([padded_text_ids, padded_segments_ids, padded_inputs_mask]).transpose(1,0,2)
        outputs = torch.LongTensor(outputs)
        return outputs[:,0,:], outputs[:,1,:], outputs[:,2,:]


    def inference(self, padded_tensors):
        ''' Input padded_tensors output by preprocess()
        '''
        self.model.eval()
        text_ids, segments_ids, inputs_mask = padded_tensors
        outputs = self.model(text_ids, segments_ids, inputs_mask)
        predictions = np.argmax(outputs.detach().numpy(), axis=-1)
        return predictions

    
    def postprocess(self, predictions):
        classes = {0: 'FR', 1: 'EN', 2: 'MA'}
        if type(predictions) == list:
            model_out = [classes[pred] for pred in predictions]
        else:
            model_out = classes[predictions]
        # model_out = json.dumps({'response': model_out})
        return model_out


    def handle(self, data, context):
        model_input = self.preprocess(data)
        predictions = self.inference(model_input)
        model_out = self.postprocess(predictions)
        print("!!!!!!!!!!!!!!!!!!!!!!!!OUTPUT", model_out)
        # '||'.join(model_out)
        # return model_out
        return [model_out]

        
_service = ELDALangClassifier()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    if data is None:
        return None
    
    return _service.handle(data, context)
