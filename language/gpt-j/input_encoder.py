from tokenizer_GPTJ import get_transformer_autotokenizer


class input_encoder():
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = get_transformer_autotokenizer(self.model_name)
    
    # encodes input from the network based on the selected tokenizer
    # TODO: code has to be modified when more tokenizers can be added, to induce modularity. Tokenizer name also could be passed, 
    #       and corresponding tokenizer could be selected from tokenizer_GPT.py
    def encode_input_from_network(self, input):
        source_encoded = self.tokenizer(input, return_tensors="pt",
                                            padding=True, truncation=True,
                                            max_length=1919)
        source_encoded_input_id = source_encoded.input_ids
        source_encoded_attn_mask = source_encoded.attention_mask
        return source_encoded_input_id, source_encoded_attn_mask
    
def get_input_encoder(model_name: str):
    return input_encoder(model_name)