class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.final_vocab_size = vocab_size
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.special_tokens.append("<|UNK|>")
        self.special_tokens.append("<|SOS|>")
        self.special_tokens.append("<|EOS|>")
        self.special_tokens.append("<|PAD|>")


        self.special_token_number = len(self.special_tokens)

        self.vocab = set(special_tokens)
        self.word_freq = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

        for i in range(len(self.special_tokens)):
            self.token_to_id[self.special_tokens[i]] = i
            self.id_to_token[i] = self.special_tokens[i]

    def train(self, corpus):
        raise NotImplementedError("Training method not implemented yet.")
    
    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
    
    def get_unk_id(self):
        return self.token_to_id["<|UNK|>"]

    def do_iteration(self):
        pass
    
    def build_token_ids(self):
        index = len(self.token_to_id)
        for token in sorted(self.vocab):
            if token not in self.token_to_id:
                self.token_to_id[token] = index
                self.id_to_token[index] = token
                index += 1
        

    

    def build_init_vocab(self, corpus):
        for sentence in corpus:
            for word in sentence.split():
                chars = tuple(list(word) + ['<WORDEND>'])
                self.word_freq[chars] = self.word_freq.get(chars, 0) + 1
                self.vocab.update(chars)

    


    


    