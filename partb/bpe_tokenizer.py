import heapq
import json


class BPETokenizer:
    def __init__(self, vocab_size = 10000, special_tokens=None):
        self.final_vocab_size = vocab_size
        self.special_tokens = list(special_tokens) if special_tokens else []
        for tok in (["<|UNK|>", "<|SOS|>", "<|EOS|>", "<|PAD|>", "<|WORDEND|>"]):
            if tok not in self.special_tokens:
                self.special_tokens.append(tok)

        self.special_token_number = len(self.special_tokens)

        self.vocab = set(self.special_tokens)
        self.word_freq = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

        for i in range(len(self.special_tokens)):
            self.token_to_id[self.special_tokens[i]] = i
            self.id_to_token[i] = self.special_tokens[i]

        self.pair_heap = []
        self.pair_freq = {}

    def train(self, corpus):
        self.build_init_vocab(corpus)

        while len(self.vocab) < self.final_vocab_size:
            self.do_iteration()
        self.build_token_ids()
    
    def encode(self, text):
        tokenized_text = self.convert_string_to_tokens(text)
        #now we need to perform merges in order on the tokenized text
        for merge in self.merges:
            new_token = "".join(merge)
            new_tokens = []
            i = 0
            while i < len(tokenized_text):
                if i < len(tokenized_text) - 1 and (tokenized_text[i], tokenized_text[i + 1]) == merge:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokenized_text[i])
                    i += 1
            tokenized_text = new_tokens
                
        token_ids = [self.token_to_id.get(token, self.get_unk_id()) for token in tokenized_text]
        return token_ids

    def decode(self, token_ids):
        str_list = [self.token_id_to_string(token_id) for token_id in token_ids]
        return "".join(str_list)

        
    def token_id_to_string(self, token_id):
        if token_id < self.special_token_number:
            if token_id == self.token_to_id["<|SOS|>"]:
                return ""
            elif token_id == self.token_to_id["<|EOS|>"]:
                return ""
            elif token_id == self.token_to_id["<|PAD|>"]:
                return " "
            elif token_id == self.token_to_id["<|UNK|>"]:
                return "<|UNK|>"
            elif token_id == self.token_to_id["<|WORDEND|>"]:
                return " "
            return ""
        return self.id_to_token.get(token_id, "<|UNK|>")

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "special_token_number": self.special_token_number,
                "vocab": list(self.vocab),
                "word_freq": self.word_freq,
                "merges": self.merges,
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "pair_freq": self.pair_freq,
                "pair_heap": self.pair_heap
            }, f)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.special_token_number = data["special_token_number"]
            self.vocab = set(data["vocab"])
            self.word_freq = {tuple(k): v for k, v in data["word_freq"].items()}
            self.merges = [tuple(merge) for merge in data["merges"]]
            self.token_to_id = {k: v for k, v in data["token_to_id"].items()}
            self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
            self.pair_freq = {tuple(k): v for k, v in data["pair_freq"].items()}
            self.pair_heap = [(-freq, tuple(pair)) for pair, freq in data["pair_heap"]]
            heapq.heapify(self.pair_heap)
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_unk_id(self):
        return self.token_to_id["<|UNK|>"]

    def do_iteration(self):
        while self.pair_heap:
            freq, pair = heapq.heappop(self.pair_heap)
            if self.pair_freq.get(pair, 0) != -freq: #defensive check for stale pairs
                continue
            best_pair = pair
            break
        else:
            return #no merge
        
        new_token = "".join(best_pair)
        self.merges.append(best_pair)
        self.vocab.add(new_token)
        self.pair_freq[best_pair] = 0
        heapq.heappush(self.pair_heap, (0, best_pair))

        for word, freq in list(self.word_freq.items()):
            #in this loop we update the word and pair frequencies and also the heap
            if best_pair in zip(word, word[1:]):
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        if (i>=1):
                            prev_pair = (word[i-1], word[i])
                            self.pair_freq[prev_pair] = self.pair_freq.get(prev_pair, 0) - freq
                            heapq.heappush(self.pair_heap, (-self.pair_freq[prev_pair], prev_pair))

                            new_pair = (word[i-1], new_token)
                            self.pair_freq[new_pair] = self.pair_freq.get(new_pair, 0) + freq
                            heapq.heappush(self.pair_heap, (-self.pair_freq[new_pair], new_pair))

                        if (i < len(word) - 2):
                            next_pair = (word[i + 1], word[i + 2])
                            self.pair_freq[next_pair] = self.pair_freq.get(next_pair, 0) - freq
                            heapq.heappush(self.pair_heap, (-self.pair_freq[next_pair], next_pair))

                            new_pair = (new_token, word[i + 2])
                            self.pair_freq[new_pair] = self.pair_freq.get(new_pair, 0) + freq
                            heapq.heappush(self.pair_heap, (-self.pair_freq[new_pair], new_pair))

                        i += 2

                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_tuple = tuple(new_word)
                self.word_freq[new_word_tuple] = self.word_freq.pop(word)


        
    
    def build_token_ids(self):
        index = len(self.token_to_id)
        for token in sorted(self.vocab):
            if token not in self.token_to_id:
                self.token_to_id[token] = index
                self.id_to_token[index] = token
                index += 1
        

    def build_pair_freq(self):
        self.pair_freq = {}
        for word, freq in self.word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_freq[pair] = self.pair_freq.get(pair, 0) + freq
        self.pair_heap = [(-freq, pair) for pair, freq in self.pair_freq.items()]
        heapq.heapify(self.pair_heap)

    def build_init_vocab(self, corpus):
        for sentence in corpus:
            for word in sentence.split():
                chars = tuple(list(word) + ['<|WORDEND|>'])
                self.word_freq[chars] = self.word_freq.get(chars, 0) + 1
                self.vocab.update(chars)
        
        self.build_pair_freq()

    
    def convert_string_to_tokens(self, str):
        tokens = []
        for word in str.split():
            chars = list(word) + ['<|WORDEND|>']
            tokens.extend(chars)
        return tokens

    


    