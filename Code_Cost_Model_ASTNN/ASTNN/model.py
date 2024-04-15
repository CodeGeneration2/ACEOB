import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

# ######################################################################################################################
class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # Pretrained embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        node_length = len(node)
        if not node_length:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(node_length, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(node_length):
            if isinstance(node[i], list) and node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if isinstance(temp[j], list) and temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(node_length, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]

# ######################################################################################################################
class BatchProgramClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encoder_hidden_dim, output_classes,
                 batch_size, use_gpu=True, pretrained_weight=None, task="Time Prediction"):
        super(BatchProgramClassifier, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.output_classes = output_classes
        self.task = task

        self.encoder = BatchTreeEncoder(self.vocab_size,
                                        self.embedding_dim,
                                        self.encoder_hidden_dim,
                                        self.batch_size,
                                        self.gpu,
                                        pretrained_weight
                                        )

        self.root2label = nn.Linear(self.encoder_hidden_dim, self.output_classes)

        # GRU
        self.BiGRU = nn.GRU(self.encoder_hidden_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True
                            )
        # Linear layer
        if task == "Time Prediction":
            self.hidden2label = nn.Linear(self.hidden_dim * 2, 1)
        elif task == "Source Code Classification":
            self.hidden2label = nn.Linear(self.hidden_dim * 2, self.output_classes)

        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    # ===================================================================================================
    # Initialize hidden state
    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.BiGRU, nn.LSTM):
                # Initialize long-term state h0 to all zeros and move it to GPU, considering bidirectional LSTM, each layer has both forward and backward hidden states.
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                # Similar to h0, initialize short-term state c0 to all zeros
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encoder_hidden_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    # ===================================================================================================
    def forward(self, x):
        batch_lengths = [len(item) for item in x]
        max_batch_length = max(batch_lengths)

        encodes = []
        for i in range(self.batch_size):
            for j in range(batch_lengths[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(batch_lengths))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += batch_lengths[i]
            seq.append(encodes[start:end])
            if max_batch_length - batch_lengths[i]:
                seq.append(self.get_zeros(max_batch_length - batch_lengths[i]))
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_batch_length, -1)
        encodes = nn.utils.rnn.pack_padded_sequence(encodes, torch.LongTensor(batch_lengths), batch_first=True, enforce_sorted=False)

        # GRU
        gru_out, _ = self.BiGRU(encodes, self.hidden)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=-1e9)

        gru_out = torch.transpose(gru_out, 1, 2)

        # Pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        # Linear
        y = self.hidden2label(gru_out)

        return y


