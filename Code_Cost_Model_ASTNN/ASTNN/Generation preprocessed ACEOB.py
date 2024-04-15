import math

import pandas as pd
import os
from tqdm.auto import tqdm
from config import *
import ast
from tree_Python import ASTNode

from prepare_data_Python import get_block_list
from gensim.models.word2vec import Word2Vec

tqdm.pandas()

# ######################################################################################################################
class GeneratePreprocessedData:
    """Generate Preprocessed Data class

    Arguments:
        data_split_ratio ([type]): The ratio for splitting the dataset.
        data_root_path (str): The folder path containing the data.
    """

    def __init__(self, raw_dataset_path: str, data_root_path: str):
        self.raw_dataset_path = raw_dataset_path
        self.data_root_path = data_root_path
        self.train_df = None
        self.test_df = None
        self.train_code_data_path = os.path.join(data_root_path, "train", "ACEOB-NPI-train.pkl")
        self.train_AST_data_path = os.path.join(data_root_path, "train", "ACEOB-NPI-train-AST.pkl")
        self.train_AST_data_path_batches = os.path.join(data_root_path, "train", "ACEOB-NPI-train-AST")
        self.test_code_data_path = os.path.join(data_root_path, "test", "ACEOB-NPI-test.pkl")
        self.test_AST_data_path = os.path.join(data_root_path, "test", "ACEOB-NPI-test-AST.pkl")
        self.embedding_size = EMBEDDING_SIZE

    # Initialize function (importing data from local)
    def init_function(self):

        if os.path.exists(self.train_code_data_path) and os.path.exists(self.test_code_data_path):
            self.train_df = pd.read_pickle(self.train_code_data_path)
            self.test_df = pd.read_pickle(self.test_code_data_path)
        else:
            for set_name in ["train", "test"]:
                data_dict = {"code": [], "label": []}
                code_files = os.listdir(os.path.join(self.raw_dataset_path, set_name))
                for code_file in tqdm(code_files):
                    with open(os.path.join(self.raw_dataset_path, set_name, code_file), 'r', encoding='UTF-8') as f:
                        code = f.read().strip()
                    runtime = int(code_file.split(",Running time ")[-1].split(" ms,")[0])

                    data_dict["code"].append(code)
                    data_dict["label"].append(runtime)

                code_df = pd.DataFrame(data_dict)
                code_df['id'] = code_df.index
                cols = code_df.columns.tolist()  # Get list of column names
                cols = cols[-1:] + cols[:-1]  # Move the last element (our 'Index' column) to the beginning of the list
                code_df = code_df[cols]  # Reorder DataFrame columns
                set_root_path = os.path.join(self.data_root_path, set_name)

                os.makedirs(set_root_path, exist_ok=True)

                if set_name == "train":
                    code_df.to_pickle(self.train_code_data_path)
                    self.train_df = code_df
                elif set_name == "test":
                    code_df.to_pickle(self.test_code_data_path)
                    self.test_df = code_df

        print(f'\033[0:36m============= Loaded train: {len(self.train_df)} records ===========\033[m')
        print(f'\033[0:36m============= Loaded test : {len(self.test_df)} records ===========\033[m')

    # Parse source code
    def parse_source_code(self) -> pd.DataFrame:
        # Use ast.parse function to parse Python code and get AST
        self.train_df['code'] = self.train_df['code'].apply(ast.parse)
        self.test_df['code'] = self.test_df['code'].apply(ast.parse)

        # Save the modified total data DataFrame object to a pickle file.
        self.train_df.to_pickle(self.train_AST_data_path)
        self.test_df.to_pickle(self.test_AST_data_path)

    # ===================================================================================================
    # Build dictionary and train word embeddings
    def build_dictionary_and_train_embeddings(self):

        train_AST_data = pd.read_pickle(self.train_AST_data_path)

        if not os.path.exists(self.data_root_path + 'train/embedding'):
            os.mkdir(self.data_root_path + 'train/embedding')

        from prepare_data_Python import recursively_get_ast_sequence

        def convert_AST_to_sequence(AST):
            AST_sequence_list = []
            recursively_get_ast_sequence(AST, AST_sequence_list)
            # Use list comprehension to filter out all 'Load', 'Store', and 'alias'
            AST_sequence_list = [item for item in AST_sequence_list if item not in ['Load', 'Store', 'alias']]
            AST_sequence_list.append('End')
            return AST_sequence_list

        AST_sequence = train_AST_data['code'].apply(convert_AST_to_sequence)
        AST_sequence_string_list = [' '.join(str(c)) for c in AST_sequence]
        train_AST_data['code'] = pd.Series(AST_sequence_string_list)
        train_AST_data.to_csv(self.data_root_path + 'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        # sg=1: Use skip-gram model. If set to 0, use CBOW model. Skip-gram and CBOW are two implementations of Word2Vec, skip-gram generally performs better on small datasets but is slower to train.
        # min_count=MIN_COUNT: Ignore all words with total frequency lower than this. This helps remove rare words, reduce model noise, and decrease vocabulary size.
        # max_final_vocab=VOCAB_SIZE: Set the maximum size of the vocabulary after training. If vocabulary exceeds this number, the least frequent words will be removed until this size is reached.
        train_word_vectors = Word2Vec(AST_sequence, vector_size=self.word_vector_size, workers=16, sg=1,
                                      min_count=MIN_COUNT,
                                      max_final_vocab=VOCAB_SIZE)
        train_word_vectors.save(self.data_root_path + 'train/embedding/node_w2v_' + str(self.word_vector_size))

    # ===================================================================================================
    # Generate block sequences with index representation
    def generate_block_sequences(self, data_path, dataset_name):

        # Load Word2Vec model. .wv accesses the model's word vectors.
        word2vec = Word2Vec.load(self.data_root_path + 'train/embedding/node_w2v_' + str(self.word_vector_size)).wv
        # Obtained the model's vocabulary, which is a dictionary where keys are vocabulary words and values are information about these words, such as their frequency in the training data, their index in the word vector matrix, etc.
        vocabulary = word2vec.key_to_index
        # The number of word vectors, which is the total number of words in the model's vocabulary.
        total_words = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token_list = node.token
            filtered_tokens = [item for item in token_list if item not in ['Load', 'Store', 'alias']]
            if len(filtered_tokens) == 0:
                return None
            encoded_list = []
            for token in filtered_tokens:
                encoded_list.extend([vocabulary[token] if token in vocabulary else total_words])
            children = node.children
            for child in children:
                child_sequence = tree_to_index(child)
                if child_sequence:
                    encoded_list.append(child_sequence)
            return encoded_list

        # Convert AST to numerical sequence. Example: (46887, [[32, [2, [30, [40, [81]]]]], [7],
        def convert_AST_to_numerical_sequence(AST_node):
            block_list = []
            get_block_list(AST_node, block_list)
            if not block_list:
                block_list.append(ASTNode(AST_node))
            tree = []
            for block in block_list:
                btree = tree_to_index(block)
                if btree:
                    tree.append(btree)
            return tree

        AST_table = pd.read_pickle(self.train_AST_data_path)
        AST_table['code'] = AST_table['code'].apply(convert_AST_to_numerical_sequence)
        # Save the processed data step by step
        AST_table.to_pickle(self.data_root_path + dataset_name + '/blocks.pkl')

    # ===================================================================================================
    # The function to process data for training
    def run(self):
        print('Initializing, importing data from local...')
        self.initialize_function()

        print('Parsing source code...')
        self.parse_source_code_data()

        print('Training word embeddings...')
        self.build_dictionary_and_train_word_embeddings()  # EMBEDDING_SIZE: 128 The embedding size for word vectors.

        print('Generating block sequences...')
        self.generate_block_sequences(self.training_AST_data_path, 'train')
        self.generate_block_sequences(self.testing_AST_data_path, 'test')

# ######################################################################################################################
if __name__ == '__main__':
    preprocessing_object = GeneratePreprocessedData(raw_dataset_path=r'../ACEOB-NPI',
                                                       data_root_path=r'ACEOB-NPI-data/')
    preprocessing_object.run()
