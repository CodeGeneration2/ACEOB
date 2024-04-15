import ast
from Code_to_AST.prepare_data_Python import get_block_list


# ===================================================================================================
# Generate a sequence of blocks with index representation
def code_to_AST(code_str):


    def tree_to_index(node):
        token_list = node.token
        token_list_filtered = [item for item in token_list if item not in ['Load', 'Store', 'alias']]
        if len(token_list_filtered) == 0:
            return None
        token_str = '[ '
        for token in token_list:
            token_str += str(token)
            token_str += " "

        children = node.children
        for child in children:
            child_sequence = tree_to_index(child)
            if child_sequence:
                token_str += child_sequence

        token_str += ']'
        return token_str

    # Convert the AST into a numeric sequence. e.g., (46887, [[32, [2, [30, [40, [81]]]]], [7],
    def convert_AST_to_number_sequence(AST_node):
        block_list = []
        get_block_list(AST_node, block_list)
        tree = '{ '
        for block in block_list:
            btree = tree_to_index(block)
            if btree:
                tree += btree
                tree += ' '
        tree += '}'
        return tree


    AST_model = ast.parse(code_str)

    AST_str = convert_AST_to_number_sequence(AST_model)

    return AST_str
