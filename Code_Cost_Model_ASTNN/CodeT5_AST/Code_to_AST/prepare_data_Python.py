from Code_to_AST.tree_Python import ASTNode, SingleNode

import ast

def recursively_get_AST_sequence(AST_node, AST_sequence_list):
    current_node = SingleNode(AST_node)
    AST_sequence_list.extend(current_node.get_token())
    if isinstance(AST_node, ast.Attribute):
        return
    child_nodes = list(ast.iter_child_nodes(AST_node))
    for child in child_nodes:
        recursively_get_AST_sequence(child, AST_sequence_list)
    # if current_node.get_token().lower() == 'compound':
    #     AST_sequence_list.append('End')


def get_block_list(AST_node, block_list):
    child_nodes_list = list(ast.iter_child_nodes(AST_node))
    node_class_name = type(AST_node).__name__
    if node_class_name in ['FunctionDef', 'If', 'For', 'While']:
        block_list.append(ASTNode(AST_node))
        for child in child_nodes_list:
            if type(child).__name__ not in ['FunctionDef', 'If', 'For', 'While']:
                block_list.append(ASTNode(child))
            get_block_list(child, block_list)
    else:
        for child in child_nodes_list:
            get_block_list(child, block_list)
