from tree_Python import ASTNode, SingleNode

import ast

def recursively_get_ast_sequence(ast_node, ast_sequence_list):
    current_node = SingleNode(ast_node)
    ast_sequence_list.extend(current_node.get_token())
    if isinstance(ast_node, ast.Attribute):
        return
    children_nodes = list(ast.iter_child_nodes(ast_node))
    for child in children_nodes:
        recursively_get_ast_sequence(child, ast_sequence_list)
    # if current_node.get_token().lower() == 'compound':
    #     ast_sequence_list.append('End')


def get_block_list(ast_node, block_list):
    children_nodes_list = list(ast.iter_child_nodes(ast_node))
    node_class_name = type(ast_node).__name__
    if node_class_name in ['FunctionDef', 'If', 'For', 'While']:
        block_list.append(ASTNode(ast_node))
        for child in children_nodes_list:
            if type(child).__name__ not in ['FunctionDef', 'If', 'For', 'While']:
                block_list.append(ASTNode(child))
            get_block_list(child, block_list)
    else:
        for child in children_nodes_list:
            get_block_list(child, block_list)

