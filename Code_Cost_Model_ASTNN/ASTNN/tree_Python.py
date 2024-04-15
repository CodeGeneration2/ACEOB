import ast

class ASTNode(object):
    def __init__(self, node):
        self.node = node
        self.token = self.get_token()
        self.children = self.add_children()

    def get_token(self, node=None) -> "list":
        if node is None:
            node = self.node
        token = type(node).__name__
        result_list = []

        # Special handling for certain nodes to extract more meaningful tokens
        # `isinstance` is a built-in function in Python used to check if an object has a given attribute.

        if isinstance(node, ast.Name):
            token = node.id
        elif isinstance(node, ast.Constant):
            token = node.value
        elif isinstance(node, ast.arg):
            token = node.arg

        result_list.append(token)

        if hasattr(node, "name"):
            result_list.append(node.name)
        elif hasattr(node, "module"):
            result_list.append(node.module)

        if isinstance(node, ast.Attribute):
            # Use the attr attribute directly, which is a string
            attribute_value = node.attr
            # Recursively call `get_token` on value to handle nested attribute access
            first_child = list(ast.iter_child_nodes(node))[0]
            prefix_list = self.get_token(first_child)
            # Maintain access order, i.e., object before attribute
            result_list.extend(prefix_list)
            result_list.append(attribute_value)

        return result_list

    def add_children(self):
        children = list(ast.iter_child_nodes(self.node))
        return [ASTNode(child) for child in children]

class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.token = self.get_token()
        self.children = []
