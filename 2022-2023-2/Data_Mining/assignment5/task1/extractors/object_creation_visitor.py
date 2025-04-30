from .visitor import Visitor
from typing import List


class ObjectCreationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.object_creation_list = []

    def get_object_creations(self, code: str) -> List[str]:
        ast_tree = self.parser.parse(code.encode())
        root = ast_tree.root_node
        class_body = root.children[0].children[3]
        for child in class_body.children:
            if child.type == 'method_declaration':
                self._get_object_creation(child)
        return self.object_creation_list
    
    def _get_object_creation(self, node):
        # recursion
        if not node.children:
            return
        first = False
        for child in node.children:
            # traverse the tree
            if first == True:
                self.object_creation_list.append(child.text.decode())
                first = False
            if child.type == 'new':
                first = True
            self._get_object_creation(child)        # traverse the tree