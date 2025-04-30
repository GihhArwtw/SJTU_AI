from .visitor import Visitor
from typing import List


class MethodInvocationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.method_invocation_list = []
        self.identifier = set()

    def get_method_invocations(self, code: str) -> List[str]:
        ast_tree = self.parser.parse(code.encode())
        root = ast_tree.root_node
        class_body = root.children[0].children[3]
        for child in class_body.children:
            if child.type == 'method_declaration':
                self._get_method_invocation(child)
        
        return self.method_invocation_list
                
    def _get_method_invocation(self, node):
        if not node.children:
            return
        for child in node.children:
            if child.type == 'method_invocation':
                identifier = child.children[0].text.decode()
                text = child.text.decode()
                if identifier in self.identifier.keys():
                    text = text.replace(identifier, self.identifier[identifier], 1)
                argument_list = child.children[-1].text.decode()
                text = text.replace(argument_list, '')
                self.method_invocation_list.append(text)
            elif child.type == 'local_variable_declaration':
                self._get_object_identifier(child)
            self._get_method_invocation(child)
                
    def _get_object_identifier(self, node):
        for child in node.children:
            if child.type == 'type_identifier':
                type_identifier = child.text.decode()
            elif child.type == 'variable_declarator':
                identifier = self._get_identifier(child)
            
        self.identifier[identifier] = type_identifier
                
    def _get_identifier(self, node) -> str:
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode()

