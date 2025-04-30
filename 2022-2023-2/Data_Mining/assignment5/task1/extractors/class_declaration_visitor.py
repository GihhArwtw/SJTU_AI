from .visitor import Visitor


class ClassDeclarationVisitor(Visitor):
    def __init__(self):
        super().__init__()

    def get_class_name(self, code: str) -> str:
        tree = self.parser.parse(code.encode())
        root = tree.root_node  # type: program
        class_decl_node = root.children[0]  # type: class_declaration
        class_name = class_decl_node.children[2]  # type: identifier
        return class_name.text.decode()
