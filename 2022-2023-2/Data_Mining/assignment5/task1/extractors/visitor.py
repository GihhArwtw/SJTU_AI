from tree_sitter import Language, Parser
from pathlib import Path
import os

filepath = Path(__file__).parent.absolute()


class Visitor:
    def __init__(self):
        self.parser = Parser()

        my_languages_path = Path.joinpath(filepath, '../parser/my-languages.so')
        if os.path.isfile(my_languages_path):
            self.parser.set_language(Language(my_languages_path, 'java'))
        else:
            raise OSError('cannot find my-languages.so')