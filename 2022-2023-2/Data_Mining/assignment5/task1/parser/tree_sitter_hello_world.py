from tree_sitter import Language, Parser


def main():
    parser = Parser()
    parser.set_language(Language('./my-languages.so', 'java'))
    code = '''
        class MyClass {
          public void readText(String file) { 
          BufferedReader br = new BufferedReader(new FileInputStream(file)); 
          String line = null; 
          while ((line = br.readLine())!= null) { 
            System.out.println(line); 
          } 
          br.close(); 
          }
        }    
    '''
    tree = parser.parse(code.encode())
    root = tree.root_node
    if root.has_error:
        raise ValueError('original code is invalid')
    else:
        print('Hello Tree Sitter.')


if __name__ == '__main__':
    main()
