import unittest
from extractors import ClassDeclarationVisitor, MethodDeclarationVisitor, ObjectCreationVisitor, MethodInvocationVisitor


class MyTest(unittest.TestCase):
    def test_get_class_name(self):
        code = '''
          public class MyClass {
            public void readText(String file) { 
              System.out.println('Hello World.'); 
            }
          }
        '''

        visitor = ClassDeclarationVisitor()
        class_name = visitor.get_class_name(code)
        self.assertEqual(class_name, 'MyClass')

    def test_get_method_name(self):
        code = '''
          public class MyClass {
            public void readText(String file) { 
              System.out.println('Hello World.'); 
            }
            public static void printName() {
              System.out.println("MyClass");
            }
          }
        '''

        visitor = MethodDeclarationVisitor()
        method_names = visitor.get_method_names(code)
        self.assertEqual(len(method_names), 2)
        for (actual_name, expected_name) in zip(method_names, ['readText', 'printName']):
            self.assertEqual(actual_name, expected_name)

    def test_get_object_creation(self):
        code = '''
          public class MyClass {
            public void readText(String file) { 
              BufferedReader br = new BufferedReader(new FileInputStream(file));
            }
          }
        '''
        visitor = ObjectCreationVisitor()
        object_creations = visitor.get_object_creations(code)
        self.assertEqual(len(object_creations), 2)
        for (actual_name, expected_name) in zip(object_creations, ['BufferedReader', 'FileInputStream']):
            self.assertEqual(actual_name, expected_name)

    def test_get_method_invocation(self):
        code = '''
            public class MyClass {
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
        visitor = MethodInvocationVisitor()
        method_invocations = visitor.get_method_invocations(code)
        self.assertEqual(len(method_invocations), 3)
        expected_names = ['BufferedReader.readLine', 'System.out.println', 'BufferedReader.close']
        for (actual_name, expected_name) in zip(method_invocations, expected_names):
            self.assertEqual(actual_name, expected_name)
