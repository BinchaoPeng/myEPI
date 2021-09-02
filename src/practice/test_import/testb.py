import sys

# sys.path.extend(['/home/pbc/Documents/PycharmProjects/myEPI', '/home/pbc/Documents/PycharmProjects/myEPI/src'])
import os

print(os.getcwd())
print(os.path.dirname(__file__))

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
print(root_path[0] + 'src')
sys.path.extend([root_path[0] + 'src'])
print(sys.path)
from practice.test_import import testa

testa.gg()
