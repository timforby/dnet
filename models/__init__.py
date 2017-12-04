from os.path import isfile
from os import listdir

fn = listdir("./models")
__all__ = [ m[:-3] for m in fn if isfile(m) and  m.endswith('.py') and not m.endswith('__init__.py')]
