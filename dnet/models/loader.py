def load(module):
    return __import__("models."+module, fromlist=[''])