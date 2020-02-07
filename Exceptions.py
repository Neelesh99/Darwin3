import sys

class Error(Exception):
    pass

class GeneError(Error):
    def __init__(self, message):
        self.message = message

class GenomeError(Error):
    def __init__(self, message):
        self.message = message
