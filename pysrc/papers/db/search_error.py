class SearchError(RuntimeError):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(RuntimeError, self).__init__(message)