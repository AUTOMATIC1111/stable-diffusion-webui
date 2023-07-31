class PDHError(Exception):
    def __init__(self, message: str):
        super(PDHError, self).__init__(message)
