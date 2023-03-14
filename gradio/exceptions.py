class DuplicateBlockError(ValueError):
    """Raised when a Blocks contains more than one Block with the same id"""

    pass


class TooManyRequestsError(Exception):
    """Raised when the Hugging Face API returns a 429 status code."""

    pass


class InvalidApiName(ValueError):
    pass


class Error(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return repr(self.message)
