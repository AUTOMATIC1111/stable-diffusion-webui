from gradio_client.documentation import document, set_documentation_group

set_documentation_group("helpers")


class DuplicateBlockError(ValueError):
    """Raised when a Blocks contains more than one Block with the same id"""

    pass


class TooManyRequestsError(Exception):
    """Raised when the Hugging Face API returns a 429 status code."""

    pass


class InvalidApiNameError(ValueError):
    pass


InvalidApiName = InvalidApiNameError  # backwards compatibility


@document()
class Error(Exception):
    """
    This class allows you to pass custom error messages to the user. You can do so by raising a gr.Error("custom message") anywhere in the code, and when that line is executed the custom message will appear in a modal on the demo.

    Demos: calculator
    """

    def __init__(self, message: str):
        """
        Parameters:
            message: The error message to be displayed to the user.
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return repr(self.message)
