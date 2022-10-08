from krita import QLabel as _QLabel
from krita import Qt


class QLabel(_QLabel):
    """QLabel with overwritten default behaviours."""

    def __init__(self, *args, **kwargs):
        super(QLabel, self).__init__(*args, **kwargs)

        self.setOpenExternalLinks(True)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
