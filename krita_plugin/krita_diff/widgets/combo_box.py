from functools import partial

from krita import QComboBox, QHBoxLayout

from ..script import Script
from .misc import QLabel


class QComboBoxLayout(QHBoxLayout):
    def __init__(
        self,
        script: Script,
        options_cfg: str,
        selected_cfg: str,
        label: str = None,
        num_chars: int = None,
        *args,
        **kwargs
    ):
        """Layout for labelled QComboBox.

        Args:
            script (Script): Script to connect to.
            options_cfg (str): Config key to read available options from.
            selected_cfg (str): Config key to read/write selected option to.
            label (str, optional): Label, uses `selected_cfg` if None. Defaults to None.
            num_chars (int, optional): Max length of qcombo in chars. Defaults to None.
        """
        super(QComboBoxLayout, self).__init__(*args, **kwargs)
        self.num_chars = num_chars

        # Used to connect to config stored in script
        self.script = script
        self.options_cfg = options_cfg
        self.selected_cfg = selected_cfg

        self.qlabel = QLabel(self.selected_cfg if label is None else label)
        self.qcombo = QComboBox()

        self.addWidget(self.qlabel)
        self.addWidget(self.qcombo)

    def cfg_init(self):
        # prevent value from getting wiped
        val = self.script.cfg(self.selected_cfg, str)
        self.qcombo.clear()
        self.qcombo.addItems(self.script.cfg(self.options_cfg, "QStringList"))
        self.qcombo.setCurrentText(val)
        if self.num_chars is not None:
            self.qcombo.setFixedWidth(
                self.qcombo.fontMetrics().width("M" * self.num_chars)
            )

    def cfg_connect(self):
        self.qcombo.currentTextChanged.connect(
            partial(self.script.cfg.set, self.selected_cfg)
        )
