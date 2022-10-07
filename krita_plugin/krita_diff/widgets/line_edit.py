from functools import partial

from krita import QHBoxLayout, QLabel, QLineEdit

from ..script import Script


class QLineEditLayout(QHBoxLayout):
    def __init__(
        self,
        script: Script,
        field_cfg: str,
        label: str = None,
        placeholder: str = "",
        *args,
        **kwargs
    ):
        """Layout for labelled QLineEdit.

        Args:
            script (Script): Script to connect to.
            field_cfg (str): Config key to read/write value to.
            label (str, optional): Label, uses `field_cfg` if None. Defaults to None.
            placeholder (str, optional): Placeholder. Defaults to "".
        """
        super(QLineEditLayout, self).__init__(*args, **kwargs)

        self.script = script
        self.field_cfg = field_cfg

        self.qedit = QLineEdit()
        self.qedit.setPlaceholderText(placeholder)
        self.addWidget(QLabel(field_cfg if label is None else label))
        self.addWidget(self.qedit)

    def cfg_init(self):
        self.qedit.setText(self.script.cfg(self.field_cfg, str))

    def cfg_connect(self):
        self.qedit.textChanged.connect(partial(self.script.set_cfg, self.field_cfg))
