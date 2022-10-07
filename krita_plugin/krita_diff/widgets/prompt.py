from krita import QLabel, QPlainTextEdit, QSizePolicy, QVBoxLayout

from ..script import Script


class QPromptEdit(QPlainTextEdit):
    placeholder: str = "Enter prompt..."
    num_lines: int = 5
    """height of prompt box in lines"""

    def __init__(self, *args, **kwargs):
        super(QPromptEdit, self).__init__(*args, **kwargs)
        self.setPlaceholderText(self.placeholder)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * self.num_lines)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)


class QPromptLayout(QVBoxLayout):
    prompt_label: str = "Prompt:"
    neg_prompt_label: str = "Negative Prompt:"

    def __init__(
        self, script: Script, prompt_cfg: str, neg_prompt_cfg: str, *args, **kwargs
    ):
        """Layout for prompt and negative prompt.

        Args:
            script (Script): Script to connect to.
            prompt_cfg (str): Config key to read/write prompt to.
            neg_prompt_cfg (str): Config key to read/write negative prompt to.
        """
        super(QPromptLayout, self).__init__(*args, **kwargs)

        # Used to connect to config stored in script
        self.script = script
        self.prompt_cfg = prompt_cfg
        self.neg_prompt_cfg = neg_prompt_cfg

        self.qlabel_prompt = QLabel(self.prompt_label)
        self.qedit_prompt = QPromptEdit()
        self.qlabel_neg_prompt = QLabel(self.neg_prompt_label)
        self.qedit_neg_prompt = QPromptEdit()

        self.addWidget(self.qlabel_prompt)
        self.addWidget(self.qedit_prompt)
        self.addWidget(self.qlabel_neg_prompt)
        self.addWidget(self.qedit_neg_prompt)

    def cfg_init(self):
        self.qedit_prompt.setPlainText(self.script.cfg(self.prompt_cfg, str))
        self.qedit_neg_prompt.setPlainText(self.script.cfg(self.neg_prompt_cfg, str))

    def cfg_connect(self):
        self.qedit_prompt.textChanged.connect(
            lambda: self.script.set_cfg(
                self.prompt_cfg, self.qedit_prompt.toPlainText()
            )
        )
        self.qedit_neg_prompt.textChanged.connect(
            lambda: self.script.set_cfg(
                self.neg_prompt_cfg, self.qedit_neg_prompt.toPlainText()
            )
        )
