from functools import partial
from typing import Union

from krita import QDoubleSpinBox, QHBoxLayout, QSpinBox

from ..script import Script
from .misc import QLabel


class QSpinBoxLayout(QHBoxLayout):
    def __init__(
        self,
        script: Script,
        field_cfg: str,
        label: str = None,
        min: Union[int, float] = 0.0,
        max: Union[int, float] = 1.0,
        step: Union[int, float] = 0.1,
        *args,
        **kwargs
    ):
        """Layout for labelled QSpinBox/QDoubleSpinBox.

        Will infer which to use based on type of min, max and step.

        Args:
            script (Script): Script to connect to.
            field_cfg (str): Config key to read/write value to.
            label (str, optional): Label, uses `field_cfg` if None. Defaults to None.
            min (Union[int, float], optional): Min value. Defaults to 0.0.
            max (Union[int, float], optional): Max value. Defaults to 1.0.
            step (Union[int, float], optional): Value step. Defaults to 0.1.
        """
        super(QSpinBoxLayout, self).__init__(*args, **kwargs)

        self.script = script
        self.field_cfg = field_cfg

        self.qlabel = QLabel(field_cfg if label is None else label)

        is_integer = (
            float(step).is_integer()
            and float(min).is_integer()
            and float(max).is_integer()
        )
        self.cast = int if is_integer else float

        self.qspin = QSpinBox() if is_integer else QDoubleSpinBox()
        self.qspin.setMinimum(self.cast(min))
        self.qspin.setMaximum(self.cast(max))
        self.qspin.setSingleStep(self.cast(step))
        self.addWidget(self.qlabel)
        self.addWidget(self.qspin)

    def cfg_init(self):
        self.qspin.setValue(self.script.cfg(self.field_cfg, self.cast))

    def cfg_connect(self):
        self.qspin.valueChanged.connect(partial(self.script.cfg.set, self.field_cfg))
