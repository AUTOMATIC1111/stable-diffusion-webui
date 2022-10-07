from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita

from .hotkeys import Hotkeys
from .krita_diff_ui import *

instance = Krita.instance()
instance.addExtension(Hotkeys(instance))
instance.addDockWidgetFactory(
    DockWidgetFactory("krita_diff", DockWidgetFactoryBase.DockLeft, KritaSDPluginDocker)
)
