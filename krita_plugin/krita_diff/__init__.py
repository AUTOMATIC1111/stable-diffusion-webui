from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita

from .docker import SDPluginDocker
from .hotkeys import Hotkeys

instance = Krita.instance()
instance.addExtension(Hotkeys(instance))
instance.addDockWidgetFactory(
    DockWidgetFactory("krita_diff", DockWidgetFactoryBase.DockLeft, SDPluginDocker)
)
