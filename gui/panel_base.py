# -*- coding: utf-8 -*-
"""Owner-proxy base used by the composed ModelCenterDialog panels.

The original dialog methods stored widget and data attributes directly on the
dialog. Redirecting panel assignments to the owner preserves that public state
contract while allowing the methods to live in separate composed objects.
"""

from PyQt5 import QtWidgets


class OwnerPanel(QtWidgets.QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        object.__setattr__(self, "_owner", owner)
        self.hide()

    @property
    def owner(self):
        return object.__getattribute__(self, "_owner")

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name == "owner" or "_owner" not in self.__dict__:
            return super().__setattr__(name, value)
        setattr(self.owner, name, value)
