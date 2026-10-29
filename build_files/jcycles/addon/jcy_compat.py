# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Keeping the fork's UI code working in other official Blender versions.

The panels are the fork's own (Blender 5.3 API). If an official Blender lacks something a panel
uses, the panel shows a short error line instead of failing silently or spamming the console.
"""

import functools

_reported = set()


def _report_once(cls, what, ex):
    key = (cls.__name__, what, type(ex).__name__, str(ex))
    if key in _reported:
        return
    _reported.add(key)
    print("JCycles: {:s}.{:s}: {:s}: {:s}".format(cls.__name__, what, type(ex).__name__, str(ex)))


def _wrap_draw(cls, name):
    original = cls.__dict__.get(name)
    if original is None or getattr(original, "_jcycles_wrapped", False):
        return

    @functools.wraps(original)
    def wrapper(self, context):
        try:
            return original(self, context)
        except Exception as ex:
            _report_once(cls, name, ex)
            try:
                self.layout.label(text="Not available in this Blender: " + str(ex)[:120], icon='ERROR')
            except Exception:
                pass

    wrapper._jcycles_wrapped = True
    setattr(cls, name, wrapper)


def _wrap_poll(cls):
    original = cls.__dict__.get("poll")
    if original is None or not isinstance(original, classmethod):
        return
    function = original.__func__
    if getattr(function, "_jcycles_wrapped", False):
        return

    @functools.wraps(function)
    def wrapper(klass, context):
        try:
            return function(klass, context)
        except Exception as ex:
            _report_once(cls, "poll", ex)
            return False

    wrapper._jcycles_wrapped = True
    cls.poll = classmethod(wrapper)


def harden(classes):
    for cls in classes:
        for name in ("draw", "draw_header", "draw_header_preset"):
            _wrap_draw(cls, name)
        _wrap_poll(cls)
