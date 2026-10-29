# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Minimal OpenEXR header reader: channel names and string attributes (pure Python)."""

import struct

EXR_MAGIC = 20000630


def _read_cstring(fh):
    data = bytearray()
    while True:
        c = fh.read(1)
        if not c:
            raise ValueError("Unexpected end of EXR header")
        if c == b"\0":
            return data.decode("utf-8", "replace")
        data += c
        if len(data) > 1024:
            raise ValueError("Invalid EXR header")


def _parse_chlist(value):
    channels = []
    i = 0
    while i < len(value) and value[i] != 0:
        end = value.index(b"\0", i)
        channels.append(value[i:end].decode("utf-8", "replace"))
        i = end + 1 + 16
    return channels


def read_header(path):
    """Returns {"channels": [...], "attributes": {name: str}} merged over all parts."""
    channels = []
    attributes = {}
    with open(path, "rb") as fh:
        magic, version = struct.unpack("<ii", fh.read(8))
        if magic != EXR_MAGIC:
            raise ValueError("Not an OpenEXR file: " + path)
        multipart = bool(version & 0x1000)
        while True:
            name = _read_cstring(fh)
            if not name:
                if not multipart:
                    break
                # End of one part header; an empty name right away ends the header list.
                peek = fh.read(1)
                if peek == b"\0" or not peek:
                    break
                fh.seek(-1, 1)
                continue
            attr_type = _read_cstring(fh)
            (size,) = struct.unpack("<i", fh.read(4))
            value = fh.read(size)
            if attr_type == "chlist":
                channels.extend(_parse_chlist(value))
            elif attr_type == "string":
                attributes[name] = value.decode("utf-8", "replace")
    return {"channels": channels, "attributes": attributes}
