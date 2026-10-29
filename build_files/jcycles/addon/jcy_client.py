# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Starting the render worker (the fork's Blender) and talking to it."""

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

from . import jcy_runtime

MESSAGE_PREFIX = "@@JCYCLES@@"


class WorkerError(RuntimeError):
    pass


class WorkerCancelled(RuntimeError):
    pass


_ids = itertools.count(1)
_all_workers = []
_all_lock = threading.Lock()


class Worker:
    """One background process of the fork's Blender."""

    def __init__(self, purpose, autoexec=False, echo=False):
        self.purpose = purpose
        self.autoexec = autoexec
        self.echo = echo
        self.process = None
        self.messages = queue.Queue()
        self.loaded_token = None
        self.log_path = None
        self._log = None
        self._reader = None
        self._tail = []

    # Process lifetime -------------------------------------------------------

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self, timeout=120.0):
        exe = jcy_runtime.executable()
        session = jcy_runtime.session_dir()
        worker_session = os.path.join(session, "worker_" + self.purpose)
        os.makedirs(worker_session, exist_ok=True)
        self.log_path = os.path.join(worker_session, "worker.log")
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        args = [
            exe, "--background", "--factory-startup", "-noaudio",
            "--enable-autoexec" if self.autoexec else "--disable-autoexec",
            "--python-exit-code", "1",
            "--python", jcy_runtime.WORKER_SCRIPT,
            "--",
            "--addon-dir", jcy_runtime.ADDON_DIR,
            "--session-dir", worker_session,
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=jcy_runtime.worker_env(), cwd=worker_session, creationflags=creationflags,
            bufsize=1, text=True, encoding="utf-8", errors="replace")
        self.loaded_token = None
        self._reader = threading.Thread(target=self._read_output, name="JCyclesWorker-" + self.purpose,
                                        daemon=True)
        self._reader.start()
        with _all_lock:
            _all_workers.append(self)
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=0.25)
            except queue.Empty:
                if not self.is_alive():
                    raise WorkerError("The JCycles render process exited at startup:\n" + self.tail())
                if time.monotonic() > deadline:
                    self.stop(kill=True)
                    raise WorkerError("The JCycles render process did not start in time:\n" + self.tail())
                continue
            if message.get("type") == "ready":
                return

    def _read_output(self):
        stream = self.process.stdout
        for line in stream:
            if line.startswith(MESSAGE_PREFIX):
                try:
                    message = json.loads(line[len(MESSAGE_PREFIX):])
                except ValueError:
                    continue
                if message.get("type") == "log":
                    self._write_log("[{:s}] {:s}\n".format(message.get("level", "INFO"), message.get("text", "")))
                    if message.get("level") in {"WARNING", "ERROR"}:
                        print("JCycles:", message.get("text", ""))
                    continue
                self.messages.put(message)
            else:
                self._write_log(line)
                self._tail.append(line.rstrip("\n"))
                if len(self._tail) > 60:
                    del self._tail[:-60]
                if self.echo or "Error" in line or "error:" in line:
                    sys.stdout.write("[JCycles] " + line)
        self.messages.put({"type": "exit", "returncode": self.process.wait()})

    def _write_log(self, text):
        try:
            self._log.write(text)
            self._log.flush()
        except Exception:
            pass

    def tail(self):
        return "\n".join(self._tail[-25:])

    def stop(self, kill=False):
        try:
            self._stop(kill)
        except Exception as ex:
            print("JCycles: stopping the render process:", ex)

    def _stop(self, kill):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            if not kill:
                try:
                    process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
                except Exception:
                    pass
            if process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=10)
                except Exception:
                    pass
        for stream in (process.stdin,):
            try:
                stream.close()
            except Exception:
                pass
        self.loaded_token = None
        with _all_lock:
            if self in _all_workers:
                _all_workers.remove(self)

    # Messaging ---------------------------------------------------------------

    def send(self, command):
        if not self.is_alive():
            raise WorkerError("The JCycles render process is not running:\n" + self.tail())
        command = dict(command)
        command["id"] = next(_ids)
        try:
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as ex:
            raise WorkerError("Cannot talk to the JCycles render process: {:s}\n{:s}".format(str(ex), self.tail()))
        return command["id"]

    def wait(self, request_id, on_event=None, should_cancel=None, timeout=None):
        """Wait for the reply to `request_id`. Events are passed to `on_event`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if should_cancel is not None and should_cancel():
                self.stop(kill=True)
                raise WorkerCancelled()
            try:
                message = self.messages.get(timeout=0.1)
            except queue.Empty:
                if not self.is_alive() and self.messages.empty():
                    raise WorkerError("The JCycles render process stopped unexpectedly:\n" + self.tail())
                if deadline is not None and time.monotonic() > deadline:
                    raise WorkerError("The JCycles render process did not answer in time")
                continue
            kind = message.get("type")
            if kind == "exit":
                raise WorkerError("The JCycles render process exited (code {!r}):\n{:s}".format(
                    message.get("returncode"), self.tail()))
            if kind == "event":
                if on_event is not None:
                    on_event(message)
                continue
            if kind == "reply" and message.get("id") == request_id:
                if not message.get("ok", False):
                    raise WorkerError(message.get("error", "Unknown error in the JCycles render process"))
                return message

    def call(self, command, **kwargs):
        return self.wait(self.send(command), **kwargs)

    def poll_messages(self):
        """Non-blocking: all queued messages."""
        result = []
        while True:
            try:
                result.append(self.messages.get_nowait())
            except queue.Empty:
                return result


def stop_all():
    with _all_lock:
        workers = list(_all_workers)
    for worker in workers:
        worker.stop()


def one_shot(command, timeout=600.0):
    """Run a single command that does not need a scene in a short-lived worker."""
    worker = Worker("tool")
    worker.start()
    try:
        return worker.call(command, timeout=timeout)
    finally:
        worker.stop()
