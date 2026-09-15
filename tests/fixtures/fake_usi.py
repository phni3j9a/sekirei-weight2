#!/usr/bin/env python3
"""Tiny deterministic USI fixture used by benchmark protocol tests."""

import os
import subprocess
import sys
import time


MODE = os.environ.get("FAKE_USI_MODE", "exact")
CHILD_PID_FILE = os.environ.get("FAKE_USI_CHILD_PID_FILE")


def emit(line):
    print(line, flush=True)


def advertised_options():
    # Advertise the union of the two pinned option sets so the fixture can be
    # used for either engine without weakening the runner's option check.
    for name, kind, default in (
        ("Threads", "spin", "1"),
        ("USI_Hash", "spin", "128"),
        ("Hash", "spin", "128"),
        ("MultiPV", "spin", "1"),
        ("USI_Ponder", "check", "false"),
        ("Ponder", "check", "false"),
        ("UseBook", "check", "false"),
        ("BookFile", "string", "no_book"),
        ("FV_SCALE", "spin", "28"),
        ("OutputFailLHPV", "check", "false"),
        ("PvInterval", "spin", "0"),
        ("SearchMode", "combo", "Speculative"),
        ("SpecTopN", "spin", "0"),
        ("EvalDir", "string", "/tmp/fake-eval"),
    ):
        if kind == "spin":
            emit(f"option name {name} type spin default {default} min 0 max 1000000")
        elif kind == "check":
            emit(f"option name {name} type check default {default}")
        elif kind == "combo":
            emit(f"option name {name} type combo default {default} var Speculative var Normal")
        else:
            emit(f"option name {name} type string default {default}")


def run_search():
    if MODE == "timeout":
        child = subprocess.Popen(["sleep", "60"])
        if CHILD_PID_FILE:
            with open(CHILD_PID_FILE, "w", encoding="ascii") as stream:
                stream.write(str(child.pid))
        while True:
            time.sleep(1)
    if MODE in (
        "parent-exit-child",
        "timeout-parent-exit",
        "immediate-parent-exit-child",
        "immediate-setsid-parent-exit-child",
    ):
        if MODE == "immediate-setsid-parent-exit-child":
            child = subprocess.Popen([
                sys.executable,
                "-c",
                "import os, signal, time; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ])
        else:
            child = subprocess.Popen(["sleep", "60"])
        if CHILD_PID_FILE:
            with open(CHILD_PID_FILE, "w", encoding="ascii") as stream:
                stream.write(str(child.pid))
        if MODE in ("immediate-parent-exit-child", "immediate-setsid-parent-exit-child"):
            # Exercise the race where a parent exits before a polling cleanup
            # thread can observe the legitimate long-lived descendant.
            os._exit(0)
        if MODE == "timeout-parent-exit":
            time.sleep(0.15)
        else:
            # Keep a delayed parent-exit variant alongside the immediate
            # _exit fixture for comparison coverage.
            time.sleep(0.05)
        raise SystemExit(0)
    if MODE == "exit":
        raise SystemExit(7)
    if MODE == "unknown-option":
        emit("unknown option: EvalDir")
        emit("bestmove 7g7f")
        return
    if MODE == "no-score":
        emit("bestmove 7g7f")
        return
    if MODE == "resign":
        emit("bestmove resign")
        return
    if MODE == "win":
        emit("bestmove win")
        return
    if MODE == "other-special":
        emit("bestmove 0000")
        return
    if MODE == "mismatch":
        emit("info depth 10 score cp 12 nodes 1000 pv 7g7f")
        emit("bestmove 3c3d")
        return
    if MODE == "bound":
        emit("info depth 9 score cp 10 nodes 400 time 4 pv 7g7f")
        emit("info depth 10 score cp 20 lowerbound nodes 1001 time 5 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "upperbound":
        emit("info depth 10 score cp -20 upperbound nodes 1001 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "multipv":
        emit("info multipv 1 depth 10 score cp 20 nodes 1000 pv 7g7f")
        emit("info multipv 2 depth 10 score cp -300 nodes 1000 pv 3c3d")
        emit("bestmove 7g7f")
        return
    if MODE == "final-bound":
        emit("info depth 9 score cp 10 nodes 400 pv 7g7f")
        emit("info depth 10 score cp 20 lowerbound nodes 1000 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "node-anomaly":
        emit("info depth 10 score cp 20 nodes 1101 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "early":
        emit("info depth 10 score cp 20 nodes 900 time 6 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "mate-plus":
        emit("info depth 10 score mate +3 nodes 1000 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "mate-minus":
        emit("info depth 10 score mate -3 nodes 1000 pv 7g7f")
        emit("bestmove 7g7f")
        return
    if MODE == "mate-minus-zero":
        emit("info depth 10 score mate -0 nodes 1000 pv 7g7f")
        emit("bestmove 7g7f")
        return
    emit("info depth 10 score cp 20 nodes 1000 time 6 pv 7g7f")
    emit("bestmove 7g7f")


for command in sys.stdin:
    command = command.strip()
    if command == "usi":
        emit("id name fake-usi")
        emit("id author test")
        advertised_options()
        emit("usiok")
    elif command.startswith("setoption"):
        if MODE == "bad-option" and "Threads" in command:
            emit("unknown option Threads")
    elif command == "isready":
        if MODE == "startup-error":
            emit("error! startup failed")
        emit("readyok")
    elif command == "go nodes 1000" or command.startswith("go nodes "):
        run_search()
    elif command == "quit":
        break
