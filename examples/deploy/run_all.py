#!/usr/bin/env python

import os
import glob
import argparse
import subprocess

EXE_EXTENSIONS = {"cc" : ".exe", "csharp" : "_cs.exe", "python" : ".py"}

def run_example(lang, ex, base_path):
    exe_path = os.path.join(os.getcwd(), lang, ex + EXE_EXTENSIONS[lang])
    model_path = os.path.join(base_path, "deploy", ex, ex + ".pb")
    images = glob.glob(os.path.join(base_path, "deploy", ex, "*.jpg"))
    if not os.path.exists(exe_path):
        print("Searched for exe at {}".format(exe_path))
        msg = "Could not find {} executable for {} function, skipping..."
        print(msg.format(lang, ex))
        return 
    if not os.path.exists(model_path):
        msg = "Could not find model file for {} function, skipping..."
        print(msg.format(ex))
        return
    cmd = [exe_path, model_path]
    cmd += images
    if lang == "python":
        cmd = ["python"] + cmd
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs all examples.")
    parser.add_argument("example_path",
        type=str,
        help="Base path to OpenEM example data.")
    parser.add_argument("--langs",
        nargs="+",
        default=["cc", "csharp", "python"],
        help="Languages to run.")
    parser.add_argument("--funcs",
        nargs="+",
        default=["find_ruler", "detect", "classify"],
        help="Functions to run.")
    args = parser.parse_args()
    for lang in args.langs:
        for ex in args.funcs:
            run_example(lang, ex, args.example_path)
    
