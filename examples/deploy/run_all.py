#!/usr/bin/env python

import os
import glob
import argparse
import subprocess

EXE_EXTENSIONS = {"cc" : ".exe", "csharp" : "_cs.exe", "python" : ".py"}

def find_model(base_path, ex):
    return os.path.join(base_path, "deploy", ex, ex + ".pb")

def run_example(lang, ex, base_path):
    exe_path = os.path.join(os.getcwd(), lang, ex + EXE_EXTENSIONS[lang])
    if ex == "video":
        models = ["find_ruler", "detect", "classify"]
        model_paths = [find_model(base_path, m) for m in models]
        inputs = glob.glob(os.path.join(base_path, "deploy", ex, "*.mp4"))
    else:
        model_paths = [find_model(base_path, ex)]
        inputs = glob.glob(os.path.join(base_path, "deploy", ex, "*.jpg"))
    if not os.path.exists(exe_path):
        print("Searched for exe at {}".format(exe_path))
        msg = "Could not find {} executable for {} function, skipping..."
        print(msg.format(lang, ex))
        return 
    for model_path in model_paths:
        if not os.path.exists(model_path):
            msg = "Could not find model file for {} function, skipping..."
            print(msg.format(ex))
            return
    cmd = [exe_path]
    cmd += model_paths
    cmd += inputs
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
        default=["find_ruler", "detect", "classify", "video"],
        help="Functions to run.")
    args = parser.parse_args()
    for lang in args.langs:
        for ex in args.funcs:
            run_example(lang, ex, args.example_path)
    
