#!/usr/bin/env python3
import os
import argparse
from engine import ensure_cv2
from gui import AnalysisGUI

def main():
    ensure_cv2()
    parser = argparse.ArgumentParser(description="Professionelle 3D-Handball-Wurfanalyse")
    parser.add_argument("--reference-json", type=str, default=None)
    args = parser.parse_args()

    app = AnalysisGUI(reference_json=args.reference_json)
    app.run()

if __name__ == "__main__":
    main()