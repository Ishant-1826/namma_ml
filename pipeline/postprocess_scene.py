#!/usr/bin/env python3
"""Post-process and compress a trained Gaussian Splat model.

Stage 5 of the Namma Space pipeline.
Reads trained splat from outputs/<scene>/splat/.
Writes compressed/pruned output to outputs/<scene>/final/.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-process / compress a trained Gaussian Splat."
    )
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # TODO: implement postprocessing (prune, compress, export)
    print(f"[postprocess_scene] Not yet implemented for scene '{args.scene}'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
