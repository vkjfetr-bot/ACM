"""Basic HTML report builder placeholder."""

import argparse


def build_report(artifacts_dir: str, equip: str) -> None:
    raise NotImplementedError("acm_report_basic.build_report pending implementation")


def main() -> None:
    parser = argparse.ArgumentParser("acm_report_basic")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--equip", default="equipment")
    args = parser.parse_args()
    build_report(args.artifacts, args.equip)


if __name__ == "__main__":
    main()
