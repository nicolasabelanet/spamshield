from argparse import ArgumentParser

from spamshield.common import signature


def main() -> None:
    """
    Entry point for the `hash-api-key` CLI tool.

    Parses a single `--key` or `-k` argument from the command line,
    computes its secure hash using `signature.hash_api_key()`, and
    prints the result to standard output.

    Raises
    ------
    SystemExit
        If the `--key` argument is missing or invalid.
    """
    parser = ArgumentParser("hash-api-key")
    parser.add_argument("--key", "-k", type=str, required=True, help="API key to hash")

    args = parser.parse_args()

    hashed_key: str = signature.hash_api_key(args.key)

    print(hashed_key)


if __name__ == "__main__":
    main()
