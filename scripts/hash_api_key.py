from argparse import ArgumentParser

from spamshield.common import signature


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument("--key", "-k", type=str, required=True)

    args = parser.parse_args()

    hashed_key: str = signature.hash_api_key(args.key)

    print(hashed_key)


if __name__ == "__main__":
    main()
