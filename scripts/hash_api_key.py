from argparse import ArgumentParser

from spamshield.common import signature


def main() -> None:
    parser = ArgumentParser("hash-api-key")
    parser.add_argument("--key", "-k", type=str, required=True, help="API key to hash")

    args = parser.parse_args()

    hashed_key: str = signature.hash_api_key(args.key)

    print(hashed_key)


if __name__ == "__main__":
    main()
