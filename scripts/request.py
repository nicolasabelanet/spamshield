from spamshield.client.client import SpamShieldAPIClient
from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument("--url", "-u", type=str, required=True)
    parser.add_argument("--message", "-m", type=str, required=True)

    args = parser.parse_args()

    client = SpamShieldAPIClient(args.url, api_key="dev-key", api_secret="foo")
    response = client.predict([args.message])

    print(response)


if __name__ == "__main__":
    main()
