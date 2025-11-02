from spamshield.client.client import SpamShieldAPIClient
from argparse import ArgumentParser


def main() -> None:
    """
    Entry point for the `request-spam-ham-prediction` CLI tool.

    Parses command-line arguments, initializes a `SpamShieldAPIClient`
    with the provided URL, and sends a single prediction request.

    Raises
    ------
    SystemExit
        If required arguments are missing or invalid.

    Notes
    -----
    - The client prints the full JSON response to stdout.
    - Default API credentials are hardcoded for local development only.
    """
    parser = ArgumentParser("request-spam-ham-prediction")

    parser.add_argument(
        "--url",
        "-u",
        type=str,
        required=True,
        help="Base URL of the SpamShield API (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        required=True,
        help="Text message to classify as spam or ham",
    )

    args = parser.parse_args()

    client = SpamShieldAPIClient(args.url, api_key="dev-key", api_secret="dev-secret")
    response = client.predict([args.message])

    print(response)


if __name__ == "__main__":
    main()
