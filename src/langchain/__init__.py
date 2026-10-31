import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print("Hello from langchain!")
    print(os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    main()
