from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

from varygen.__main__ import main  # pylint: disable=wrong-import-position

main(None)
