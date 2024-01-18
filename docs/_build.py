#!/Users/jeffnewman/opt/anaconda3/envs/autumn22/bin/python3.1
import re
import sys

from jupyter_book.cli.main import main

if __name__ == "__main__":
    exec(open("_scripts/hide_test_cells.py").read())
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
