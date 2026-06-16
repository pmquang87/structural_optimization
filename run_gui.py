"""Launch the oropt Streamlit GUI.

Streamlit apps cannot be started as a plain script (``python oropt/gui/app.py``)
— they need the Streamlit runtime. This wrapper boots it programmatically, so you
can run it from a terminal *or* with PyCharm's green Run button:

    python run_gui.py
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> int:
    app = Path(__file__).resolve().parent / "oropt" / "gui" / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
