"""Launch the oropt Streamlit GUI.

Streamlit apps cannot be started as a plain script (``python oropt/gui/app.py``)
— they need the Streamlit runtime. This wrapper boots it programmatically, so you
can run it from a terminal *or* with PyCharm's green Run button:

    python run_gui.py
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli


def _silence_proactor_connection_reset() -> None:
    """Swallow the benign ``ConnectionResetError`` (WinError 10054) that
    asyncio's ``ProactorEventLoop`` logs whenever a browser client drops its
    socket. It fires from inside a transport callback on Windows, so there is
    nothing to catch it and it surfaces as a noisy "Exception in callback".
    Only that specific error is suppressed; real failures still propagate.
    """
    if sys.platform != "win32":
        return

    from asyncio.proactor_events import _ProactorBasePipeTransport

    original = _ProactorBasePipeTransport._call_connection_lost

    def _call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost


def main() -> int:
    _silence_proactor_connection_reset()
    app = Path(__file__).resolve().parent / "oropt" / "gui" / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
