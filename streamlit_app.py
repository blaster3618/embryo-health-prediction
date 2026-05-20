"""Streamlit Community Cloud entrypoint.

The main app lives in src/app/streamlit_app.py so it can also be run locally
from the project structure documented in README.md.
"""

import runpy


runpy.run_path("src/app/streamlit_app.py", run_name="__main__")
