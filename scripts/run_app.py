import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli

# Insert path before custom module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    app_path = Path(__file__).parent.parent / "src" / "app" / "streamlit_app.py"
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
    ]
    sys.exit(stcli.main())
