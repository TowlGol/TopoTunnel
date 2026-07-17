#!/usr/bin/env python3
"""
TopoTunnelMiner Desktop UI - PySide6 Native Application

Usage:
    python -m TopoTunnelMiner_UI [db_path_or_pkl]
"""
import sys
import os

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def _configure_qt_webengine():
    """
    Improve QtWebEngine stability on Windows where embedded WebGL views may
    render as a black surface even though JavaScript rendering completed.
    """
    existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    extra_flags = [
        "--ignore-gpu-blocklist",
        "--enable-webgl",
        "--enable-accelerated-2d-canvas",
    ]
    if "--disable-gpu" not in existing and "--disable-gpu-compositing" not in existing:
        extra_flags.append("--use-angle=d3d11")
    merged = " ".join(flag for flag in [existing, *extra_flags] if flag).strip()
    if merged:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = merged


def load_stylesheet(app):
    """Load QSS theme file."""
    qss_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resources", "theme.qss"
    )
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())


def _project_database_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")


def _resolve_db_path(selected_path: str) -> tuple[str, str | None]:
    """
    Resolve a selected .db/.pkl path to the database path used by the app.

    Returns:
        (db_path, pkl_path_to_import)
    """
    selected_path = os.path.abspath(selected_path)
    if selected_path.lower().endswith(".pkl"):
        os.makedirs(_project_database_dir(), exist_ok=True)
        db_name = f"{os.path.splitext(os.path.basename(selected_path))[0]}.db"
        return os.path.join(_project_database_dir(), db_name), selected_path
    return selected_path, None


def main():
    _configure_qt_webengine()

    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName("TopoTunnelMiner")
    app.setOrganizationName("TopoTunnelMiner")

    load_stylesheet(app)

    db_path = None
    pkl_path = None
    if len(sys.argv) > 1:
        db_path, pkl_path = _resolve_db_path(sys.argv[1])
    if pkl_path:
        print(f"Importing {pkl_path} -> {db_path} ...")
        from TopoTunnelMiner_UI.scripts.import_data import import_pkl_to_sqlite
        import_pkl_to_sqlite(pkl_path, db_path)

    if db_path and not os.path.exists(db_path):
        QMessageBox.critical(None, "Error", f"Database not found: {db_path}")
        sys.exit(1)

    from TopoTunnelMiner_UI.app.main_window import MainWindow
    window = MainWindow(db_path)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
