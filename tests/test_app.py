from streamlit.testing.v1 import AppTest
from pathlib import Path
import subprocess
import sys


def test_streamlit_chat_shell() -> None:
    app = AppTest.from_file("app.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "Opcenter Chatbot"
    if app.error:
        assert any(
            phrase in app.error[0].value
            for phrase in ("GROQ_API_KEY is missing", "indexes are unavailable", "index schema changed")
        )
        if app.code:
            assert app.code[0].value == "python -m src.ingest"
        return
    assert app.chat_input[0].placeholder == "Ask a question about the Opcenter manuals"
    assert [button.label for button in app.sidebar.button] == [
        "New conversation",
        "Delete current conversation",
    ]
    assert [checkbox.label for checkbox in app.sidebar.checkbox] == [
        "Show sources",
        "Generate diagrams when useful",
    ]


def test_streamlit_does_not_rebuild_indexes() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "ingest_manuals" not in source
    assert "build_indexes" not in source


def test_streamlit_avoids_deprecated_width_and_arrow_table_paths() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "use_container_width" not in source
    assert "st.dataframe(" not in source
    assert 'width="stretch"' in source


def test_streamlit_streams_only_the_verified_final_answer() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "graph.stream(" in source
    assert 'stream_mode="updates"' in source
    assert 'version="v2"' in source
    assert "graph.get_state(config).values" in source
    assert "st.write_stream(" in source
    assert "graph.invoke(" not in source
    assert "status.write(PROGRESS_LABELS[node])" in source
    assert "status.write(event)" not in source
    assert source.index("st.write_stream(") < source.index("render_artifacts(final_state")


def test_progress_labels_are_safe_and_user_facing() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    for label in (
        "Understanding question",
        "Searching manuals",
        "Expanding context",
        "Reranking evidence",
        "Checking coverage",
        "Preparing answer",
        "Verifying answer",
        "Preparing diagram",
    ):
        assert label in source


def test_streamlit_errors_do_not_echo_provider_details() -> None:
    source = Path("app.py").read_text(encoding="utf-8")

    assert "except GroqRequestError:" in source
    assert "logger.exception(" in source
    assert "{exc}" not in source


def test_streamlit_file_watcher_is_disabled() -> None:
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")

    assert 'fileWatcherType = "none"' in config


def test_graph_import_does_not_eagerly_load_torch() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.graph; assert 'torch' not in sys.modules",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert process.returncode == 0, process.stderr
