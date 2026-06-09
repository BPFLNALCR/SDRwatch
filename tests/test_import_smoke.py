from __future__ import annotations


def test_cli_module_import_smoke() -> None:
    import sdrwatch.cli as cli

    assert callable(cli.parse_args)
    assert callable(cli.run)


def test_web_factory_import_smoke() -> None:
    from sdrwatch_web import create_app

    assert callable(create_app)


def test_web_factory_creation_without_hardware(tmp_path) -> None:
    from sdrwatch_web import create_app

    db_path = tmp_path / "missing.db"
    app = create_app(str(db_path))

    assert app.config["DB_PATH"] == str(db_path)
    assert app.config["CONTROLLER_CLIENT"] is not None