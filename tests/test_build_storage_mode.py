import build_app


def test_get_package_storage_mode_defaults_to_portable(monkeypatch):
    monkeypatch.delenv("GAKUMAS_PACKAGE_STORAGE_MODE", raising=False)

    assert build_app._get_package_storage_mode([]) == build_app.STORAGE_MODE_PORTABLE


def test_get_package_storage_mode_accepts_merged_aliases(monkeypatch):
    monkeypatch.delenv("GAKUMAS_PACKAGE_STORAGE_MODE", raising=False)

    assert build_app._get_package_storage_mode(["--merged"]) == build_app.STORAGE_MODE_MERGED
    assert build_app._get_package_storage_mode(["--storage-mode=managed"]) == build_app.STORAGE_MODE_MERGED


def test_resolve_nuitka_runtime_target_uses_macos_portable_dir(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    app_dist_dir = output_dir / f"{build_app.PROJECT_NAME}.dist"
    app_dist_dir.mkdir(parents=True)

    monkeypatch.setattr(build_app, "TARGET_PLATFORM", "Darwin")
    monkeypatch.setattr(build_app, "NUITKA_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_app, "MACOS_PORTABLE_DIR", output_dir / build_app.PROJECT_NAME)
    monkeypatch.setattr(build_app, "APP_DIST_DIR", app_dist_dir)
    monkeypatch.setattr(build_app, "APP_BUNDLE_DIR", output_dir / f"{build_app.PROJECT_NAME}.app")

    assert build_app._resolve_nuitka_runtime_target_dir(build_app.STORAGE_MODE_PORTABLE) == app_dist_dir


def test_finalize_macos_portable_renames_nuitka_output(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    portable_dir = output_dir / build_app.PROJECT_NAME
    app_dist_dir = output_dir / f"{build_app.PROJECT_NAME}.dist"
    app_dist_dir.mkdir(parents=True)
    binary_path = app_dist_dir / build_app.PROJECT_NAME
    binary_path.write_text("binary", encoding="utf-8")

    monkeypatch.setattr(build_app, "TARGET_PLATFORM", "Darwin")
    monkeypatch.setattr(build_app, "MACOS_PORTABLE_DIR", portable_dir)
    monkeypatch.setattr(build_app, "APP_DIST_DIR", app_dist_dir)
    monkeypatch.setattr(build_app, "_codesign_macos_portable", lambda _output_dir: None)
    monkeypatch.setattr(build_app, "_create_macos_launch_script", lambda _output_dir: None)

    assert build_app._finalize_macos_portable_output(app_dist_dir) == portable_dir
    assert (portable_dir / build_app.PROJECT_NAME).exists()
    assert not app_dist_dir.exists()


def test_nuitka_platform_options_use_project_dist_dir_for_standalone(monkeypatch):
    for platform_name in ("Windows", "Linux", "Darwin"):
        monkeypatch.setattr(build_app, "TARGET_PLATFORM", platform_name)

        options = build_app._get_nuitka_platform_options(build_app.STORAGE_MODE_PORTABLE)

        assert f"--output-folder-name={build_app.PROJECT_NAME}" in options


def test_prepare_nuitka_output_paths_cleans_project_build_dir(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    app_build_dir = output_dir / f"{build_app.PROJECT_NAME}.build"
    removed_paths = []

    monkeypatch.delenv("NUITKA_CLEAN_BUILD", raising=False)
    monkeypatch.setattr(build_app, "NUITKA_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_app, "APP_BUILD_DIR", app_build_dir)
    monkeypatch.setattr(build_app, "APP_DIST_DIR", output_dir / f"{build_app.PROJECT_NAME}.dist")
    monkeypatch.setattr(build_app, "APP_BUNDLE_DIR", output_dir / f"{build_app.PROJECT_NAME}.app")
    monkeypatch.setattr(build_app, "MACOS_PORTABLE_DIR", output_dir / build_app.PROJECT_NAME)
    monkeypatch.setattr(build_app, "_remove_existing_path", removed_paths.append)

    build_app._prepare_nuitka_output_paths()

    assert app_build_dir in removed_paths


def test_resolve_nuitka_runtime_target_accepts_default_app_dist(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    app_dist_dir = output_dir / "app.dist"
    app_dist_dir.mkdir(parents=True)

    monkeypatch.setattr(build_app, "TARGET_PLATFORM", "Linux")
    monkeypatch.setattr(build_app, "NUITKA_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_app, "APP_DIST_DIR", output_dir / f"{build_app.PROJECT_NAME}.dist")

    assert build_app._resolve_nuitka_runtime_target_dir(build_app.STORAGE_MODE_PORTABLE) == app_dist_dir


def test_resolve_nuitka_runtime_target_prefers_project_dist(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    project_dist_dir = output_dir / f"{build_app.PROJECT_NAME}.dist"
    project_dist_dir.mkdir(parents=True)

    monkeypatch.setattr(build_app, "TARGET_PLATFORM", "Windows")
    monkeypatch.setattr(build_app, "NUITKA_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(build_app, "APP_DIST_DIR", project_dist_dir)

    assert build_app._resolve_nuitka_runtime_target_dir(build_app.STORAGE_MODE_PORTABLE) == project_dist_dir
