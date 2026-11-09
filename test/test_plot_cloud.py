import sys

from src.utils.plot_cloud import _show_plotter


def test_show_plotter_uses_interactive_updates_on_macos(monkeypatch):
    callbacks = {}

    class Interactor:
        def AddObserver(self, event, callback):
            callbacks[event] = callback

    class Plotter:
        def __init__(self):
            self.iren = type("Renderer", (), {"interactor": Interactor()})()
            self.show_options = None

        def show(self, **options):
            self.show_options = options
            callbacks["ExitEvent"]()

        def update(self, **options):
            raise AssertionError("closed plotter must not be updated")

    monkeypatch.setattr(sys, "platform", "darwin")
    plotter = Plotter()

    _show_plotter(plotter)  # pyright: ignore[reportArgumentType]

    assert plotter.show_options == {"interactive_update": True, "auto_close": False}
