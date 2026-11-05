import itertools

import pytest
from PyQt6.QtWidgets import QApplication

import main


def _fail_dialog(*args, **kwargs):
    raise AssertionError(f"Программа показала сообщение об ошибке: {args[1:]}")


@pytest.fixture(scope="module")
def window():
    app = QApplication.instance() or QApplication([])
    # Модальные окна в автоматическом прогоне заблокировали бы тест - превращаем их в ошибку.
    main.QMessageBox.critical = _fail_dialog
    main.QMessageBox.warning = _fail_dialog
    w = main.GuidanceGUI()
    yield w
    w.close()


def _select(combo, value):
    combo.setCurrentIndex(combo.findData(value))


def test_run_button_requires_pu_motion(window):
    _select(window.method_combo, 1)
    _select(window.graph_combo, 1)
    _select(window.maneuver_combo, 0)
    _select(window.pu_motion_combo, -1)
    assert not window.run_button.isEnabled()
    _select(window.pu_motion_combo, 1)
    assert window.run_button.isEnabled()


@pytest.mark.parametrize("pu, target", list(itertools.product(range(4), range(4))))
def test_all_motion_combinations_plot(window, pu, target):
    window.z2_edit.setText("1500")
    window.V2_edit.setText("100")
    _select(window.pu_motion_combo, pu)
    _select(window.maneuver_combo, target)
    for method in (1, 2, 3):
        _select(window.method_combo, method)
        for graph in range(1, 9):
            _select(window.graph_combo, graph)
            window.run_simulation()
            assert window.results is not None
            assert window.sim.sw_dvizh_pu == pu
            assert window.sim.sw_dvizh_celi == target
