import sys
from datetime import datetime

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Цвета из f_Colours.m
BlueDark = [0.0000, 0.4470, 0.7410]
BlueBright = [0.3010, 0.7450, 0.9330]
OrangeDark = [0.8500, 0.3250, 0.0980]
OrangeBright = [0.9290, 0.6940, 0.1250]
GreenBright = [0.4660, 0.6740, 0.1880]
GreenDark = [0.1660, 0.5740, 0.2880]
Red = [1.0, 0.0, 0.0]
YellowOrange = [0.9, 0.6, 0.0]
Grey = [0.5, 0.5, 0.5]


EPS = 1e-9


def angle_correct(x: float) -> float:
    """Аналог ANGLE_CORRECT.m: приведение угла к диапазону [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Безопасное деление для моментов D -> 0 и V -> 0."""
    if abs(den) < EPS:
        return default
    return num / den


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout(pad=2.5)
        self.fig.subplots_adjust(left=0.10, bottom=0.12, right=0.98, top=0.95)

        # Интерактивный режим: щелчок по графику показывает координаты точки.
        self.click_marker = None
        self.click_annotation = None
        self.coord_callback = None
        self.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # Привязка курсора к ближайшей точке линии графика.
        nearest_x = x
        nearest_y = y
        min_dist = float("inf")

        for line in self.ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()

            if len(xdata) == 0:
                continue

            for px, py in zip(xdata, ydata):
                dist = (px - x) ** 2 + (py - y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    nearest_x = float(px)
                    nearest_y = float(py)

        x = nearest_x
        y = nearest_y

        if self.click_marker is not None:
            self.click_marker.remove()
            self.click_marker = None
        if self.click_annotation is not None:
            self.click_annotation.remove()
            self.click_annotation = None

        self.click_marker = self.ax.plot(x, y, "o", markersize=6, color="black", zorder=10)[0]
        self.click_annotation = self.ax.annotate(
            f"x={x:.3f}\ny={y:.3f}",
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
        if self.coord_callback is not None:
            self.coord_callback(x, y)
        self.draw_idle()

    def clear_click_marker(self):
        if self.click_marker is not None:
            self.click_marker.remove()
            self.click_marker = None
        if self.click_annotation is not None:
            self.click_annotation.remove()
            self.click_annotation = None

    def clear(self):
        self.ax.clear()
        self.click_marker = None
        self.click_annotation = None
        self.draw()


class GuidanceSimulation:
    """
    Перенос логики Guidance_3_fg.m.

    Обозначения сохранены максимально близко к MATLAB-исходнику:
    0 - ПУ / пункт управления;
    1 - объект управления;
    2 - цель.
    """

    def __init__(self):
        # ПУ
        self.z_0_0 = 100.0
        self.x_0_0 = 500.0
        self.V_0_0 = 250.0
        self.psi_0_deg_0 = 90.0

        # Характер движения ПУ (по образцу маневра цели):
        # 0 - неподвижен, 1 - прямолинейно, 2 - постоянное поперечное ускорение,
        # 3 - синусоидальное поперечное ускорение.
        self.sw_dvizh_pu = 1
        self.j_0ppr_m1 = -1.5
        self.T_sin_0 = 120.0
        self.j_0ppr_m2 = -10.0

        # ОУ
        self.V_1_0 = 500.0
        self.psi_1_deg_0 = 90.0

        # Цель
        self.z_2_0 = 5100.0
        self.x_2_0 = 0.0
        self.V_2_0 = 0.0
        self.psi_2_deg_0 = 90.0

        # Характер движения цели:
        # 0 - неподвижна, 1 - прямолинейно, 2 - постоянное поперечное ускорение,
        # 3 - синусоидальное поперечное ускорение.
        # None - определяется по sw_manevr_celi (как в исходной версии программы).
        self.sw_dvizh_celi = None

        # Маневр цели
        self.sw_manevr_celi = 0
        self.j_2ppr_m1 = -1.5
        self.T_sin = 120.0
        self.j_2ppr_m2 = -10.0

        # Метод совмещения
        self.sw_method = 1
        self.K_m1 = 1000.0
        self.K_m2 = 2.0
        self.q31 = 1000.0
        self.q32 = 3000.0
        self.q33 = 10000.0

        # Ограничения и интегрирование
        # Ограничение поперечного ускорения j1 отключено по требованию лабораторной модели.
        self.sw_j_1ppr_gr = False
        self.j_1ppr_gr = None
        self.sw_inerc = False
        self.T_j = 10.0
        self.sw_stop = True
        self.D_por = 30.0
        self.T = 0.01
        self.tk = 12.0
        self.t_linviz = 1.0
        self.sw_linviz = False

        self.results = None

    def run_simulation(self):
        # Неподвижный объект - нулевая скорость независимо от введенного значения.
        V_0_start = 0.0 if self.sw_dvizh_pu == 0 else self.V_0_0
        if self.sw_dvizh_celi is not None:
            self.sw_manevr_celi = max(int(self.sw_dvizh_celi) - 1, 0)
        V_2_start = 0.0 if self.sw_dvizh_celi == 0 else self.V_2_0

        t = np.arange(0.0, self.tk + self.T, self.T)
        K = len(t)

        # ПУ
        z_0 = np.zeros(K); x_0 = np.zeros(K); V_0 = np.zeros(K)
        psi_0 = np.zeros(K); psi_0_deg = np.zeros(K)
        j_0prd = np.zeros(K); j_0ppr = np.zeros(K)
        V_0z = np.zeros(K); V_0x = np.zeros(K)
        a_0 = np.zeros(K); psi_a0 = np.zeros(K); psi_a0_deg = np.zeros(K)
        a_0z = np.zeros(K); a_0x = np.zeros(K)

        # ОУ
        z_1 = np.zeros(K); x_1 = np.zeros(K); V_1 = np.zeros(K)
        psi_1 = np.zeros(K); psi_1_deg = np.zeros(K)
        j_1prd = np.zeros(K); j_1ppr = np.zeros(K)
        V_1z = np.zeros(K); V_1x = np.zeros(K)
        a_1 = np.zeros(K); psi_a1 = np.zeros(K); psi_a1_deg = np.zeros(K)
        a_1z = np.zeros(K); a_1x = np.zeros(K)

        # Цель
        z_2 = np.zeros(K); x_2 = np.zeros(K); V_2 = np.zeros(K)
        psi_2 = np.zeros(K); psi_2_deg = np.zeros(K)
        j_2prd = np.zeros(K); j_2ppr = np.zeros(K)
        V_2z = np.zeros(K); V_2x = np.zeros(K)
        a_2 = np.zeros(K); psi_a2 = np.zeros(K); psi_a2_deg = np.zeros(K)
        a_2z = np.zeros(K); a_2x = np.zeros(K)

        # Кинематика ОУ-цель
        alpha = np.zeros(K); alpha_deg = np.zeros(K)
        phi_1 = np.zeros(K); phi_1_deg = np.zeros(K)
        phi_2 = np.zeros(K); phi_2_deg = np.zeros(K)
        D = np.zeros(K); w = np.zeros(K); w_deg = np.zeros(K)
        V_r = np.zeros(K); V_t = np.zeros(K); V_o = np.zeros(K)
        h = np.zeros(K); S_h = np.zeros(K)

        # Геометрия ПУ-ОУ-цель из Guidance_3_fg.m
        beta_1 = np.zeros(K); beta_1_deg = np.zeros(K)
        beta_2 = np.zeros(K); beta_2_deg = np.zeros(K)
        eps_1 = np.zeros(K); eps_1_deg = np.zeros(K)
        eps_2 = np.zeros(K); eps_2_deg = np.zeros(K)
        delta_eps = np.zeros(K); delta_eps_deg = np.zeros(K)
        D_1 = np.zeros(K); D_2 = np.zeros(K)
        hv = np.zeros(K)
        wv_1 = np.zeros(K); wv_1_deg = np.zeros(K)
        wv_2 = np.zeros(K); wv_2_deg = np.zeros(K)
        delta_wv = np.zeros(K); delta_wv_deg = np.zeros(K)

        flag_stop = False
        flag_break = False
        K_stop = K

        # Начальные условия ПУ
        z_0[0] = self.z_0_0
        x_0[0] = self.x_0_0
        V_0[0] = V_0_start
        psi_0_deg[0] = self.psi_0_deg_0
        psi_0[0] = np.deg2rad(psi_0_deg[0])
        V_0z[0] = V_0[0] * np.sin(psi_0[0])
        V_0x[0] = V_0[0] * np.cos(psi_0[0])
        psi_a0[0] = angle_correct(psi_0[0] + np.arctan2(j_0ppr[0], j_0prd[0]))
        psi_a0_deg[0] = np.rad2deg(psi_a0[0])
        a_0[0] = np.hypot(j_0prd[0], j_0ppr[0])
        a_0z[0] = a_0[0] * np.sin(psi_a0[0])
        a_0x[0] = a_0[0] * np.cos(psi_a0[0])

        # Начальные условия цели
        z_2[0] = self.z_2_0
        x_2[0] = self.x_2_0
        V_2[0] = V_2_start
        psi_2_deg[0] = self.psi_2_deg_0
        psi_2[0] = np.deg2rad(psi_2_deg[0])
        V_2z[0] = V_2[0] * np.sin(psi_2[0])
        V_2x[0] = V_2[0] * np.cos(psi_2[0])
        psi_a2[0] = angle_correct(psi_2[0] + np.arctan2(j_2ppr[0], j_2prd[0]))
        psi_a2_deg[0] = np.rad2deg(psi_a2[0])
        a_2[0] = np.hypot(j_2prd[0], j_2ppr[0])
        a_2z[0] = a_2[0] * np.sin(psi_a2[0])
        a_2x[0] = a_2[0] * np.cos(psi_a2[0])

        # Начальные условия ОУ: как в Guidance_3_fg.m, стартует из точки ПУ
        z_1[0] = z_0[0]
        x_1[0] = x_0[0]
        V_1[0] = self.V_1_0
        psi_1_deg[0] = self.psi_1_deg_0
        psi_1[0] = np.deg2rad(psi_1_deg[0])
        V_1z[0] = V_1[0] * np.sin(psi_1[0])
        V_1x[0] = V_1[0] * np.cos(psi_1[0])
        psi_a1[0] = angle_correct(psi_1[0] + np.arctan2(j_1ppr[0], j_1prd[0]))
        psi_a1_deg[0] = np.rad2deg(psi_a1[0])
        a_1[0] = np.hypot(j_1prd[0], j_1ppr[0])
        a_1z[0] = a_1[0] * np.sin(psi_a1[0])
        a_1x[0] = a_1[0] * np.cos(psi_a1[0])

        self._calc_geometry(
            0, z_0, x_0, V_0, psi_0, z_1, x_1, V_1, psi_1,
            z_2, x_2, V_2, psi_2, alpha, alpha_deg, phi_1, phi_1_deg,
            phi_2, phi_2_deg, D, w, w_deg, V_r, V_t, V_o, h, S_h,
            beta_1, beta_1_deg, beta_2, beta_2_deg, eps_1, eps_1_deg,
            eps_2, eps_2_deg, delta_eps, delta_eps_deg, D_1, D_2,
            hv, wv_1, wv_1_deg, wv_2, wv_2_deg, delta_wv, delta_wv_deg,
        )
        wv_1[0] = 0.0
        wv_1_deg[0] = 0.0
        delta_wv[0] = wv_2[0] - wv_1[0]
        delta_wv_deg[0] = np.rad2deg(delta_wv[0])

        for k in range(1, K):
            # ПУ
            z_0[k] = z_0[k - 1] + V_0z[k - 1] * self.T + a_0z[k - 1] * self.T**2 / 2
            x_0[k] = x_0[k - 1] + V_0x[k - 1] * self.T + a_0x[k - 1] * self.T**2 / 2
            V_0[k] = V_0[k - 1] + j_0prd[k - 1] * self.T
            psi_0[k] = angle_correct(psi_0[k - 1] + safe_div(j_0ppr[k - 1], V_0[k - 1]) * self.T)
            psi_0_deg[k] = np.rad2deg(psi_0[k])
            j_0prd[k] = 0.0
            # Поперечное ускорение ПУ - по тем же законам, что и маневр цели.
            if self.sw_dvizh_pu == 2 and V_0[k] > 0:
                j_0ppr[k] = self.j_0ppr_m1
            elif self.sw_dvizh_pu == 3 and V_0[k] > 0:
                j_0ppr[k] = self.j_0ppr_m2 * np.sin(2 * np.pi * k * self.T / self.T_sin_0 + np.pi / 2)
            else:
                j_0ppr[k] = 0.0
            V_0z[k] = V_0z[k - 1] + a_0z[k - 1] * self.T
            V_0x[k] = V_0x[k - 1] + a_0x[k - 1] * self.T
            a_0[k] = np.hypot(j_0prd[k], j_0ppr[k])
            psi_a0[k] = angle_correct(psi_0[k] + np.arctan2(j_0ppr[k], j_0prd[k]))
            psi_a0_deg[k] = np.rad2deg(psi_a0[k])
            a_0z[k] = a_0[k] * np.sin(psi_a0[k])
            a_0x[k] = a_0[k] * np.cos(psi_a0[k])

            # ОУ
            if not flag_stop:
                z_1[k] = z_1[k - 1] + V_1z[k - 1] * self.T + a_1z[k - 1] * self.T**2 / 2
                x_1[k] = x_1[k - 1] + V_1x[k - 1] * self.T + a_1x[k - 1] * self.T**2 / 2
                V_1[k] = V_1[k - 1] + j_1prd[k - 1] * self.T
                psi_1[k] = angle_correct(psi_1[k - 1] + safe_div(j_1ppr[k - 1], V_1[k - 1]) * self.T)
                psi_1_deg[k] = np.rad2deg(psi_1[k])

                if self.sw_method == 1:
                    j_1ppr[k] = self.K_m1 * delta_eps[k - 1]
                elif self.sw_method == 2:
                    j_1ppr[k] = self.K_m2 * hv[k - 1]
                elif self.sw_method == 3:
                    j_1ppr[k] = safe_div(
                        self.q31 * hv[k - 1] + self.q32 * delta_eps[k - 1] + self.q33 * delta_wv[k - 1],
                        D_1[k - 1] + 1.0,
                    )
                else:
                    j_1ppr[k] = 0.0


                if self.sw_inerc:
                    j_1ppr[k] = self.T / (self.T_j + self.T) * j_1ppr[k] + self.T_j / (self.T_j + self.T) * j_1ppr[k - 1]

                j_1prd[k] = 0.0
                V_1z[k] = V_1z[k - 1] + a_1z[k - 1] * self.T
                V_1x[k] = V_1x[k - 1] + a_1x[k - 1] * self.T
                a_1[k] = np.hypot(j_1prd[k], j_1ppr[k])
                psi_a1[k] = angle_correct(psi_1[k] + np.arctan2(j_1ppr[k], j_1prd[k]))
                psi_a1_deg[k] = np.rad2deg(psi_a1[k])
                a_1z[k] = a_1[k] * np.sin(psi_a1[k])
                a_1x[k] = a_1[k] * np.cos(psi_a1[k])
            else:
                z_1[k] = z_1[k - 1]
                x_1[k] = x_1[k - 1]
                V_1[k] = V_1[k - 1]
                psi_1[k] = psi_1[k - 1]
                psi_1_deg[k] = psi_1_deg[k - 1]
                j_1ppr[k] = 0.0

            # Цель
            z_2[k] = z_2[k - 1] + V_2z[k - 1] * self.T + a_2z[k - 1] * self.T**2 / 2
            x_2[k] = x_2[k - 1] + V_2x[k - 1] * self.T + a_2x[k - 1] * self.T**2 / 2
            V_2[k] = V_2[k - 1] + j_2prd[k - 1] * self.T
            psi_2[k] = angle_correct(psi_2[k - 1] + safe_div(j_2ppr[k - 1], V_2[k - 1]) * self.T)
            psi_2_deg[k] = np.rad2deg(psi_2[k])
            j_2prd[k] = 0.0
            if self.sw_manevr_celi == 0:
                j_2ppr[k] = 0.0
            elif self.sw_manevr_celi == 1:
                j_2ppr[k] = self.j_2ppr_m1
            elif self.sw_manevr_celi == 2:
                j_2ppr[k] = self.j_2ppr_m2 * np.sin(2 * np.pi * k * self.T / self.T_sin + np.pi / 2)
            V_2z[k] = V_2z[k - 1] + a_2z[k - 1] * self.T
            V_2x[k] = V_2x[k - 1] + a_2x[k - 1] * self.T
            a_2[k] = np.hypot(j_2prd[k], j_2ppr[k])
            psi_a2[k] = angle_correct(psi_2[k] + np.arctan2(j_2ppr[k], j_2prd[k]))
            psi_a2_deg[k] = np.rad2deg(psi_a2[k])
            a_2z[k] = a_2[k] * np.sin(psi_a2[k])
            a_2x[k] = a_2[k] * np.cos(psi_a2[k])

            self._calc_geometry(
                k, z_0, x_0, V_0, psi_0, z_1, x_1, V_1, psi_1,
                z_2, x_2, V_2, psi_2, alpha, alpha_deg, phi_1, phi_1_deg,
                phi_2, phi_2_deg, D, w, w_deg, V_r, V_t, V_o, h, S_h,
                beta_1, beta_1_deg, beta_2, beta_2_deg, eps_1, eps_1_deg,
                eps_2, eps_2_deg, delta_eps, delta_eps_deg, D_1, D_2,
                hv, wv_1, wv_1_deg, wv_2, wv_2_deg, delta_wv, delta_wv_deg,
            )

            if self.sw_stop and not flag_stop:
                if V_r[k] < 0:
                    flag_stop = True
                    flag_break = True
                    K_stop = k
                elif D[k] <= self.D_por:
                    flag_stop = True
                    K_stop = k

        self.results = locals()
        self.results.update({"t": t, "K_stop": K_stop, "flag_stop": flag_stop, "flag_break": flag_break})
        return self.results

    def _calc_geometry(
        self, k, z_0, x_0, V_0, psi_0, z_1, x_1, V_1, psi_1, z_2, x_2, V_2, psi_2,
        alpha, alpha_deg, phi_1, phi_1_deg, phi_2, phi_2_deg, D, w, w_deg, V_r, V_t, V_o, h, S_h,
        beta_1, beta_1_deg, beta_2, beta_2_deg, eps_1, eps_1_deg, eps_2, eps_2_deg,
        delta_eps, delta_eps_deg, D_1, D_2, hv, wv_1, wv_1_deg, wv_2, wv_2_deg, delta_wv, delta_wv_deg,
    ):
        alpha[k] = np.arctan2(x_2[k] - x_1[k], z_2[k] - z_1[k])
        alpha_deg[k] = np.rad2deg(alpha[k])

        phi_1[k] = angle_correct(np.pi / 2 - psi_1[k] - alpha[k])
        phi_1_deg[k] = np.rad2deg(phi_1[k])
        phi_2[k] = angle_correct(np.pi / 2 - psi_2[k] - alpha[k])
        phi_2_deg[k] = np.rad2deg(phi_2[k])

        D[k] = np.hypot(z_2[k] - z_1[k], x_2[k] - x_1[k])
        w[k] = safe_div(V_1[k] * np.sin(phi_1[k]) - V_2[k] * np.sin(phi_2[k]), D[k])
        w_deg[k] = np.rad2deg(w[k])
        V_r[k] = V_1[k] * np.cos(phi_1[k]) - V_2[k] * np.cos(phi_2[k])
        V_t[k] = V_1[k] * np.sin(phi_1[k]) - V_2[k] * np.sin(phi_2[k])
        V_o[k] = np.hypot(V_r[k], V_t[k])
        h[k] = safe_div(D[k] ** 2 * w[k], V_o[k])
        if k > 0:
            S_h[k] = S_h[k - 1] + abs((h[k - 1] + h[k]) / 2 * self.T)

        beta_1[k] = np.arctan2(x_0[k] - x_1[k], z_1[k] - z_0[k])
        beta_1_deg[k] = np.rad2deg(beta_1[k])
        beta_2[k] = np.arctan2(x_0[k] - x_2[k], z_2[k] - z_0[k])
        beta_2_deg[k] = np.rad2deg(beta_2[k])

        eps_1[k] = beta_1[k] + np.pi / 2 - psi_0[k]
        eps_1[k] = angle_correct(eps_1[k])
        eps_1_deg[k] = np.rad2deg(eps_1[k])
        eps_2[k] = beta_2[k] + np.pi / 2 - psi_0[k]
        eps_2[k] = angle_correct(eps_2[k])
        eps_2_deg[k] = np.rad2deg(eps_2[k])

        delta_eps[k] = angle_correct(eps_2[k] - eps_1[k])
        delta_eps_deg[k] = np.rad2deg(delta_eps[k])

        D_1[k] = np.hypot(z_0[k] - z_1[k], x_0[k] - x_1[k])
        D_2[k] = np.hypot(z_0[k] - z_2[k], x_0[k] - x_2[k])
        hv[k] = D_1[k] * np.sin(delta_eps[k])

        wv_1[k] = safe_div(V_0[k] * np.sin(eps_1[k]), D_1[k])
        wv_1_deg[k] = np.rad2deg(wv_1[k])
        # Угловая скорость ЛВ ПУ-цель: вклад движения ПУ и собственного движения цели.
        # При неподвижной цели (V_2 = 0) совпадает с исходной формулой.
        wv_2[k] = safe_div(V_0[k] * np.sin(eps_2[k]) - V_2[k] * np.cos(beta_2[k] - psi_2[k]), D_2[k])
        wv_2_deg[k] = np.rad2deg(wv_2[k])
        delta_wv[k] = wv_2[k] - wv_1[k]
        delta_wv_deg[k] = np.rad2deg(delta_wv[k])



class GuidanceGUI(QMainWindow):
    """GUI в стиле исходного MATLAB/Guidance интерфейса: график слева, установки справа."""

    def __init__(self):
        super().__init__()
        self.sim = GuidanceSimulation()
        self.results = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Моделирование метода совмещения")
        self.resize(1500, 820)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(18)

        # Левая область: заголовок, график, кнопки
        graph_panel = QWidget()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(8)

        title_graph = QLabel("Область графика")
        title_graph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_graph.setStyleSheet("font-size: 18px; font-weight: bold;")
        graph_layout.addWidget(title_graph)

        self.canvas = MplCanvas(self, width=10.8, height=7.2, dpi=100)
        self.canvas.coord_callback = self._show_clicked_coordinates
        graph_layout.addWidget(self.canvas, stretch=1)

        self.coord_label = QLabel("Координаты точки: щелкните по графику")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coord_label.setStyleSheet("font-size: 11px; color: #333333;")
        graph_layout.addWidget(self.coord_label)

        self.prepare_empty_axes()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(55)
        self.run_button = QPushButton("Построить")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setEnabled(False)
        self.clear_button = QPushButton("Очистить")
        self.clear_button.clicked.connect(self.clear_plot)
        self.clear_button.setEnabled(False)
        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)
        for btn in (self.run_button, self.clear_button, self.save_button):
            btn.setMinimumHeight(42)
            btn.setMinimumWidth(210)
        button_layout.addStretch(1)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch(1)
        graph_layout.addLayout(button_layout)

        # Правая область: установки как в MATLAB GUI
        settings_panel = QWidget()
        settings_panel.setFixedWidth(405)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        title_settings = QLabel("Установки")
        title_settings.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_settings.setStyleSheet("font-size: 18px; font-weight: bold;")
        settings_layout.addWidget(title_settings)

        duration_box = self._group("Продолжительность наведения")
        duration_grid = QGridLayout(duration_box)
        self.tk_edit = QLineEdit("12")
        self.tk_edit.setFixedWidth(58)
        duration_grid.addWidget(self.tk_edit, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        duration_grid.addWidget(QLabel("с"), 0, 1)
        self.linviz_check = QCheckBox("ЛВ")
        duration_grid.addWidget(self.linviz_check, 0, 3, alignment=Qt.AlignmentFlag.AlignRight)
        duration_grid.setColumnStretch(2, 1)
        settings_layout.addWidget(duration_box)

        settings_layout.addWidget(self._separator())

        pu_coord_box = self._group("Начальные координаты ПУ")
        pu_coord_grid = QGridLayout(pu_coord_box)
        self.z0_edit = self._small_edit("100")
        self.x0_edit = self._small_edit("500")
        pu_coord_grid.addWidget(QLabel("l0, м"), 0, 0)
        pu_coord_grid.addWidget(self.z0_edit, 0, 1)
        pu_coord_grid.addWidget(QLabel("h0, м"), 0, 2)
        pu_coord_grid.addWidget(self.x0_edit, 0, 3)
        settings_layout.addWidget(pu_coord_box)

        pu_row = QHBoxLayout()
        pu_speed_box = self._group("Скорость ПУ")
        pu_speed_grid = QGridLayout(pu_speed_box)
        self.V0_edit = self._small_edit("250")
        pu_speed_grid.addWidget(QLabel("V0, м/с"), 0, 0)
        pu_speed_grid.addWidget(self.V0_edit, 0, 1)
        pu_course_box = self._group("Начальный ПУ")
        pu_course_grid = QGridLayout(pu_course_box)
        self.psi0_edit = self._small_edit("90")
        pu_course_grid.addWidget(QLabel("Psi0, град"), 0, 0)
        pu_course_grid.addWidget(self.psi0_edit, 0, 1)
        pu_row.addWidget(pu_speed_box)
        pu_row.addWidget(pu_course_box)
        settings_layout.addLayout(pu_row)

        settings_layout.addWidget(self._separator())

        missile_coord_box = self._group("Начальные координаты ракеты")
        missile_coord_grid = QGridLayout(missile_coord_box)
        self.z1_edit = self._small_edit("")
        self.x1_edit = self._small_edit("")
        self.z1_edit.setPlaceholderText("из ПУ")
        self.x1_edit.setPlaceholderText("из ПУ")
        missile_coord_grid.addWidget(QLabel("l1, м"), 0, 0)
        missile_coord_grid.addWidget(self.z1_edit, 0, 1)
        missile_coord_grid.addWidget(QLabel("h1, м"), 0, 2)
        missile_coord_grid.addWidget(self.x1_edit, 0, 3)
        settings_layout.addWidget(missile_coord_box)

        missile_row = QHBoxLayout()
        missile_speed_box = self._group("Скорость ракеты")
        missile_speed_grid = QGridLayout(missile_speed_box)
        self.V1_edit = self._small_edit("500")
        missile_speed_grid.addWidget(QLabel("V1, м/с"), 0, 0)
        missile_speed_grid.addWidget(self.V1_edit, 0, 1)
        missile_course_box = self._group("Начальный курс ракеты")
        missile_course_grid = QGridLayout(missile_course_box)
        self.psi1_edit = self._small_edit("90")
        missile_course_grid.addWidget(QLabel("Psi1, град"), 0, 0)
        missile_course_grid.addWidget(self.psi1_edit, 0, 1)
        missile_row.addWidget(missile_speed_box)
        missile_row.addWidget(missile_course_box)
        settings_layout.addLayout(missile_row)

        settings_layout.addWidget(self._separator())

        target_coord_box = self._group("Начальные координаты цели")
        target_coord_grid = QGridLayout(target_coord_box)
        self.z2_edit = self._small_edit("5100")
        self.x2_edit = self._small_edit("0")
        target_coord_grid.addWidget(QLabel("l2, м"), 0, 0)
        target_coord_grid.addWidget(self.z2_edit, 0, 1)
        target_coord_grid.addWidget(QLabel("h2, м"), 0, 2)
        target_coord_grid.addWidget(self.x2_edit, 0, 3)
        settings_layout.addWidget(target_coord_box)

        target_row = QHBoxLayout()
        target_speed_box = self._group("Скорость цели")
        target_speed_grid = QGridLayout(target_speed_box)
        self.V2_edit = self._small_edit("0")
        target_speed_grid.addWidget(QLabel("V2, м/с"), 0, 0)
        target_speed_grid.addWidget(self.V2_edit, 0, 1)
        target_course_box = self._group("Начальный курс цели")
        target_course_grid = QGridLayout(target_course_box)
        self.psi2_edit = self._small_edit("90")
        target_course_grid.addWidget(QLabel("Psi2, град"), 0, 0)
        target_course_grid.addWidget(self.psi2_edit, 0, 1)
        target_row.addWidget(target_speed_box)
        target_row.addWidget(target_course_box)
        settings_layout.addLayout(target_row)

        settings_layout.addWidget(self._separator())

        coeff_row = QHBoxLayout()
        k1_box = self._group("Коэффициент K1")
        k1_grid = QGridLayout(k1_box)
        self.K1_edit = self._small_edit("1000")
        k1_grid.addWidget(self.K1_edit, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        k2_box = self._group("Коэффициент K2")
        k2_grid = QGridLayout(k2_box)
        self.K2_edit = self._small_edit("5")
        k2_grid.addWidget(self.K2_edit, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        coeff_row.addWidget(k1_box)
        coeff_row.addWidget(k2_box)
        settings_layout.addLayout(coeff_row)

        q_box = self._group("Коэффициенты K31, K32, K33")
        q_grid = QGridLayout(q_box)
        self.q31_edit = self._small_edit("1000")
        self.q32_edit = self._small_edit("1000")
        self.q33_edit = self._small_edit("1000")
        q_grid.addWidget(self.q31_edit, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        q_grid.addWidget(self.q32_edit, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        q_grid.addWidget(self.q33_edit, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(q_box)

        inertia_row = QHBoxLayout()
        inertia_box = self._group("Инерционность управления")
        inertia_grid = QGridLayout(inertia_box)
        self.inerc_check = QCheckBox("включить")
        self.inerc_check.setChecked(False)
        inertia_grid.addWidget(self.inerc_check, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        ti_box = self._group("Постоянная времени инерции")
        ti_grid = QGridLayout(ti_box)
        self.Ti_edit = self._small_edit("5")
        ti_grid.addWidget(QLabel("Tи, с"), 0, 0)
        ti_grid.addWidget(self.Ti_edit, 0, 1)

        inertia_row.addWidget(inertia_box)
        inertia_row.addWidget(ti_box)
        settings_layout.addLayout(inertia_row)

        settings_layout.addWidget(self._separator())

        self.pu_motion_combo = QComboBox()
        self.pu_motion_combo.addItem("--- Выберите характер движения ПУ ---", -1)
        self.pu_motion_combo.addItem("ПУ неподвижен", 0)
        self.pu_motion_combo.addItem("ПУ: прямолинейное движение", 1)
        self.pu_motion_combo.addItem("ПУ: постоянное поперечное ускорение", 2)
        self.pu_motion_combo.addItem("ПУ: синусоидальное поперечное ускорение", 3)
        self.pu_motion_combo.currentIndexChanged.connect(self._update_run_button_state)
        settings_layout.addWidget(self.pu_motion_combo)

        self.maneuver_combo = QComboBox()
        self.maneuver_combo.addItem("--- Выберите характер движения цели ---", -1)
        self.maneuver_combo.addItem("Цель неподвижна", 0)
        self.maneuver_combo.addItem("Цель: прямолинейное движение", 1)
        self.maneuver_combo.addItem("Цель: постоянное поперечное ускорение", 2)
        self.maneuver_combo.addItem("Цель: синусоидальное поперечное ускорение", 3)
        self.maneuver_combo.currentIndexChanged.connect(self._update_run_button_state)
        settings_layout.addWidget(self.maneuver_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItem("--- Выберите метод наведения ---", 0)
        self.method_combo.addItem("Метод совмещения", 1)
        self.method_combo.addItem("Метод совмещения по промаху", 2)
        self.method_combo.addItem("Метод совмещения оптимальный", 3)
        self.method_combo.currentIndexChanged.connect(self._update_run_button_state)
        settings_layout.addWidget(self.method_combo)

        self.graph_combo = QComboBox()
        self.graph_combo.addItem("--- Выберите вид графика ---", 0)
        self.graph_combo.addItem("Траектории", 1)
        self.graph_combo.addItem("Углы визирования eps", 2)
        self.graph_combo.addItem("Линейный промах hv", 3)
        self.graph_combo.addItem("Угловые скорости wv", 4)
        self.graph_combo.addItem("Поперечное ускорение ракеты", 5)
        self.graph_combo.addItem("Дальности D, D1, D2", 6)
        self.graph_combo.addItem("Параметр h и интеграл S_h", 7)
        self.graph_combo.addItem("Углы beta", 8)
        self.graph_combo.currentIndexChanged.connect(self._update_run_button_state)
        settings_layout.addWidget(self.graph_combo)
        settings_layout.addStretch(1)

        main_layout.addWidget(graph_panel, stretch=1)
        main_layout.addWidget(settings_panel)

    def _group(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-size: 10px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        return box

    def _separator(self) -> QLabel:
        label = QLabel("====================================================")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-family: monospace; color: #333;")
        return label

    def _small_edit(self, text: str) -> QLineEdit:
        edit = QLineEdit(text)
        edit.setFixedWidth(58)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return edit

    def _update_run_button_state(self):
        method_selected = bool(self.method_combo.currentData())
        graph_selected = bool(self.graph_combo.currentData())
        maneuver_selected = self.maneuver_combo.currentData() is not None and int(self.maneuver_combo.currentData()) >= 0
        pu_selected = self.pu_motion_combo.currentData() is not None and int(self.pu_motion_combo.currentData()) >= 0
        self.run_button.setEnabled(method_selected and graph_selected and maneuver_selected and pu_selected)

    def prepare_empty_axes(self):
        self.canvas.ax.clear()
        self.canvas.ax.set_xlim(0, 6000)
        self.canvas.ax.set_ylim(0, 4000)
        self.canvas.ax.set_xlabel("L, Pj")
        self.canvas.ax.set_ylabel("H, Pj")
        self.canvas.ax.grid(True, alpha=0.35)
        self.canvas.draw()

    def get_parameters_from_ui(self):
        try:
            self.sim.z_0_0 = float(self.z0_edit.text())
            self.sim.x_0_0 = float(self.x0_edit.text())
            self.sim.V_0_0 = float(self.V0_edit.text())
            self.sim.psi_0_deg_0 = float(self.psi0_edit.text())
            self.sim.V_1_0 = float(self.V1_edit.text())
            self.sim.psi_1_deg_0 = float(self.psi1_edit.text())
            self.sim.z_2_0 = float(self.z2_edit.text())
            self.sim.x_2_0 = float(self.x2_edit.text())
            self.sim.V_2_0 = float(self.V2_edit.text())
            self.sim.psi_2_deg_0 = float(self.psi2_edit.text())
            self.sim.sw_method = int(self.method_combo.currentData())
            self.sim.K_m1 = float(self.K1_edit.text())
            self.sim.K_m2 = float(self.K2_edit.text())
            self.sim.q31 = float(self.q31_edit.text())
            self.sim.q32 = float(self.q32_edit.text())
            self.sim.q33 = float(self.q33_edit.text())
            self.sim.tk = float(self.tk_edit.text())
            self.sim.sw_linviz = self.linviz_check.isChecked()
            self.sim.sw_dvizh_celi = int(self.maneuver_combo.currentData())
            self.sim.sw_dvizh_pu = int(self.pu_motion_combo.currentData())
            self.sim.sw_inerc = self.inerc_check.isChecked()
            self.sim.T_j = float(self.Ti_edit.text())
            # Ограничение на j1 убрано: команда поперечного ускорения не клипируется.
            self.sim.sw_j_1ppr_gr = False

            # В GUI оставлены поля координат ракеты как в исходнике. В расчетной модели
            # по умолчанию ракета стартует из точки ПУ; непустые значения перезаписывают ПУ.
            if self.z1_edit.text().strip() and self.x1_edit.text().strip():
                self.sim.z_0_0 = float(self.z1_edit.text())
                self.sim.x_0_0 = float(self.x1_edit.text())

            if self.sim.T <= 0 or self.sim.tk <= 0:
                raise ValueError("T и продолжительность должны быть положительными")
            if self.sim.T_j < 0:
                raise ValueError("Tи должна быть неотрицательной")
            if self.sim.sw_dvizh_celi < 0:
                raise ValueError("выберите характер движения цели")
            if self.sim.sw_dvizh_pu < 0:
                raise ValueError("выберите характер движения ПУ")
            if not self.sim.sw_method:
                raise ValueError("выберите метод наведения")
            if not self.graph_combo.currentData():
                raise ValueError("выберите вид графика")
            return True
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Неверные установки: {e}")
            return False

    def run_simulation(self):
        if not self.get_parameters_from_ui():
            return
        self.run_button.setEnabled(False)
        QApplication.processEvents()
        try:
            self.results = self.sim.run_simulation()
            self.plot_results()
            self.clear_button.setEnabled(True)
            self.save_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при моделировании:\n{e}")
        finally:
            self._update_run_button_state()

    def _show_clicked_coordinates(self, x: float, y: float):
        self.coord_label.setText(f"Координаты точки: X = {x:.3f}; Y = {y:.3f}")

    def _auto_fit_axes(self, ax, x_values, y_values, min_padding: float = 1.0):
        """Автоматически подбирает границы графика по всем отображаемым точкам."""
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return

        x = x[mask]
        y = y[mask]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))

        dx = xmax - xmin
        dy = ymax - ymin
        if dx < EPS:
            dx = max(abs(xmax) * 0.1, min_padding)
            xmin -= dx
            xmax += dx
        else:
            pad_x = max(dx * 0.08, min_padding)
            xmin -= pad_x
            xmax += pad_x

        if dy < EPS:
            dy = max(abs(ymax) * 0.1, min_padding)
            ymin -= dy
            ymax += dy
        else:
            pad_y = max(dy * 0.08, min_padding)
            ymin -= pad_y
            ymax += pad_y

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    def plot_results(self):
        if self.results is None:
            return
        self.canvas.clear()
        ax = self.canvas.ax
        r = self.results
        t = r["t"]
        stop = max(1, int(r["K_stop"]))
        graph_type = int(self.graph_combo.currentData())
        ax.grid(True, alpha=0.35)

        if graph_type == 1:
            ax.set_xlabel("L, Pj")
            ax.set_ylabel("H, Pj")
            ax.set_aspect("auto")
            ax.plot(r["z_0"][:stop], r["x_0"][:stop], color=Grey, linewidth=1.2, label="ПУ")
            ax.plot(r["z_2"][:stop], r["x_2"][:stop], color=Red, linewidth=1.5, label="Цель")
            ax.plot(r["z_1"][:stop], r["x_1"][:stop], color=BlueDark, linewidth=1.5, label="Ракета")
            ax.plot(r["z_0"][0], r["x_0"][0], "*", color=Grey, markersize=10)
            ax.plot(r["z_2"][0], r["x_2"][0], "*", color=Red, markersize=10)
            ax.plot(r["z_1"][0], r["x_1"][0], "*", color=BlueDark, markersize=10)
            ax.text(r["z_0"][0], r["x_0"][0], "ПУ", va="top", ha="center")
            ax.text(r["z_2"][0], r["x_2"][0], "Ц", va="top", ha="center")
            ax.text(r["z_1"][0], r["x_1"][0], "Р", va="bottom", ha="center")
            if r["flag_stop"]:
                ax.plot(r["z_1"][stop - 1], r["x_1"][stop - 1], "*", color=YellowOrange if not r["flag_break"] else Red, markersize=14)
            if self.sim.sw_linviz:
                step = max(1, int(self.sim.t_linviz / self.sim.T))
                for n in range(0, stop, step):
                    ax.plot([r["z_1"][n], r["z_2"][n]], [r["x_1"][n], r["x_2"][n]], color=BlueBright, linewidth=0.8)

            # Автомасштабирование трассы: учитываем ПУ, цель, ракету и конечную точку.
            x_values = np.concatenate([r["z_0"][:stop], r["z_1"][:stop], r["z_2"][:stop]])
            y_values = np.concatenate([r["x_0"][:stop], r["x_1"][:stop], r["x_2"][:stop]])
            self._auto_fit_axes(ax, x_values, y_values, min_padding=50.0)

        elif graph_type == 2:
            ax.set_xlabel("t, с"); ax.set_ylabel("eps, град")
            ax.plot(t[:stop], r["eps_1_deg"][:stop], color=BlueDark, linewidth=1.5, label="eps1 ПУ-ракета")
            ax.plot(t[:stop], r["eps_2_deg"][:stop], color=Red, linewidth=1.5, label="eps2 ПУ-цель")
            ax.plot(t[:stop], r["delta_eps_deg"][:stop], color=GreenDark, linewidth=1.2, label="delta_eps")

        elif graph_type == 3:
            ax.set_xlabel("t, с"); ax.set_ylabel("hv, м")
            ax.plot(t[:stop], r["hv"][:stop], color=BlueDark, linewidth=1.5, label="hv")
            ax.plot(t[:stop], np.zeros(stop), color=Red, linewidth=1)

        elif graph_type == 4:
            ax.set_xlabel("t, с"); ax.set_ylabel("wv, град/с")
            ax.plot(t[:stop], r["wv_1_deg"][:stop], color=BlueDark, linewidth=1.5, label="wv1")
            ax.plot(t[:stop], r["wv_2_deg"][:stop], color=Red, linewidth=1.5, label="wv2")
            ax.plot(t[:stop], r["delta_wv_deg"][:stop], color=GreenDark, linewidth=1.2, label="delta_wv")

        elif graph_type == 5:
            ax.set_xlabel("t, с"); ax.set_ylabel("j1ппр, м/с²")
            ax.plot(t[:stop], r["j_1ppr"][:stop], color=BlueDark, linewidth=1.5, label="ракета")
            ax.plot(t, r["j_2ppr"], color=Red, linewidth=1.2, label="цель")

        elif graph_type == 6:
            ax.set_xlabel("t, с"); ax.set_ylabel("D, м")
            ax.plot(t[:stop], r["D"][:stop], color=BlueDark, linewidth=1.5, label="ракета-цель")
            ax.plot(t[:stop], r["D_1"][:stop], color=GreenDark, linewidth=1.2, label="ПУ-ракета")
            ax.plot(t[:stop], r["D_2"][:stop], color=Red, linewidth=1.2, label="ПУ-цель")

        elif graph_type == 7:
            ax.set_xlabel("t, с"); ax.set_ylabel("h")
            ax.plot(t[:stop], r["h"][:stop], color=BlueDark, linewidth=1.5, label="h")
            ax.plot(t[:stop], r["S_h"][:stop], color=OrangeDark, linewidth=1.2, label="S_h")

        elif graph_type == 8:
            ax.set_xlabel("t, с"); ax.set_ylabel("beta, град")
            ax.plot(t[:stop], r["beta_1_deg"][:stop], color=BlueDark, linewidth=1.5, label="beta1 ПУ-ракета")
            ax.plot(t[:stop], r["beta_2_deg"][:stop], color=Red, linewidth=1.5, label="beta2 ПУ-цель")

        if graph_type != 1:
            ax.relim()
            ax.autoscale_view()
            ax.margins(x=0.10, y=0.18)

        self.coord_label.setText("Координаты точки: щелкните по графику")
        ax.legend(loc="best")
        self.canvas.draw()

    def clear_plot(self):
        self.results = None
        self.prepare_empty_axes()
        self.coord_label.setText("Координаты точки: щелкните по графику")
        self.clear_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def save_data(self):
        if self.results is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить данные", "", "CSV files (*.csv);;All files (*.*)")
        if not file_path:
            return
        try:
            r = self.results
            fields = ["t", "z_0", "x_0", "z_1", "x_1", "z_2", "x_2", "D", "hv", "delta_eps_deg", "delta_wv_deg", "j_1ppr"]
            data = np.column_stack([r[f] for f in fields])
            header = ",".join(fields) + f"\n# saved {datetime.now():%d-%m-%Y %H:%M}"
            np.savetxt(file_path, data, delimiter=",", header=header, comments="# ")
            QMessageBox.information(self, "Сохранение", f"Данные сохранены в:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении:\n{e}")


def main():
    app = QApplication(sys.argv)
    window = GuidanceGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
