"""Modern light-theme stylesheet for the logo2svg GUI.

Colour palette based on Tailwind's gray/indigo scales for a clean,
readable light appearance.  Every GUI module imports tokens from here
so the look-and-feel stays consistent.
"""

from __future__ import annotations

# ── Colour Palette ──────────────────────────────────────────────────

BG           = "#f5f5f7"     # main window background
SURFACE      = "#ffffff"     # sidebar / header / card surface
CARD         = "#edf0f4"     # input backgrounds, secondary cards
CARD_HOVER   = "#e2e5ea"     # hovered card
BORDER       = "#d1d5db"     # default border
BORDER_LIGHT = "#b0b8c4"     # active / hovered border

ACCENT       = "#4f46e5"     # primary accent (indigo-600)
ACCENT_HOVER = "#6366f1"     # lighter accent on hover (indigo-500)
ACCENT_MUTED = "#eef2ff"     # very subtle accent bg

TEXT         = "#111827"     # primary text (gray-900)
TEXT_SEC     = "#6b7280"     # secondary text (gray-500)
TEXT_MUTED   = "#9ca3af"     # muted / placeholder (gray-400)

SUCCESS      = "#16a34a"
ERROR        = "#dc2626"
WARNING      = "#ca8a04"

# ── Geometry Tokens ─────────────────────────────────────────────────

RADIUS    = "6px"
RADIUS_SM = "4px"
RADIUS_LG = "8px"

# ── Full QSS ───────────────────────────────────────────────────────

STYLESHEET = f"""

/* ================================================================
   GLOBAL
   ================================================================ */

* {{
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    outline: none;
}}

QMainWindow {{
    background-color: {BG};
    color: {TEXT};
}}

QDialog {{
    background-color: {SURFACE};
    color: {TEXT};
}}

/* ================================================================
   MENU BAR
   ================================================================ */

QMenuBar {{
    background-color: {SURFACE};
    border-bottom: 1px solid {BORDER};
    padding: 2px 0px;
    font-size: 13px;
    color: {TEXT_SEC};
}}

QMenuBar::item {{
    padding: 5px 10px;
    background: transparent;
    border-radius: {RADIUS_SM};
    color: {TEXT_SEC};
}}

QMenuBar::item:selected {{
    background-color: {CARD_HOVER};
    color: {TEXT};
}}

QMenu {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 30px 6px 14px;
    border-radius: {RADIUS_SM};
    color: {TEXT};
}}

QMenu::item:selected {{
    background-color: {ACCENT};
    color: #ffffff;
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ================================================================
   SCROLL BARS
   ================================================================ */

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {BORDER_LIGHT};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    height: 0px;
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {BORDER_LIGHT};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    width: 0px;
    background: transparent;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

/* ================================================================
   BUTTONS
   ================================================================ */

QPushButton {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 500;
    color: {TEXT};
}}

QPushButton:hover {{
    background-color: {CARD_HOVER};
    border-color: {BORDER_LIGHT};
}}

QPushButton:pressed {{
    background-color: {BORDER};
}}

QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: {BG};
    border-color: {BORDER};
}}

/* -- Primary / accent buttons -- */

QPushButton[cssClass="primary"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}

QPushButton[cssClass="primary"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}

QPushButton[cssClass="primary"]:pressed {{
    background-color: #4338ca;
    border-color: #4338ca;
}}

QPushButton[cssClass="primary"]:disabled {{
    background-color: {BORDER};
    border-color: {BORDER};
    color: {TEXT_MUTED};
}}

/* -- Icon buttons (small, square) -- */

QPushButton[cssClass="icon"] {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM};
    padding: 2px;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    font-size: 14px;
}}

QPushButton[cssClass="icon"]:hover {{
    background-color: {CARD_HOVER};
    border-color: {BORDER};
}}

QPushButton[cssClass="icon"]:checked {{
    background-color: {ACCENT_MUTED};
    border-color: {ACCENT};
    color: {ACCENT_HOVER};
}}

/* -- Danger buttons (delete, etc.) -- */

QPushButton[cssClass="danger"] {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM};
    padding: 2px;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    font-size: 13px;
    color: {TEXT_MUTED};
}}

QPushButton[cssClass="danger"]:hover {{
    background-color: rgba(239, 68, 68, 0.15);
    border-color: {ERROR};
    color: {ERROR};
}}

/* ================================================================
   SPIN BOXES
   ================================================================ */

QSpinBox, QDoubleSpinBox {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    padding: 4px 8px;
    color: {TEXT};
    min-height: 24px;
    font-size: 13px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border: none;
    background: transparent;
    border-top-right-radius: {RADIUS};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border: none;
    background: transparent;
    border-bottom-right-radius: {RADIUS};
}}

/* ================================================================
   LINE EDITS
   ================================================================ */

QLineEdit {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    padding: 6px 10px;
    color: {TEXT};
    font-size: 13px;
}}

QLineEdit:focus {{
    border-color: {ACCENT};
}}

/* ================================================================
   CHECKBOXES
   ================================================================ */

QCheckBox {{
    spacing: 8px;
    color: {TEXT};
    font-size: 13px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {BORDER_LIGHT};
    border-radius: {RADIUS_SM};
    background: {CARD};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {TEXT_SEC};
}}

/* ================================================================
   STATUS BAR
   ================================================================ */

QStatusBar {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_SEC};
    font-size: 12px;
    padding: 2px 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* ================================================================
   PROGRESS BAR
   ================================================================ */

QProgressBar {{
    background-color: {CARD};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ================================================================
   SPLITTER
   ================================================================ */

QSplitter {{
    background: transparent;
}}

QSplitter::handle {{
    background: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ================================================================
   TOOLTIPS
   ================================================================ */

QToolTip {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 6px 10px;
    color: {TEXT};
    font-size: 12px;
}}

/* ================================================================
   GROUP BOX (settings dialog)
   ================================================================ */

QGroupBox {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    margin-top: 20px;
    padding: 20px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {TEXT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: {TEXT_SEC};
    font-size: 12px;
    font-weight: 600;
}}

/* ================================================================
   DIALOG BUTTONS
   ================================================================ */

QDialogButtonBox QPushButton {{
    min-width: 80px;
    padding: 8px 20px;
}}

/* ================================================================
   LABELS
   ================================================================ */

QLabel {{
    background: transparent;
    color: {TEXT};
    font-size: 13px;
}}

QLabel#sectionHeader {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT_MUTED};
    padding: 0px;
    margin: 0px;
}}

/* ================================================================
   APP-SPECIFIC: left sidebar
   ================================================================ */

QFrame#leftSidebar {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}

/* ================================================================
   APP-SPECIFIC: layer card
   ================================================================ */

QFrame[cssClass="layerCard"] {{
    background-color: {CARD};
    border: 1px solid transparent;
    border-radius: {RADIUS};
}}

QFrame[cssClass="layerCard"]:hover {{
    background-color: {CARD_HOVER};
    border-color: {BORDER};
}}

/* ================================================================
   APP-SPECIFIC: header bar
   ================================================================ */

QFrame#headerBar {{
    background-color: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}

/* ================================================================
   APP-SPECIFIC: layer panel (in splitter)
   ================================================================ */

QWidget#layerPanel {{
    background-color: {SURFACE};
}}
"""
