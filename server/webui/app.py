"""
DeepX OCR Server Web UI
严格基于 PP-OCRv5_Online_demo UI 布局，适配 DeepX OCR Server API
"""

import atexit
import base64
import io
import json
import os
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

# Get absolute path for static files
BASE_DIR = Path(__file__).parent.resolve()

# ============ API 配置 (修改为 DeepX OCR Server) ============
API_URL = os.environ.get("API_URL", "http://localhost:8080/ocr")
API_BASE = os.environ.get("API_BASE", "http://localhost:8080")
TOKEN = os.environ.get("API_TOKEN", "deepx_token")

TITLE = "DeepX OCR Server Demo"

TEMP_DIR = tempfile.TemporaryDirectory()
atexit.register(TEMP_DIR.cleanup)

# ============ 主题配置 (与原项目完全一致) ============
paddle_theme = gr.themes.Soft(
    font=(gr.themes.GoogleFont("Roboto"), "Open Sans", "Arial", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("Fira Code"), "monospace"),
    primary_hue=gr.themes.Color(
        c50="#e8eafc",
        c100="#c5c9f7",
        c200="#a1a7f2",
        c300="#7d85ed",
        c400="#5963e8",
        c500="#2932e1",  # 主色调
        c600="#242bb4",
        c700="#1e2487",
        c800="#181d5a",
        c900="#12162d",
        c950="#0c0f1d",
    ),
)

MAX_NUM_PAGES = 10
TMP_DELETE_TIME = 900
THREAD_WAKEUP_TIME = 600

# ============ CSS 样式 (与原项目完全一致) ============
CSS = """
/* ===== Baidu AI Studio PaddleOCR Style CSS ===== */

/* ===== CSS Variables ===== */
:root {
    --primary-color: #2932E1;
    --primary-hover: #515eed;
    --primary-light: #e8eafc;
    --title-color: #140E35;
    --text-color: #565772;
    --text-light: #9498AC;
    --text-disabled: #C8CEDE;
    --bg-main: #F8F9FB;
    --bg-white: #ffffff;
    --bg-hover: #F7F7F9;
    --bg-disabled: #f5f5f5;
    --border-color: #E8EDF6;
    --border-input: #d9d9d9;
    --shadow-card: 0 2px 8px rgba(37, 38, 94, 0.08);
    --shadow-hover: 0 4px 12px rgba(37, 38, 94, 0.12);
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
}

/* ===== Global Styles ===== */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: var(--text-color) !important;
}

/* Global text color for all elements */
.gradio-container * {
    color: inherit;
}

/* Ensure all text elements have proper color */
p, span, div, li, td, th {
    color: var(--text-color);
}

/* Fix for any white text on light background */
.gr-box, .gr-panel, .white-container {
    color: var(--text-color) !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 10px !important;
}

/* Force all containers to use full width */
.gradio-container .app,
.gradio-container main,
.gradio-container .wrap,
.gradio-container .contain,
.row,
#results-column,
#sidebar-column,
.white-container,
.column {
    max-width: none !important;
    width: 100% !important;
}

/* Override any flex basis constraints */
.row > .column {
    flex-basis: 0 !important;
}

/* ===== Typography ===== */
#markdown-title {
    text-align: center;
    color: var(--title-color) !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

#markdown-title h1 {
    color: var(--primary-color) !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

label, .gr-label {
    color: var(--title-color) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    margin-bottom: 4px !important;
}

/* Remove block-info background */
span[data-testid="block-info"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: inherit !important;
    font-weight: inherit !important;
}

/* Hide Examples default label */
.gallery.svelte-p5q82i {
    margin-top: 0 !important;
}

.block.svelte-1svsvh2 .label.svelte-p5q82i {
    display: none !important;
}

/* Custom Markdown Headers - All levels use same title color */
.custom-markdown h3,
.custom-markdown h4,
.custom-markdown h5,
.custom-markdown h6 {
    color: var(--title-color) !important;
    font-weight: 600 !important;
    margin-bottom: 16px !important;
}

.custom-markdown h3 {
    font-size: 20px !important;
}

.custom-markdown h4 {
    font-size: 16px !important;
}

.custom-markdown h5 {
    font-size: 14px !important;
    margin-bottom: 12px !important;
}

/* ===== Sidebar Toggle ===== */
#sidebar-toggle-btn {
    position: fixed !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 1000 !important;
    background: linear-gradient(135deg, var(--primary-color) 0%, #4658FF 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 18px 12px !important;
    cursor: pointer !important;
    box-shadow: 3px 0 12px rgba(41, 50, 225, 0.4) !important;
    transition: all 0.3s ease !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 8px !important;
    line-height: 1 !important;
    letter-spacing: 0.5px !important;
}

#sidebar-toggle-btn:hover {
    background: linear-gradient(135deg, #4658FF 0%, var(--primary-color) 100%) !important;
    box-shadow: 3px 0 16px rgba(41, 50, 225, 0.5) !important;
    padding-right: 16px !important;
}

#sidebar-toggle-btn .toggle-icon {
    font-size: 20px !important;
    display: block !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
}

#sidebar-toggle-btn .toggle-text {
    font-size: 12px !important;
    display: block !important;
    white-space: nowrap !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
    writing-mode: vertical-rl !important;
    text-orientation: mixed !important;
}

.sidebar-column {
    transition: all 0.3s ease !important;
    overflow: visible !important;
    position: relative !important;
}

.sidebar-hidden {
    transform: translateX(-90%) !important;
    opacity: 0.3 !important;
    pointer-events: none !important;
}

.sidebar-hidden:hover {
    opacity: 0.5 !important;
}

/* ===== Card & Panel ===== */
.gr-panel, .gr-box, .gr-group {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-card) !important;
}

.form { background: transparent !important; }

/* ===== Buttons ===== */
#analyze-btn, #unzip-btn {
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 32px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

#analyze-btn {
    background: linear-gradient(135deg, var(--primary-color) 0%, #4658FF 100%) !important;
    box-shadow: 0 4px 12px rgba(41, 50, 225, 0.25) !important;
}

#analyze-btn:hover {
    background: linear-gradient(135deg, #4658FF 0%, var(--primary-color) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(41, 50, 225, 0.35) !important;
}

#unzip-btn {
    background: linear-gradient(135deg, #52c41a 0%, #73d13d 100%) !important;
    box-shadow: 0 4px 12px rgba(82, 196, 26, 0.25) !important;
}

#unzip-btn:hover {
    background: linear-gradient(135deg, #73d13d 0%, #52c41a 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(82, 196, 26, 0.35) !important;
}

/* Drag and Drop File Upload Area */
.upload-area {
    width: 100% !important;
}

.drag-drop-file {
    background: #FAFBFF !important;
    border: 1px dashed #D9D9D9 !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.3s ease !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    width: 100% !important;
    font-size: 14px !important;
    text-align: center !important;
}

.drag-drop-file:active {
    background: #E8EAFF !important;
}

.drag-drop-file:hover {
    border-color: var(--primary-color) !important;
    background: #F0F2FF !important;
}

.drag-drop-file-custom {
    background: #FAFBFF !important;
    border: 1px dashed #D9D9D9 !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.3s ease !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    width: 100% !important;
    font-size: 14px !important;
    text-align: center !important;
}

.drag-drop-file-custom button {
    min-height: 100px !important;
    height: 100px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    flex-direction: column !important;
}

/* Hide original content */
.drag-drop-file-custom button .wrap {
    display: none !important;
}

/* Add custom content - only when no file */
.drag-drop-file-custom:not(:has(.file-preview-holder)) button::before {
    content: "📤" !important;
    display: block !important;
    font-size: 32px !important;
    margin-bottom: 8px !important;
}

.drag-drop-file-custom:not(:has(.file-preview-holder)) button::after {
    content: "Click or drag file to upload\\ASupport formats: PDF, JPG, PNG, JPEG" !important;
    white-space: pre-wrap !important;
    display: block !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    color: var(--text-color) !important;
    text-align: center !important;
}

/* When file is uploaded, show normal layout */
.drag-drop-file-custom:has(.file-preview-holder) {
    border-style: solid !important;
    min-height: auto !important;
}

.drag-drop-file-custom:has(.file-preview-holder) button {
    min-height: auto !important;
    height: auto !important;
    padding: 0 !important;
}

.drag-drop-file-custom:has(.file-preview-holder) button::before,
.drag-drop-file-custom:has(.file-preview-holder) button::after {
    display: none !important;
}

.drag-drop-file-custom:hover button::after {
    color: var(--primary-color) !important;
}

.drag-drop-file-custom:active {
    background: #E8EAFF !important;
}

.drag-drop-file-custom:hover {
    border-color: var(--primary-color) !important;
    background: #F0F2FF !important;
}

.drag-drop-file-custom label[data-testid="block-label"] {
    display: none !important;
}

.drag-drop-file-custom .upload-container {
    padding: 0 !important;
}

.drag-drop-file-custom .file-preview-holder {
    margin-top: 8px !important;
    background: #F0F2FF !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px !important;
}

.file-status {
    margin-top: 8px !important;
    color: #52c41a !important;
    font-weight: 500 !important;
}

/* ===== Tabs ===== */
.tabs, .gr-tabs {
    background: var(--bg-white) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px !important;
    border: 1px solid var(--border-color) !important;
}

/* White Container (same style as Tabs) */
.white-container {
    background: var(--bg-white) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px !important;
    border: 1px solid var(--border-color) !important;
}

.tab-nav, .gr-tab-nav {
    background: var(--bg-hover) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 4px !important;
}

.tab-nav button, .gr-tab-nav button {
    background: transparent !important;
    color: var(--title-color) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected, .gr-tab-nav button.selected,
.tab-nav button[aria-selected="true"], .gr-tab-nav button[aria-selected="true"] {
    background: var(--bg-white) !important;
    color: var(--primary-color) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.tab-nav button:hover, .gr-tab-nav button:hover {
    color: var(--primary-color) !important;
}

/* ===== Common Form Item Base Style ===== */
#use_doc_orientation_classify_cb,
#use_doc_unwarping_cb,
#use_textline_orientation_cb,
#text_det_thresh_nb,
#text_det_box_thresh_nb,
#text_det_unclip_ratio_nb,
#text_rec_score_thresh_nb,
#pdf_dpi_nb,
#pdf_max_pages_nb {
    padding: 8px 0 !important;
    background: transparent !important;
    border: none !important;
    border-width: 0 !important;
    border-radius: 0 !important;
    margin-bottom: 4px !important;
}

#use_doc_orientation_classify_cb:hover,
#use_doc_unwarping_cb:hover,
#use_textline_orientation_cb:hover,
#text_det_thresh_nb:hover,
#text_det_box_thresh_nb:hover,
#text_det_unclip_ratio_nb:hover,
#text_rec_score_thresh_nb:hover,
#pdf_dpi_nb:hover,
#pdf_max_pages_nb:hover {
    border-color: transparent !important;
    box-shadow: none !important;
}

/* ===== Common Label Style ===== */
#text_det_thresh_nb span[data-testid="block-info"],
#text_det_box_thresh_nb span[data-testid="block-info"],
#text_det_unclip_ratio_nb span[data-testid="block-info"],
#text_rec_score_thresh_nb span[data-testid="block-info"],
#pdf_dpi_nb span[data-testid="block-info"],
#pdf_max_pages_nb span[data-testid="block-info"] {
    font-size: 14px !important;
    font-weight: 400 !important;
    color: var(--title-color) !important;
}

/* ===== Toggle Switch Style (Module Tab) ===== */
#use_doc_orientation_classify_cb > label,
#use_doc_unwarping_cb > label,
#use_textline_orientation_cb > label {
    display: flex !important;
    flex-direction: row-reverse !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    cursor: pointer !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: var(--title-color) !important;
}

#use_doc_orientation_classify_cb input[type="checkbox"],
#use_doc_unwarping_cb input[type="checkbox"],
#use_textline_orientation_cb input[type="checkbox"] {
    width: 36px !important;
    height: 20px !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    background: #bfbfbf !important;
    border-radius: 10px !important;
    position: relative !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    flex-shrink: 0 !important;
    margin: 0 !important;
}

#use_doc_orientation_classify_cb input[type="checkbox"]::before,
#use_doc_unwarping_cb input[type="checkbox"]::before,
#use_textline_orientation_cb input[type="checkbox"]::before {
    content: '' !important;
    position: absolute !important;
    width: 16px !important;
    height: 16px !important;
    background: white !important;
    border-radius: 50% !important;
    top: 2px !important;
    left: 2px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
}

#use_doc_orientation_classify_cb input[type="checkbox"]:checked,
#use_doc_unwarping_cb input[type="checkbox"]:checked,
#use_textline_orientation_cb input[type="checkbox"]:checked {
    background: var(--primary-color) !important;
}

#use_doc_orientation_classify_cb input[type="checkbox"]:checked::before,
#use_doc_unwarping_cb input[type="checkbox"]:checked::before,
#use_textline_orientation_cb input[type="checkbox"]:checked::before {
    left: 18px !important;
}

/* ===== Number Input Style ===== */
#text_det_thresh_nb > label,
#text_det_box_thresh_nb > label,
#text_det_unclip_ratio_nb > label,
#text_rec_score_thresh_nb > label,
#pdf_dpi_nb > label,
#pdf_max_pages_nb > label {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
}

#text_det_thresh_nb span[data-testid="block-info"],
#text_det_box_thresh_nb span[data-testid="block-info"],
#text_det_unclip_ratio_nb span[data-testid="block-info"],
#text_rec_score_thresh_nb span[data-testid="block-info"],
#pdf_dpi_nb span[data-testid="block-info"],
#pdf_max_pages_nb span[data-testid="block-info"] {
    flex: 1 !important;
}

#text_det_thresh_nb input,
#text_det_box_thresh_nb input,
#text_det_unclip_ratio_nb input,
#text_rec_score_thresh_nb input,
#pdf_dpi_nb input,
#pdf_max_pages_nb input {
    border: 1px solid var(--border-input) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
    width: 70px !important;
    height: 24px !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
    background: #fff !important;
    flex-shrink: 0 !important;
}

#text_det_thresh_nb input:focus,
#text_det_box_thresh_nb input:focus,
#text_det_unclip_ratio_nb input:focus,
#text_rec_score_thresh_nb input:focus,
#pdf_dpi_nb input:focus,
#pdf_max_pages_nb input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(41, 50, 225, 0.1) !important;
    outline: none !important;
}

/* ===== Loader ===== */
.loader {
    border: 4px solid var(--bg-hover);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    width: 48px;
    height: 48px;
    animation: spin 1s linear infinite;
    margin: 24px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loader-container {
    text-align: center;
    margin: 24px 0;
    color: var(--text-color);
}

.loader-container-prepare {
    text-align: left;
    margin: 16px 0;
}

.loader-container-prepare > div {
    background: linear-gradient(135deg, #f8faff 0%, #f0f4ff 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-left: 4px solid var(--primary-color) !important;
}

/* ===== Gallery ===== */
.gr-gallery, .gallery {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

.gradio-gallery-item:hover {
    background-color: transparent !important;
    filter: none !important;
    transform: none !important;
}

/* ===== Spacing Classes ===== */
.tight-spacing { margin-bottom: -5px !important; }

.tight-spacing-as {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
    padding: 12px 16px !important;
    background: var(--bg-white) !important;
    border-radius: var(--radius-md) !important;
    border-left: 3px solid var(--primary-color) !important;
    color: var(--text-color) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

.image-container img { display: inline-block !important; }

/* ===== File Download & JSON ===== */
.file-download { margin-top: 16px !important; }

.json-holder {
    background: var(--bg-white) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
}

/* ===== JSON Viewer Text Color Fix ===== */
.json-holder span,
.json-holder .line,
.json-holder code,
[data-testid="json"] span,
[data-testid="json"] .line,
.gr-json span,
.gr-json .line {
    color: var(--text-color) !important;
}

/* JSON key names */
.json-holder .key,
[data-testid="json"] .key {
    color: var(--primary-color) !important;
}

/* JSON string values */
.json-holder .string,
[data-testid="json"] .string {
    color: #22863a !important;
}

/* JSON number values */
.json-holder .number,
[data-testid="json"] .number {
    color: #005cc5 !important;
}

/* ===== Examples Gallery Button Text Fix ===== */
.gallery button,
.gr-samples button,
.gr-examples button,
[data-testid="examples"] button {
    color: var(--text-color) !important;
}

.gallery button span,
.gr-samples button span,
.gr-examples button span {
    color: var(--text-color) !important;
}

/* Examples file names */
.gr-sample-textbox,
.sample-textbox,
[data-testid="textbox"] input {
    color: var(--text-color) !important;
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
    .gradio-container { padding: 12px !important; }
    
    #analyze-btn, #unzip-btn {
        padding: 10px 20px !important;
        font-size: 14px !important;
    }
}

/* ===== Banner ===== */
.banner-container {
    background: transparent !important;
    margin: -16px -16px 16px -16px !important;
    padding: 0 !important;
    width: calc(100% + 32px) !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

.banner-container .image-container {
    background: transparent !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.banner-container .image-container button {
    cursor: default !important;
    background: transparent !important;
}

.banner-container .image-frame {
    background: transparent !important;
    display: flex !important;
    justify-content: center !important;
}

.banner-container img {
    max-width: 100% !important;
    width: auto !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
    object-fit: contain !important;
}

.banner-container .icon-button-wrapper,
.banner-container .icon-buttons,
.banner-container .top-panel {
    display: none !important;
}

/* ===== FORCE TEXT COLOR FIXES ===== */

/* Force all Tab buttons to have visible text */
button[role="tab"],
.tabs button,
.gr-tabs button,
[data-testid="tab-nav"] button,
.tabitem button,
div[class*="tab"] button {
    color: var(--title-color) !important;
}

button[role="tab"][aria-selected="true"],
button[role="tab"]:hover {
    color: var(--primary-color) !important;
}

/* Force Number input labels to be visible */
.gr-number label,
.gr-number span,
.gr-number input,
input[type="number"],
.gr-input label,
.gr-input span {
    color: var(--title-color) !important;
}

/* Force all form labels */
.gr-form label,
.gr-form span[data-testid="block-info"],
.gr-block label,
.gr-block span {
    color: var(--title-color) !important;
}

/* JSON component text */
.gr-json,
.gr-json *,
pre, code,
.json-container,
.json-container * {
    color: var(--text-color) !important;
}

/* Specific JSON styling */
.gr-json .key { color: var(--primary-color) !important; }
.gr-json .string { color: #22863a !important; }
.gr-json .number { color: #005cc5 !important; }
.gr-json .boolean { color: #d73a49 !important; }
.gr-json .null { color: #6a737d !important; }

/* Svelte component overrides */
[class*="svelte"] button,
[class*="svelte"] span,
[class*="svelte"] label,
[class*="svelte"] input {
    color: var(--title-color) !important;
}

/* Specific Gradio 5.x overrides */
.block span,
.block label,
.wrap span,
.wrap label {
    color: var(--title-color) !important;
}

/* Input fields text color */
input, textarea, select {
    color: var(--title-color) !important;
}

/* Placeholder text */
input::placeholder,
textarea::placeholder {
    color: var(--text-light) !important;
}
"""

EXAMPLE_DIR = BASE_DIR / "examples"
EXAMPLE_PDF_DIR = BASE_DIR / "examples_pdf"


# Dynamically load example files from directories
def load_examples_from_dir(directory, extensions):
    """Load all files with specified extensions from directory"""
    examples = []
    if directory.exists() and directory.is_dir():
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                examples.append([str(file_path)])
    return examples


# Load image examples (png, jpg, jpeg)
EXAMPLE_TEST = load_examples_from_dir(EXAMPLE_DIR, {'.png', '.jpg', '.jpeg'})

# Load PDF examples
EXAMPLE_PDF = load_examples_from_dir(EXAMPLE_PDF_DIR, {'.pdf'})

DESC_DICT = {
    "use_doc_orientation_classify": "Enable the document image orientation classification module. When enabled, you can correct distorted images, such as wrinkles, tilts, etc.",
    "use_doc_unwarping": "Enable the document unwarping module. When enabled, you can correct distorted images, such as wrinkles, tilts, etc.",
    "use_textline_orientation": "Enable the text line orientation classification module to support the distinction and correction of text lines of 0 degrees and 180 degrees.",
    "text_det_thresh_nb": "In the output probability map, only pixels with scores greater than the threshold are considered text pixels, and the value range is 0~1.",
    "text_det_box_thresh_nb": "When the average score of all pixels in the detection result border is greater than the threshold, the result will be considered as a text area, and the value range is 0 to 1. If missed detection occurs, this value can be appropriately lowered.",
    "text_det_unclip_ratio_nb": "Use this method to expand the text area. The larger the value, the larger the expanded area.",
    "text_rec_score_thresh_nb": "After text detection, the text box performs text recognition, and the text results with scores greater than the threshold will be retained. The value range is 0~1.",
    "pdf_dpi_nb": "PDF rendering DPI. Higher values produce clearer images but use more memory. Recommended: 72-300.",
    "pdf_max_pages_nb": "Maximum number of PDF pages to process. Pages beyond this limit will not be processed.",
}

tmp_time = {}
lock = threading.Lock()


def gen_tooltip_radio(desc_dict):
    tooltip = {}
    for key, desc in desc_dict.items():
        suffixes = ["_cb", "_rb", "_md"]
        if key.endswith("_nb"):
            suffix = "_nb"
            suffixes = ["_nb", "_md"]
            key = key[: -len(suffix)]
        for suffix in suffixes:
            tooltip[f"{key}{suffix}"] = desc
    return tooltip


TOOLTIP_RADIO = gen_tooltip_radio(DESC_DICT)


def url_to_bytes(url, *, timeout=10):
    """Download image from URL"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def base64_to_bytes(base64_str):
    """Decode base64 string to bytes"""
    return base64.b64decode(base64_str)


def get_image_bytes(image_data):
    """Get image bytes from either URL or base64 string"""
    if image_data is None:
        return None
    
    # Check if it's a URL (starts with http:// or https://)
    if isinstance(image_data, str) and (image_data.startswith('http://') or image_data.startswith('https://')):
        return url_to_bytes(image_data)
    # Check if it's a relative URL (starts with /static/)
    elif isinstance(image_data, str) and image_data.startswith('/static/'):
        full_url = f"{API_BASE}{image_data}"
        return url_to_bytes(full_url)
    # Otherwise assume it's base64
    elif isinstance(image_data, str):
        return base64_to_bytes(image_data)
    else:
        return None


def bytes_to_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


# ============ API 处理函数 (适配 DeepX OCR Server) ============
def process_file(
    file_path,
    image_input,
    use_doc_orientation_classify,
    use_doc_unwarping,
    use_textline_orientation,
    text_det_thresh,
    text_det_box_thresh,
    text_det_unclip_ratio,
    text_rec_score_thresh,
    pdf_dpi,
    pdf_max_pages,
):
    """Process uploaded file with DeepX OCR Server API"""
    try:
        if not file_path and not image_input:
            return None
        
        if file_path:
            if Path(file_path).suffix.lower() == ".pdf":
                file_type = 0  # PDF
            else:
                file_type = 1  # Image
        else:
            file_path = image_input
            file_type = 1  # Image
        
        # Read file content
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Call DeepX OCR Server API
        file_data = base64.b64encode(file_bytes).decode("ascii")
        headers = {
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
        }

        request_body = {
            "file": file_data,
            "fileType": file_type,
            "visualize": True,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": use_doc_unwarping,
            "useTextlineOrientation": use_textline_orientation,
            "textDetThresh": text_det_thresh,
            "textDetBoxThresh": text_det_box_thresh,
            "textDetUnclipRatio": text_det_unclip_ratio,
            "textRecScoreThresh": text_rec_score_thresh,
        }
        
        # Add PDF-specific parameters
        if file_type == 0:
            request_body["pdfDpi"] = int(pdf_dpi)
            request_body["pdfMaxPages"] = int(pdf_max_pages)

        response = requests.post(
            API_URL,
            json=request_body,
            headers=headers,
            timeout=300,
        )
        
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {response.text}") from e
        
        # Parse API response
        result = response.json()
        
        if result.get("errorCode", 0) != 0:
            raise gr.Error(f"OCR processing failed: {result.get('errorMsg', 'Unknown error')}")
        
        overall_ocr_res_images = []
        output_json = result.get("result", {})
        input_images = []
        
        # Handle Image response (fileType=1)
        if file_type == 1:
            ocr_results = output_json.get("ocrResults", [])
            ocr_image_url = output_json.get("ocrImage", "")
            
            # Get visualization image
            if ocr_image_url:
                ocr_image_bytes = get_image_bytes(ocr_image_url)
                if ocr_image_bytes:
                    overall_ocr_res_images.append(ocr_image_bytes)
            
            # Add original input image
            input_images.append(file_bytes)
        
        # Handle PDF response (fileType=0)
        else:
            pages = output_json.get("pages", [])
            for page in pages:
                page_index = page.get("pageIndex", 0)
                ocr_image_url = page.get("ocrImage", "")
                
                if ocr_image_url:
                    ocr_image_bytes = get_image_bytes(ocr_image_url)
                    if ocr_image_bytes:
                        overall_ocr_res_images.append(ocr_image_bytes)
                        input_images.append(ocr_image_bytes)

        return {
            "original_file": file_path,
            "file_type": "pdf" if file_type == 0 else "image",
            "overall_ocr_res_images": overall_ocr_res_images,
            "output_json": output_json,
            "input_images": input_images,
            "api_response": result,
        }

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error processing file: {str(e)}")


def export_full_results(results):
    """Create ZIP file with all analysis results"""
    try:
        global tmp_time
        if not results:
            raise ValueError("No results to export")

        filename = Path(results["original_file"]).stem + f"_{uuid.uuid4().hex}.zip"
        zip_path = Path(TEMP_DIR.name, filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, img_bytes in enumerate(results["overall_ocr_res_images"]):
                zipf.writestr(f"overall_ocr_res_images/page_{i+1}.jpg", img_bytes)

            zipf.writestr(
                "output.json",
                json.dumps(results["output_json"], indent=2, ensure_ascii=False),
            )

            # Add API response
            api_response = results.get("api_response", {})
            zipf.writestr(
                "api_response.json",
                json.dumps(api_response, indent=2, ensure_ascii=False),
            )

            for i, img_bytes in enumerate(results["input_images"]):
                zipf.writestr(f"input_images/page_{i+1}.jpg", img_bytes)
        
        with lock:
            tmp_time[zip_path] = time.time()
        return str(zip_path)

    except Exception as e:
        raise gr.Error(f"Error creating ZIP file: {str(e)}")


def on_file_change_from_examples_image(file):
    return on_common_change(file, "examples_image")


def on_file_change_from_examples_pdf(file):
    return on_common_change(file, "examples_pdf")


def on_file_change_from_input_impl(file_select, self_input, ref_input, called_from):
    if file_select != '':
        if ref_input is not None:
            if self_input is not None:
                if self_input != ref_input:
                    return on_common_change(self_input, called_from)
                else:
                    return gr.Textbox(value=None, visible=False), gr.File(value=None), gr.Image(value=None)
            else:
                return gr.skip(), gr.skip(), gr.skip()
        else:
            if self_input is not None:
                if (file_select in os.path.basename(self_input)):
                    return gr.Textbox(value=None, visible=False), gr.File(value=None), gr.Image(value=None)
                else:
                    return on_common_change(self_input, called_from)
            else:
                if self_input is None:
                    return gr.Textbox(value=None, visible=False), gr.File(value=None), gr.Image(value=None)
                else:
                    return gr.skip(), gr.skip(), gr.skip()
    else:
        return gr.skip(), gr.skip(), gr.skip()


def on_file_change_from_file_input(file_select, self_input, ref_input):
    return on_file_change_from_input_impl(file_select, self_input, ref_input, "file_input")


def on_file_change_from_image_input(file_select, self_input, ref_input):
    return on_file_change_from_input_impl(file_select, self_input, ref_input, "image_input")


def on_common_change_impl(file):
    """Handle file input change and return status textbox"""
    if file is not None:
        try:
            filename = os.path.basename(file.name) if hasattr(file, 'name') else os.path.basename(str(file))
            return gr.Textbox(value=f"✅ Chosen file: {filename}", visible=True)
        except Exception:
            return gr.Textbox(value="✅ File selected", visible=True)
    return gr.Textbox(value=None, visible=False)


def on_common_change(file, called_from):
    """Handle file input change and return status textbox"""
    input_select = on_common_change_impl(file)

    if called_from == 'examples_image':
        file_input = gr.File(value=None)
        image_input = gr.skip()
    elif called_from == 'examples_pdf':
        file_input = gr.skip()
        image_input = gr.Image(value=None)
    elif called_from == 'file_input':
        file_input = gr.skip()
        image_input = gr.Image(value=None)
    elif called_from == 'image_input':
        file_input = gr.File(value=None)
        image_input = gr.skip()
    else:
        raise ValueError("Invalid called_from value")
    
    return input_select, file_input, image_input


def validate_file_input(file_path, image_input):
    """Validate file selection"""
    if not file_path and not image_input:
        gr.Warning("📁 Please select a file first before parsing.")


def toggle_spinner(file_path, image_input):
    """Show spinner when file is present"""
    if not file_path and not image_input:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
    return (
        gr.Column(visible=True),
        gr.Column(visible=False),
        gr.File(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def hide_spinner(results):
    """Hide spinner and show tabs when results are available"""
    if results:
        return gr.Column(visible=False), gr.update(visible=True)
    else:
        return gr.Column(visible=False), gr.skip()


def update_display(results):
    if not results:
        return [gr.skip()] * (MAX_NUM_PAGES + 1 + len(gallery_list))
    
    # Validate results
    assert len(results["overall_ocr_res_images"]) <= MAX_NUM_PAGES, len(
        results["overall_ocr_res_images"]
    )
    
    # Prepare OCR images
    ocr_imgs = []
    for img in results["overall_ocr_res_images"]:
        ocr_imgs.append(gr.Image(value=bytes_to_image(img), visible=True))
    for _ in range(len(results["overall_ocr_res_images"]), MAX_NUM_PAGES):
        ocr_imgs.append(gr.Image(visible=False))
    
    # Prepare JSON output
    output_json = [gr.JSON(value=results["output_json"], visible=True)]
    
    # Prepare gallery images - convert bytes to PIL Image for gallery
    gallery_images = []
    for img_data in results["input_images"]:
        if isinstance(img_data, bytes):
            gallery_images.append(bytes_to_image(img_data))
        else:
            gallery_images.append(img_data)
    
    # Update all galleries with the same images
    gallery_list_imgs = []
    for i in range(len(gallery_list)):
        gallery_list_imgs.append(
            gr.Gallery(
                value=gallery_images,
                rows=len(gallery_images) if len(gallery_images) > 0 else 1,
            )
        )
    
    return ocr_imgs + output_json + gallery_list_imgs


def update_image(evt: gr.SelectData):
    update_images = []
    for index in range(MAX_NUM_PAGES):
        update_images.append(
            gr.Image(visible=False) if index != evt.index else gr.Image(visible=True)
        )
    return update_images


def delete_file_periodically():
    global tmp_time
    while True:
        current_time = time.time()
        delete_tmp = []
        for filename, start_time in list(tmp_time.items()):
            if (current_time - start_time) >= TMP_DELETE_TIME:
                if os.path.exists(filename):
                    os.remove(filename)
                    delete_tmp.append(filename)
        for filename in delete_tmp:
            with lock:
                del tmp_time[filename]
        time.sleep(THREAD_WAKEUP_TIME)


# Banner paths
BANNER_PATH = str(BASE_DIR / "res" / "img" / "deepx-baidu-pp-banner.png")
BANNER_CES_PATH = str(BASE_DIR / "res" / "img" / "DEEPX-Banner-CES-2026-01.png")

# Force English language script
FORCE_EN_SCRIPT = """
<script>
    try {
        Object.defineProperty(navigator, 'language', {
            get: function() { return 'en-US'; }
        });
        Object.defineProperty(navigator, 'languages', {
            get: function() { return ['en-US', 'en']; }
        });
    } catch (e) {
        console.log("Language override failed");
    }
</script>
"""

# ============ 构建 Gradio 界面 (与原项目布局完全一致) ============
with gr.Blocks(css=CSS, title=TITLE, theme=paddle_theme, head=FORCE_EN_SCRIPT) as demo:
    results_state = gr.State()

    # Top Banner
    gr.Image(
        value=BANNER_PATH,
        show_label=False,
        show_download_button=False,
        show_fullscreen_button=False,
        container=False,
        elem_classes=["banner-container"],
    )
    
    with gr.Row():
        with gr.Column(scale=3, elem_classes=["sidebar-column"], elem_id="sidebar-column"):
        
            # Upload section
            gr.Markdown("#### 📁 Input File", elem_classes="custom-markdown")
            with gr.Column(elem_classes=["white-container"]):
                with gr.Column(elem_classes=["upload-area"]):
                    file_input = gr.File(
                        file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                        type="filepath",
                        visible=True,
                        show_label=False,
                        elem_classes=["drag-drop-file-custom"],
                    )

                    file_select = gr.Textbox(
                        show_label=False, 
                        visible=False,
                        interactive=False,
                        elem_classes=["file-status"],
                    )

                process_btn = gr.Button(
                    "🚀 Parse Document", elem_id="analyze-btn", variant="primary"
                )

                gr.Markdown("##### 📷 Image Examples", elem_classes="custom-markdown")

                image_input = gr.Image(
                    label="Image",
                    sources="upload",
                    type="filepath",
                    visible=False,
                    interactive=True,
                    placeholder="Click to upload file",
                )

                examples_image = gr.Examples(
                    fn=on_file_change_from_examples_image,
                    inputs=image_input,
                    outputs=[file_select, file_input, image_input],
                    examples_per_page=8,
                    examples=EXAMPLE_TEST,
                    run_on_click=True,
                )
                
                gr.Markdown("##### 📄 PDF Examples", elem_classes="custom-markdown")
                examples_pdf = gr.Examples(
                    fn=on_file_change_from_examples_pdf,
                    inputs=file_input,
                    outputs=[file_select, file_input, image_input],
                    examples_per_page=5,
                    examples=EXAMPLE_PDF,
                    run_on_click=True,
                )

                image_input.change(
                    fn=on_file_change_from_image_input,
                    inputs=[file_select, image_input, file_input],
                    outputs=[file_select, file_input, image_input],
                )

                file_input.change(
                    fn=on_file_change_from_file_input, 
                    inputs=[file_select, file_input, image_input],
                    outputs=[file_select, file_input, image_input]
                )
            
            # Settings section
            gr.Markdown("#### ⚙️ Settings", elem_classes="custom-markdown")
            with gr.Tabs() as advance_options_tabs:
                with gr.Tab("Module Selection") as Module_Options:
                    use_doc_orientation_classify_cb = gr.Checkbox(
                        value=False,
                        interactive=True,
                        label="Image Orientation Correction",
                        show_label=True,
                        elem_id="use_doc_orientation_classify_cb",
                    )
                    use_doc_unwarping_cb = gr.Checkbox(
                        value=False,
                        interactive=True,
                        label="Image Distortion Correction",
                        show_label=True,
                        elem_id="use_doc_unwarping_cb",
                    )
                    use_textline_orientation_cb = gr.Checkbox(
                        value=False,
                        interactive=True,
                        label="Text Line Orientation Correction",
                        show_label=True,
                        elem_id="use_textline_orientation_cb",
                    )
                
                with gr.Tab("OCR Settings") as Text_detection_Options:
                    text_det_thresh_nb = gr.Number(
                        value=0.30,
                        step=0.01,
                        minimum=0.00,
                        maximum=1.00,
                        interactive=True,
                        label="Text Detection Pixel Threshold",
                        show_label=True,
                        elem_id="text_det_thresh_nb",
                    )
                    text_det_box_thresh_nb = gr.Number(
                        value=0.60,
                        step=0.01,
                        minimum=0.00,
                        maximum=1.00,
                        interactive=True,
                        label="Text Detection Box Threshold",
                        show_label=True,    
                        elem_id="text_det_box_thresh_nb",
                    )
                    text_det_unclip_ratio_nb = gr.Number(
                        value=1.5,
                        step=0.1,
                        minimum=1.0,
                        maximum=3.0,
                        interactive=True,
                        label="Expansion Coefficient",
                        show_label=True,
                        elem_id="text_det_unclip_ratio_nb",
                    )
                    text_rec_score_thresh_nb = gr.Number(
                        value=0.00,
                        step=0.01,
                        minimum=0,
                        maximum=1.00,
                        interactive=True,
                        label="Text Recognition Score Threshold",
                        show_label=True,
                        elem_id="text_rec_score_thresh_nb",
                    )

                with gr.Tab("PDF Settings") as PDF_Options:
                    pdf_dpi_nb = gr.Number(
                        value=150,
                        step=10,
                        minimum=72,
                        maximum=300,
                        interactive=True,
                        label="PDF Render DPI",
                        show_label=True,
                        elem_id="pdf_dpi_nb",
                    )
                    pdf_max_pages_nb = gr.Number(
                        value=10,
                        step=1,
                        minimum=1,
                        maximum=100,
                        interactive=True,
                        label="PDF Max Pages",
                        show_label=True,
                        elem_id="pdf_max_pages_nb",
                    )
                    gr.HTML(
                        """
                        <div style="
                            padding: 12px 16px;
                            background: #FFF7E6;
                            border-left: 3px solid #FAAD14;
                            border-radius: 6px;
                            margin-top: 12px;
                            font-size: 13px;
                            color: #8C6D1F;
                            line-height: 1.5;
                        ">
                            <strong style="color: #D48806;">⚠️ Memory Notice:</strong><br>
                            A4 page @ 150 DPI ≈ 8.7MB/page<br>
                            Recommended: DPI=150, Max Pages=10
                        </div>
                        """
                    )

        # Results display section
        with gr.Column(scale=7, elem_classes=["white-container"], elem_id="results-column"):
            
            gr.Markdown("### 📋 Results", elem_classes="custom-markdown")

            loading_spinner = gr.Column(
                visible=False, elem_classes=["loader-container"]
            )
            with loading_spinner:
                gr.HTML(
                    """
                    <div class="loader"></div>
                    <p style="color: #565772; font-size: 14px;">Processing, please wait...</p>
                    """
                )
            prepare_spinner = gr.Column(
                visible=True, elem_classes=["loader-container-prepare"]
            )
            with prepare_spinner:
                gr.HTML(
                    """
                    <div style="font-size: 18px; font-weight: 600; color: #140E35; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                        <span style="background: #2932E1; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px;">GUIDE</span>
                        User Guide
                    </div>
                    <div style="display: grid; gap: 12px; color: #565772;">
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="background: #2932E1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0;">1</span>
                            <div><b style="color: #140E35;">Upload Your File</b><br><span style="font-size: 13px; color: #565772;">Upload directly or select from Image/PDF Examples below<br>Supported formats: JPG, PNG, PDF, JPEG</span></div>
                        </div>
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="background: #2932E1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0;">2</span>
                            <div><b style="color: #140E35;">Click Parse Document Button</b><br><span style="font-size: 13px; color: #565772;">System will process automatically</span></div>
                        </div>
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="background: #2932E1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0;">3</span>
                            <div><b style="color: #140E35;">View & Download Results</b><br><span style="font-size: 13px; color: #565772;">Results will be displayed after processing</span></div>
                        </div>
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <span style="background: #2932E1; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0;">4</span>
                            <div><b style="color: #140E35;">Expand Results View</b><br><span style="font-size: 13px; color: #565772;">Click <b style="color: #140E35;">HIDE LEFT MENU</b> button on the left to view results in full screen</span></div>
                        </div>
                    </div>
                    <div style="margin-top: 16px; padding: 12px 16px; background: #F6FFED; border-radius: 6px; border-left: 3px solid #52C41A;">
                        <span style="font-weight: 600; color: #389E0D;">✅ Server Status:</span>
                        <span style="color: #237804;">DeepX OCR Server Ready</span>
                    </div>
                    """
                )

            overall_ocr_res_images = []
            output_json_list = []
            gallery_list = []
            with gr.Tabs(visible=False) as tabs:
                with gr.Tab("OCR"):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=1):
                            gallery_ocr_det = gr.Gallery(
                                show_label=False,
                                allow_preview=False,
                                preview=False,
                                columns=1,
                                min_width=10,
                                object_fit="contain",
                            )
                            gallery_list.append(gallery_ocr_det)
                        with gr.Column(scale=10):
                            for i in range(MAX_NUM_PAGES):
                                overall_ocr_res_images.append(
                                    gr.Image(
                                        label=f"OCR Image {i}",
                                        show_label=True,
                                        visible=False,
                                    )
                                )
                with gr.Tab("JSON"):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=1):
                            gallery_json = gr.Gallery(
                                show_label=False,
                                allow_preview=False,
                                preview=False,
                                columns=1,
                                min_width=10,
                                object_fit="contain",
                            )
                            gallery_list.append(gallery_json)
                        with gr.Column(scale=10):
                            gr.HTML(
                                """
                            <style>
                            .line.svelte-19ir0ev svg {
                                width: 30px !important;
                                height: 30px !important;
                                min-width: 30px !important;
                                min-height: 30px !important;
                                padding: 0 !important;
                                font-size: 18px !important;
                            }
                            .line.svelte-19ir0ev span:contains('Object(') {
                                font-size: 12px;
                                }
                            </style>
                            """
                            )
                            output_json_list.append(
                                gr.JSON(
                                    visible=False,
                                )
                            )
            download_all_btn = gr.Button(
                "📦 Download Full Results (ZIP)",
                elem_id="unzip-btn",
                variant="primary",
                visible=False,
            )

            download_file = gr.File(visible=False, label="Download File")

            gr.Markdown("", elem_classes="custom-markdown")

            gr.Image(
                value=BANNER_CES_PATH,
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False,
                container=False,
                elem_classes=["banner-container"],
            )

    # Sidebar toggle button
    gr.HTML(
        """
        <button id="sidebar-toggle-btn">
            <span class="toggle-icon">◀</span>
            <span class="toggle-text">HIDE LEFT MENU</span>
        </button>
        """
    )        
    
    gr.Markdown("", elem_classes="custom-markdown")
    gr.Image(
        value=BANNER_PATH,
        show_label=False,
        show_download_button=False,
        show_fullscreen_button=False,
        container=False,
        elem_classes=["banner-container"],
    )
    
    # ============ 事件处理 ============
    process_btn.click(
        validate_file_input,
        inputs=[file_input, image_input],
        outputs=[],
    ).then(
        toggle_spinner,
        inputs=[file_input, image_input],
        outputs=[
            loading_spinner,
            prepare_spinner,
            download_file,
            tabs,
            download_all_btn,
        ],
    ).then(
        process_file,
        inputs=[
            file_input,
            image_input,
            use_doc_orientation_classify_cb,
            use_doc_unwarping_cb,
            use_textline_orientation_cb,
            text_det_thresh_nb,
            text_det_box_thresh_nb,
            text_det_unclip_ratio_nb,
            text_rec_score_thresh_nb,
            pdf_dpi_nb,
            pdf_max_pages_nb,
        ],
        outputs=[results_state],
    ).then(
        hide_spinner, inputs=[results_state], outputs=[loading_spinner, tabs]
    ).then(
        update_display,
        inputs=[results_state],
        outputs=overall_ocr_res_images + output_json_list + gallery_list,
    ).then(
        lambda results: gr.update(visible=True) if results else gr.skip(),
        inputs=[results_state],
        outputs=download_all_btn,
    )

    gallery_ocr_det.select(update_image, outputs=overall_ocr_res_images)

    download_all_btn.click(
        export_full_results, inputs=[results_state], outputs=[download_file]
    ).success(lambda: gr.update(visible=True), outputs=[download_file])

    demo.load(
        fn=lambda: None,
        inputs=[],
        outputs=[],
        js=f"""
        () => {{
            // Sidebar toggle functionality
            let sidebarVisible = true;
            const toggleBtn = document.getElementById('sidebar-toggle-btn');
            const sidebar = document.getElementById('sidebar-column');
            const resultsColumn = document.getElementById('results-column');
            
            if (toggleBtn && sidebar) {{
                toggleBtn.addEventListener('click', () => {{
                    sidebarVisible = !sidebarVisible;
                    const icon = toggleBtn.querySelector('.toggle-icon');
                    const text = toggleBtn.querySelector('.toggle-text');
                    
                    if (sidebarVisible) {{
                        // Show: restore display first, then animate
                        sidebar.style.display = '';
                        sidebar.classList.remove('sidebar-hidden');
                        setTimeout(() => {{
                            sidebar.style.transform = 'translateX(0)';
                            sidebar.style.opacity = '1';
                        }}, 10);
                        if (resultsColumn) {{
                            resultsColumn.style.flexGrow = '8';
                        }}
                        if (icon) icon.textContent = '◀';
                        if (text) text.textContent = 'HIDE LEFT MENU';
                    }} else {{
                        // Hide: animate to 90%, then apply display:none
                        sidebar.classList.add('sidebar-hidden');
                        sidebar.style.transform = 'translateX(-90%)';
                        sidebar.style.opacity = '0.3';
                        setTimeout(() => {{
                            if (!sidebarVisible) {{
                                sidebar.style.display = 'none';
                            }}
                        }}, 300);
                        if (resultsColumn) {{
                            resultsColumn.style.flexGrow = '12';
                        }}
                        if (icon) icon.textContent = '▶';
                        if (text) text.textContent = 'SHOW LEFT MENU';
                    }}
                }});
            }}
            
            const tooltipTexts = {TOOLTIP_RADIO};
            let tooltip = document.getElementById("custom-tooltip");
            if (!tooltip) {{
                tooltip = document.createElement("div");
                tooltip.id = "custom-tooltip";
                tooltip.style.position = "fixed";
                tooltip.style.background = "rgba(0, 0, 0, 0.75)";
                tooltip.style.color = "white";
                tooltip.style.padding = "6px 10px";
                tooltip.style.borderRadius = "4px";
                tooltip.style.fontSize = "13px";
                tooltip.style.maxWidth = "300px";
                tooltip.style.zIndex = "10000";
                tooltip.style.pointerEvents = "none";
                tooltip.style.transition = "opacity 0.2s";
                tooltip.style.opacity = "0";
                tooltip.style.whiteSpace = "normal";
                document.body.appendChild(tooltip);
            }}
            Object.keys(tooltipTexts).forEach(id => {{
                const elem = document.getElementById(id);
                if (!elem) return;
                function showTooltip(e) {{
                    tooltip.style.opacity = "1";
                    tooltip.innerText = tooltipTexts[id];
                    let x = e.clientX + 10;
                    let y = e.clientY + 10;
                    if (x + tooltip.offsetWidth > window.innerWidth) {{
                        x = e.clientX - tooltip.offsetWidth - 10;
                    }}
                    if (y + tooltip.offsetHeight > window.innerHeight) {{
                        y = e.clientY - tooltip.offsetHeight - 10;
                    }}
                    tooltip.style.left = x + "px";
                    tooltip.style.top = y + "px";
                }}
                function hideTooltip() {{
                    tooltip.style.opacity = "0";
                }}
                elem.addEventListener("mousemove", showTooltip);
                elem.addEventListener("mouseleave", hideTooltip);
            }});
        }}
        """,
    )

if __name__ == "__main__":
    t = threading.Thread(target=delete_file_periodically, daemon=True)
    t.start()

    allowed_dirs = [
        str(BASE_DIR / "res"), 
        str(BASE_DIR / "examples"),
        str(BASE_DIR / "examples_pdf"),
    ]

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=False,
        allowed_paths=allowed_dirs,
    )