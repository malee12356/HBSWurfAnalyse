import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import base64
import os
import re
import sys
from typing import Optional, Dict, Any
import numpy as np
# TEST für update
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from engine import AnalysisEngine, select_video_capture, pick_default_pose_backend, optional_import, load_reference_sequence, build_detailed_report, ensure_cv2, create_text_panel


class ToggledFrame(tk.Frame):
    def __init__(self, parent, text="", expanded=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.show = tk.IntVar(value=1 if expanded else 0)
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill="x", expand=1)
        init_symbol = '▼' if expanded else '▶'
        self.toggle_button = ttk.Checkbutton(self.title_frame, width=2, text=init_symbol, command=self.toggle, variable=self.show, style='Toolbutton')
        self.toggle_button.pack(side="left")
        ttk.Label(self.title_frame, text=text, font=("Helvetica", 10, "bold")).pack(side="left", fill="x", expand=1)
        self.sub_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
        if expanded:
            self.sub_frame.pack(fill="x", expand=1, pady=2, padx=5)

    def toggle(self):
        if bool(self.show.get()):
            self.sub_frame.pack(fill="x", expand=1, pady=2, padx=5)
            self.toggle_button.configure(text='▼')
        else:
            self.sub_frame.pack_forget()
            self.toggle_button.configure(text='▶')


class AnalysisGUI:
    def __init__(self, reference_json: Optional[str]):
        self.root = tk.Tk()
        self.root.title("Professionelle 3D-Handball-Wurfanalyse")
        self.root.geometry("1600x950") 

        self.source_var = tk.StringVar(value="webcam")
        self.file_var = tk.StringVar(value="")
        self.camera_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Bereit")
        self.pose_backend_var = tk.StringVar(value=pick_default_pose_backend())
        
        self.total_frames = 1
        self.current_frame = 0
        self.fps = 30.0
        self.timeline_var = tk.DoubleVar(value=0)
        
        self.kpt_thr_var = tk.DoubleVar(value=0.3)
        self.use_mog2_var = tk.BooleanVar(value=True)
        self.ball_s_max_var = tk.IntVar(value=80)
        self.ball_v_min_var = tk.IntVar(value=20)
        self.hough_p1_var = tk.IntVar(value=100)
        self.hough_p2_var = tk.IntVar(value=20)
        self.ball_min_rad_var = tk.IntVar(value=6)
        self.ball_max_rad_var = tk.IntVar(value=45)
        
        self.real_length_var = tk.DoubleVar(value=1.65)
        self.calibration_mode_var = tk.StringVar(value="body_height")
        self.use_two_pass_var = tk.BooleanVar(value=True)
        self.target_speed_var = tk.DoubleVar(value=90.0)
        
        self.start_frame_var = tk.StringVar(value="0")
        self.end_frame_var = tk.StringVar(value="999999")
        self.speed_start_frame_var = tk.StringVar(value="0")
        self.speed_end_frame_var = tk.StringVar(value="999999")
        self.loop_var = tk.BooleanVar(value=False)

        self.roi_polygon = []

        self.metric_speed = tk.StringVar(value="-")
        self.metric_trunk = tk.StringVar(value="-")
        self.metric_shoulder = tk.StringVar(value="-")
        self.metric_elbow = tk.StringVar(value="-")
        self.metric_knee = tk.StringVar(value="-")
        self.metric_score = tk.StringVar(value="- / 100")

        self.cap = None
        self.engine = None
        self.running = False
        self.paused = False
        self.seekable = False
        self.report_generated_auto = False
        self.preanalysis_pending = False
        self.preanalysis_running = False
        self.loop_thread = None
        
        self.reference_seq = load_reference_sequence(reference_json)
        self.last_frames = None
        self.ball_color_samples = []
        self.manual_reference = None

        self._photo = None
        self.pil_image = optional_import("PIL.Image")
        self.pil_imagetk = optional_import("PIL.ImageTk")
        
        self._build_layout()

        self.is_analyzing = False # Steuert, ob nur Vorschau oder harte KI-Analyse läuft

    def _build_layout(self):
        frame_top = ttk.Frame(self.root, padding=10)
        frame_top.pack(fill=tk.X)

        ttk.Radiobutton(frame_top, text="Webcam", value="webcam", variable=self.source_var).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(frame_top, text="Datei", value="file", variable=self.source_var).pack(side=tk.LEFT, padx=4)
        ttk.Entry(frame_top, textvariable=self.file_var, width=35).pack(side=tk.LEFT, padx=(15, 4))
        ttk.Button(frame_top, text="Datei wählen", command=self._pick_file).pack(side=tk.LEFT)

        ttk.Button(frame_top, text="Start", command=self.start).pack(side=tk.LEFT, padx=(15, 4))
        ttk.Button(frame_top, text="Stop", command=self.stop).pack(side=tk.LEFT)
        self.btn_pause = ttk.Button(frame_top, text="Pause", command=self._toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_analyse_starten = ttk.Button(frame_top, text="🚀 Analyse Starten", command=self.start_full_analysis)
        self.btn_analyse_starten.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame_top, text="🎬 Export Video", command=self._export_video).pack(side=tk.LEFT, padx=(15, 4))

        self.btn_back = ttk.Button(frame_top, text="<< 1s", command=lambda: self._step(-30))
        self.btn_back.pack(side=tk.LEFT, padx=2)
        self.btn_step_back = ttk.Button(frame_top, text="< 1 Frame", command=lambda: self._step(-1))
        self.btn_step_back.pack(side=tk.LEFT, padx=2)
        self.btn_step_fwd = ttk.Button(frame_top, text="1 Frame >", command=lambda: self._step(1))
        self.btn_step_fwd.pack(side=tk.LEFT, padx=2)
        self.btn_fwd = ttk.Button(frame_top, text="1s >>", command=lambda: self._step(30))
        self.btn_fwd.pack(side=tk.LEFT, padx=2)
        
        frame_timeline = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frame_timeline.pack(fill=tk.X)
        self.lbl_timeline = ttk.Label(frame_timeline, text="Frame: 0 / 0", font=("Helvetica", 10, "bold"))
        self.lbl_timeline.pack(side=tk.LEFT, padx=(0, 10))
        self.timeline_slider = ttk.Scale(frame_timeline, from_=0, to=100, variable=self.timeline_var, orient="horizontal")
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.timeline_slider.bind("<ButtonRelease-1>", self._on_timeline_seek)

        self.frame_right_container = ttk.Frame(self.root, width=280)
        self.frame_right_container.pack_propagate(False) 
        self.frame_right_container.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.notebook = ttk.Notebook(self.frame_right_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_report = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_settings, text="⚙️ Einstellungen")
        self.notebook.add(self.tab_report, text="📝 Bericht")

        self.canvas_settings = tk.Canvas(self.tab_settings, highlightthickness=0)
        self.scrollbar_settings = ttk.Scrollbar(self.tab_settings, orient="vertical", command=self.canvas_settings.yview)
        self.frame_settings = ttk.Frame(self.canvas_settings)
        
        self.frame_settings.bind(
            "<Configure>",
            lambda e: self.canvas_settings.configure(scrollregion=self.canvas_settings.bbox("all"))
        )
        self.canvas_settings.create_window((0, 0), window=self.frame_settings, anchor="nw", width=250)
        self.canvas_settings.configure(yscrollcommand=self.scrollbar_settings.set)
        
        self.canvas_settings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_settings.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            if sys.platform == "win32":
                self.canvas_settings.yview_scroll(-1 * int((event.delta / 120)), "units")
            else:
                self.canvas_settings.yview_scroll(-1 if event.num == 4 else 1, "units")
                
        self.canvas_settings.bind("<Enter>", lambda e: self.canvas_settings.bind_all("<MouseWheel>", _on_mousewheel))
        self.canvas_settings.bind("<Leave>", lambda e: self.canvas_settings.unbind_all("<MouseWheel>"))

        tf_ball = ToggledFrame(self.frame_settings, text="1. Ball Filter & Tracker", expanded=False)
        tf_ball.pack(fill=tk.X, pady=5)
        ttk.Button(tf_ball.sub_frame, text="🔳 Spieler-Box (ROI) ziehen", command=self._start_player_bbox_marking).pack(fill=tk.X, pady=2)
        ttk.Button(tf_ball.sub_frame, text="🎯 Ball manuell markieren", command=self._start_ball_tracker).pack(fill=tk.X, pady=2)
        ttk.Button(tf_ball.sub_frame, text="Tracker zurücksetzen", command=self._reset_ball_tracker).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tf_ball.sub_frame, text="🎨 Ballfarbe sampeln", command=self._start_ball_color_sampling).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tf_ball.sub_frame, text="Farbfilter löschen", command=self._clear_ball_color_filter).pack(fill=tk.X, pady=(0, 5))
        self.lbl_ball_color_status = ttk.Label(tf_ball.sub_frame, text="Kein Farbfilter", foreground="gray")
        self.lbl_ball_color_status.pack(anchor="w", pady=(0, 8))

        tf_mog2 = ToggledFrame(tf_ball.sub_frame, text="MOG2 & Filter", expanded=False)
        tf_mog2.pack(fill=tk.X, pady=(4, 2))
        ttk.Checkbutton(tf_mog2.sub_frame, text="MOG2 Filter an", variable=self.use_mog2_var).pack(anchor="w", pady=(0, 5))
        lbl_min_rad = ttk.Label(tf_mog2.sub_frame, text=f"Min. Radius: {self.ball_min_rad_var.get()} px")
        lbl_min_rad.pack(anchor="w")
        ttk.Scale(tf_mog2.sub_frame, from_=2, to=80, variable=self.ball_min_rad_var, command=lambda v: lbl_min_rad.configure(text=f"Min. Radius: {int(float(v))} px")).pack(fill=tk.X)
        lbl_max_rad = ttk.Label(tf_mog2.sub_frame, text=f"Max. Radius: {self.ball_max_rad_var.get()} px")
        lbl_max_rad.pack(anchor="w")
        ttk.Scale(tf_mog2.sub_frame, from_=6, to=120, variable=self.ball_max_rad_var, command=lambda v: lbl_max_rad.configure(text=f"Max. Radius: {int(float(v))} px")).pack(fill=tk.X)
        lbl_s_max = ttk.Label(tf_mog2.sub_frame, text=f"Max. Sättigung: {self.ball_s_max_var.get()}")
        lbl_s_max.pack(anchor="w", pady=(5,0))
        ttk.Scale(tf_mog2.sub_frame, from_=10, to=255, variable=self.ball_s_max_var, command=lambda v: lbl_s_max.configure(text=f"Max. Sättigung: {int(float(v))}")).pack(fill=tk.X)
        lbl_v_min = ttk.Label(tf_mog2.sub_frame, text=f"Min. Helligkeit: {self.ball_v_min_var.get()}")
        lbl_v_min.pack(anchor="w")
        ttk.Scale(tf_mog2.sub_frame, from_=0, to=200, variable=self.ball_v_min_var, command=lambda v: lbl_v_min.configure(text=f"Min. Helligkeit: {int(float(v))}")).pack(fill=tk.X)
        
        tf_flight = ToggledFrame(self.frame_settings, text="2. Flugbahn & Speed-Fenster", expanded=False)
        tf_flight.pack(fill=tk.X, pady=5)
        ttk.Button(tf_flight.sub_frame, text="Flugfeld markieren", command=self._start_roi_marking).pack(fill=tk.X, pady=2)
        ttk.Button(tf_flight.sub_frame, text="Flugfeld zurücksetzen", command=self._reset_roi).pack(fill=tk.X)
        self.lbl_roi_status = ttk.Label(tf_flight.sub_frame, text="Kein Flugfeld (gesamtes Bild)", foreground="green")
        self.lbl_roi_status.pack(anchor="w", pady=(5, 0))

        ttk.Label(tf_flight.sub_frame, text="Speed nur von Frame ... bis ...:").pack(anchor="w", pady=(8, 2))
        row_speed = ttk.Frame(tf_flight.sub_frame)
        row_speed.pack(fill=tk.X)
        ttk.Label(row_speed, text="Von:").pack(side=tk.LEFT)
        self.entry_speed_start_frame = ttk.Entry(row_speed, textvariable=self.speed_start_frame_var, width=8)
        self.entry_speed_start_frame.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row_speed, text="Bis:").pack(side=tk.LEFT)
        self.entry_speed_end_frame = ttk.Entry(row_speed, textvariable=self.speed_end_frame_var, width=8)
        self.entry_speed_end_frame.pack(side=tk.LEFT)
        ttk.Button(tf_flight.sub_frame, text="Speed-Fenster übernehmen", command=self._apply_speed_frame_range_inputs).pack(fill=tk.X, pady=(5, 0))
        self.entry_speed_start_frame.bind("<Return>", self._apply_speed_frame_range_inputs)
        self.entry_speed_end_frame.bind("<Return>", self._apply_speed_frame_range_inputs)
        self.entry_speed_start_frame.bind("<FocusOut>", self._apply_speed_frame_range_inputs)
        self.entry_speed_end_frame.bind("<FocusOut>", self._apply_speed_frame_range_inputs)
        ttk.Button(tf_flight.sub_frame, text="Speed-Start = aktueller Frame", command=self._set_speed_start_to_current).pack(fill=tk.X, pady=(5, 2))
        ttk.Button(tf_flight.sub_frame, text="Speed-Ende = aktueller Frame", command=self._set_speed_end_to_current).pack(fill=tk.X, pady=2)
        ttk.Button(tf_flight.sub_frame, text="Speed = ganzer Videobereich", command=self._reset_speed_range).pack(fill=tk.X, pady=(2, 5))

        tf_calib = ToggledFrame(self.frame_settings, text="3. Kalibrierung", expanded=False)
        tf_calib.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(tf_calib.sub_frame, text="Körpergröße (Auto, dann fix)", variable=self.calibration_mode_var, value="body_height", command=self._on_calibration_mode_change).pack(anchor="w")
        ttk.Radiobutton(tf_calib.sub_frame, text="Manuelle Referenzlinie", variable=self.calibration_mode_var, value="manual_line", command=self._on_calibration_mode_change).pack(anchor="w", pady=(0, 4))
        self.lbl_ref_length = ttk.Label(tf_calib.sub_frame, text="Körpergröße (m):")
        self.lbl_ref_length.pack(anchor="w")
        ttk.Entry(tf_calib.sub_frame, textvariable=self.real_length_var).pack(fill=tk.X, pady=(0, 5))
        self.btn_set_manual_ref = ttk.Button(tf_calib.sub_frame, text="Referenzlinie markieren", command=self._start_manual_reference)
        self.btn_set_manual_ref.pack(fill=tk.X, pady=(0, 5))
        self.lbl_ref_mode_hint = ttk.Label(tf_calib.sub_frame, text="Auto über Skelett: Nase -> Fuß", foreground="gray")
        self.lbl_ref_mode_hint.pack(anchor="w", pady=(0, 4))
        self.lbl_calib_status = ttk.Label(tf_calib.sub_frame, text="Nicht kalibriert", foreground="red")
        self.lbl_calib_status.pack(anchor="w", pady=(0, 10))
        lbl_target = ttk.Label(tf_calib.sub_frame, text=f"Ziel-Geschwindigkeit: {self.target_speed_var.get():.0f} km/h")
        lbl_target.pack(anchor="w")
        ttk.Scale(tf_calib.sub_frame, from_=30.0, to=140.0, variable=self.target_speed_var, command=lambda v: lbl_target.configure(text=f"Ziel-Geschwindigkeit: {float(v):.0f} km/h")).pack(fill=tk.X)
        
        tf_time = ToggledFrame(self.frame_settings, text="4. Loop", expanded=False)
        tf_time.pack(fill=tk.X, pady=5)
        row1 = ttk.Frame(tf_time.sub_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Start:").pack(side=tk.LEFT)
        self.entry_start_frame = ttk.Entry(row1, textvariable=self.start_frame_var, width=8)
        self.entry_start_frame.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row1, text="Ende:").pack(side=tk.LEFT)
        self.entry_end_frame = ttk.Entry(row1, textvariable=self.end_frame_var, width=8)
        self.entry_end_frame.pack(side=tk.LEFT)
        row2 = ttk.Frame(tf_time.sub_frame)
        row2.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(row2, text="Bereich übernehmen", command=self._apply_frame_range_inputs).pack(fill=tk.X)
        self.entry_start_frame.bind("<Return>", self._apply_frame_range_inputs)
        self.entry_end_frame.bind("<Return>", self._apply_frame_range_inputs)
        self.entry_start_frame.bind("<FocusOut>", self._apply_frame_range_inputs)
        self.entry_end_frame.bind("<FocusOut>", self._apply_frame_range_inputs)
        ttk.Button(tf_time.sub_frame, text="Start = aktueller Frame", command=self._set_start_to_current).pack(fill=tk.X, pady=(5, 2))
        ttk.Button(tf_time.sub_frame, text="Ende = aktueller Frame", command=self._set_end_to_current).pack(fill=tk.X, pady=2)
        ttk.Button(tf_time.sub_frame, text="Bereich = ganzes Video", command=self._reset_loop_range).pack(fill=tk.X, pady=(2, 5))
        ttk.Checkbutton(tf_time.sub_frame, text="Loop in Bereich", variable=self.loop_var).pack(anchor="w", pady=(5,0))
        ttk.Checkbutton(tf_time.sub_frame, text="2-Pass Voranalyse (Datei)", variable=self.use_two_pass_var).pack(anchor="w", pady=(5,0))

        tf_ki = ToggledFrame(self.frame_settings, text="5. Skelett (KI)", expanded=False)
        tf_ki.pack(fill=tk.X, pady=5)
        ttk.Label(tf_ki.sub_frame, text="Backend:").pack(anchor="w")
        ttk.Combobox(tf_ki.sub_frame, textvariable=self.pose_backend_var, values=("auto", "mediapipe", "mmpose"), state="readonly").pack(fill=tk.X, pady=(0,5))
        
        lbl_kpt = ttk.Label(tf_ki.sub_frame, text=f"Sicherheit: {self.kpt_thr_var.get():.2f}")
        lbl_kpt.pack(anchor="w")
        ttk.Scale(tf_ki.sub_frame, from_=0.0, to=1.0, variable=self.kpt_thr_var, command=lambda v: lbl_kpt.configure(text=f"Sicherheit: {float(v):.2f}")).pack(fill=tk.X)

        ttk.Button(self.tab_report, text="Generieren", command=self._generate_report).pack(side=tk.TOP, pady=10, padx=10, fill=tk.X)
        self.txt_report = tk.Text(self.tab_report, wrap=tk.WORD, font=("Helvetica", 10))
        self.txt_report.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        frame_center = ttk.Frame(self.root, padding=10)
        frame_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frame_metrics = ttk.Frame(frame_center)
        frame_metrics.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(frame_metrics, text="Speed (km/h):").grid(row=0, column=0, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_speed, font=("Helvetica", 12, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 15))
        ttk.Label(frame_metrics, text="Rumpf (°):").grid(row=0, column=2, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_trunk).grid(row=0, column=3, sticky="w", padx=(0, 15))
        ttk.Label(frame_metrics, text="Schulter (°):").grid(row=0, column=4, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_shoulder).grid(row=0, column=5, sticky="w", padx=(0, 15))
        ttk.Label(frame_metrics, text="Arm (°):").grid(row=0, column=6, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_elbow).grid(row=0, column=7, sticky="w", padx=(0, 15))
        ttk.Label(frame_metrics, text="Knie (°):").grid(row=0, column=8, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_knee).grid(row=0, column=9, sticky="w", padx=(0, 15))
        ttk.Label(frame_metrics, text="Punkte:").grid(row=0, column=10, sticky="w")
        ttk.Label(frame_metrics, textvariable=self.metric_score, font=("Helvetica", 14, "bold"), foreground="blue").grid(row=0, column=11, sticky="w")

        self.view_notebook = ttk.Notebook(frame_center)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_labels = []
        tab_names = ["1. Übersicht (4er)", "2. Original", "3. Ball Filter", "4. Skelett", "5. Kombiniert"]
        for text in tab_names:
            f = ttk.Frame(self.view_notebook)
            self.view_notebook.add(f, text=text)
            lbl = ttk.Label(f, text="Wähle Quelle und klicke Start", anchor="center", background="black", foreground="white")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.tab_labels.append(lbl)
            
        self.lbl_export_status = ttk.Label(frame_top, textvariable=self.status_var, foreground="blue")
        self.lbl_export_status.pack(side=tk.LEFT, padx=15)
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_calibration_mode_change()

    def _generate_report(self):
        if not self.engine or not self.engine.per_track_state:
            self.txt_report.delete("1.0", tk.END)
            self.txt_report.insert("1.0", "Keine Messdaten vorhanden. Bitte starte das Video zuerst.")
            return
        self.txt_report.delete("1.0", tk.END)
        self.txt_report.insert("1.0", build_detailed_report(self.engine.per_track_state[1].metrics, self.target_speed_var.get()))

    def _get_body_height_m(self) -> Optional[float]:
        try:
            raw = str(self.real_length_var.get()).strip().replace(",", ".")
            value = float(raw)
        except (ValueError, tk.TclError):
            return None
        return value if value > 0 else None

    def _on_calibration_mode_change(self):
        mode = self.calibration_mode_var.get()
        if mode == "manual_line":
            self.lbl_ref_length.configure(text="Referenzlänge (m):")
            self.lbl_ref_mode_hint.configure(text="Setze zwei Punkte auf ein bekanntes Maß", foreground="gray")
            self.btn_set_manual_ref.configure(state=tk.NORMAL)
        else:
            self.lbl_ref_length.configure(text="Körpergröße (m):")
            self.lbl_ref_mode_hint.configure(text="Auto über Skelett: Nase -> Fuß", foreground="gray")
            self.btn_set_manual_ref.configure(state=tk.DISABLED)

        if self.engine:
            self.engine.set_calibration_mode(mode)
            if mode == "manual_line" and self.manual_reference is not None:
                pt1, pt2, ref_len = self.manual_reference
                self.engine.set_manual_reference(pt1, pt2, ref_len)
            elif mode == "body_height":
                self.engine.reset_calibration(keep_mode=True)
        self._update_calibration_status()

    def _apply_runtime_preferences_to_engine(self):
        if not self.engine:
            return
        self.engine.set_calibration_mode(self.calibration_mode_var.get())
        if self.ball_color_samples:
            self.engine.set_ball_color_samples(self.ball_color_samples)
        else:
            self.engine.clear_ball_color_filter()
        if self.calibration_mode_var.get() == "manual_line" and self.manual_reference is not None:
            pt1, pt2, ref_len = self.manual_reference
            self.engine.set_manual_reference(pt1, pt2, ref_len)

    def _update_calibration_status(self):
        mode = self.calibration_mode_var.get()
        ref_len = self._get_body_height_m()

        if mode == "manual_line":
            if ref_len is None:
                self.lbl_calib_status.configure(text="Bitte gültige Referenzlänge eingeben", foreground="orange")
                return
            if self.engine and self.engine.pixels_per_meter:
                self.lbl_calib_status.configure(text=f"Manuell kalibriert: {self.engine.pixels_per_meter:.1f} px/m", foreground="green")
            elif self.manual_reference is not None:
                self.lbl_calib_status.configure(text="Referenz gesetzt, warte auf Video-Start", foreground="blue")
            else:
                self.lbl_calib_status.configure(text="Bitte Referenzlinie markieren", foreground="red")
            return

        if ref_len is None:
            self.lbl_calib_status.configure(text="Bitte gültige Körpergröße eingeben", foreground="orange")
            return

        if self.engine and self.engine.pixels_per_meter:
            sample_count, sample_target, is_locked = self.engine.get_calibration_progress()
            if is_locked:
                self.lbl_calib_status.configure(text=f"Auto kalibriert (fix): {self.engine.pixels_per_meter:.1f} px/m", foreground="green")
            else:
                self.lbl_calib_status.configure(
                    text=f"Kalibriere... {sample_count}/{sample_target} Frames ({self.engine.pixels_per_meter:.1f} px/m)",
                    foreground="blue",
                )
        else:
            self.lbl_calib_status.configure(text="Warte auf Kopf+Fuß im Bild", foreground="red")

    def _pick_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi"), ("Alle", "*.*")])
        if path:
            if self.running:
                self.stop()
            self.file_var.set(path)
            self.source_var.set("file")
            self.status_var.set("Datei gewählt. Mit Start neu laden.")

    def _open_source(self):
        ensure_cv2()
        self.cap, source_name = select_video_capture(self.source_var.get(), self.file_var.get().strip(), self.camera_var.get())
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.seekable = self.source_var.get() == "file"
        self.video_path = self.file_var.get().strip() if self.seekable else None
        self.is_analyzing = False
        selected_backend = self.pose_backend_var.get()
        if selected_backend == "none":
            fallback_backend = pick_default_pose_backend()
            if fallback_backend != "none":
                selected_backend = fallback_backend
                self.pose_backend_var.set(fallback_backend)
                self.status_var.set(f"Pose-Backend automatisch auf '{fallback_backend}' gesetzt.")
        
        if self.seekable:
            self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.timeline_slider.configure(to=max(0, self.total_frames - 1))
            self.end_frame_var.set(str(max(0, self.total_frames - 1)))
            self._apply_speed_frame_range_inputs(update_status=False)
            
        self._update_controls()
        self.report_generated_auto = False
        self.preanalysis_pending = self.seekable and self.use_two_pass_var.get()
        self.preanalysis_running = False
        self.current_frame = 0
        self.paused = False
        self.btn_pause.configure(text="Pause")

        self.engine = AnalysisEngine(
            fps=self.fps,
            reference_seq=self.reference_seq,
            pose_backend=selected_backend,
            calibration_mode=self.calibration_mode_var.get(),
        )
        active_mode = getattr(self.engine.pose_estimator, "mode", "none")
        if active_mode == "none":
            fallback_backend = pick_default_pose_backend()
            if fallback_backend != "none" and fallback_backend != selected_backend:
                self.engine = AnalysisEngine(
                    fps=self.fps,
                    reference_seq=self.reference_seq,
                    pose_backend=fallback_backend,
                    calibration_mode=self.calibration_mode_var.get(),
                )
                active_mode = getattr(self.engine.pose_estimator, "mode", "none")
                self.pose_backend_var.set(fallback_backend)
                self.status_var.set(f"Pose-Backend '{selected_backend}' nicht verfügbar. Fallback auf '{fallback_backend}'.")
            else:
                self.status_var.set("Kein Pose-Backend aktiv. Bitte mediapipe oder mmpose installieren.")
        else:
            self.status_var.set(f"Quelle geöffnet. Pose-Backend aktiv: {active_mode}")

        self._apply_runtime_preferences_to_engine()
        if self.ball_color_samples:
            self.lbl_ball_color_status.configure(text=f"Farbfilter aktiv ({len(self.ball_color_samples)} Punkte)", foreground="green")
        else:
            self.lbl_ball_color_status.configure(text="Kein Farbfilter", foreground="gray")
        self._update_calibration_status()
        
    def _update_controls(self):
        state = tk.NORMAL if self.seekable else tk.DISABLED
        self.btn_back.configure(state=state)
        self.btn_step_back.configure(state=state)
        self.btn_step_fwd.configure(state=state)
        self.btn_fwd.configure(state=state)
        self.timeline_slider.configure(state=state)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.configure(text="Play" if self.paused else "Pause")

    def _get_frame_range(self):
        try:
            start_f = int(self.start_frame_var.get())
        except ValueError:
            start_f = 0
        try:
            end_f = int(self.end_frame_var.get())
        except ValueError:
            end_f = 999999
        if self.seekable:
            start_f = max(0, min(start_f, self.total_frames - 1))
            end_f = max(0, min(end_f, self.total_frames - 1))
            if end_f < start_f:
                end_f = start_f
        return start_f, end_f

    def _get_speed_frame_range(self):
        try:
            start_f = int(self.speed_start_frame_var.get())
        except ValueError:
            start_f = 0
        try:
            end_f = int(self.speed_end_frame_var.get())
        except ValueError:
            end_f = 999999
        if self.seekable:
            start_f = max(0, min(start_f, self.total_frames - 1))
            end_f = max(0, min(end_f, self.total_frames - 1))
            if end_f < start_f:
                end_f = start_f
        return start_f, end_f

    def _apply_frame_range_inputs(self, event=None):
        start_f, end_f = self._get_frame_range()
        self.start_frame_var.set(str(start_f))
        self.end_frame_var.set(str(end_f))
        self.status_var.set(f"Loop-Bereich: {start_f} bis {end_f}")

        if self.seekable and self.cap:
            target = max(start_f, min(self.current_frame, end_f))
            if target != self.current_frame:
                self._show_frame_at(target)

    def _apply_speed_frame_range_inputs(self, event=None, update_status: bool = True):
        start_f, end_f = self._get_speed_frame_range()
        self.speed_start_frame_var.set(str(start_f))
        self.speed_end_frame_var.set(str(end_f))
        if update_status:
            self.status_var.set(f"Speed-Fenster: {start_f} bis {end_f}")

    def _set_speed_start_to_current(self):
        if not self.seekable:
            return
        current = int(max(0, self.current_frame))
        self.speed_start_frame_var.set(str(current))
        try:
            end_f = int(self.speed_end_frame_var.get())
        except ValueError:
            end_f = current
        if current > end_f:
            self.speed_end_frame_var.set(str(current))
        self.status_var.set(f"Speed-Start gesetzt: {current}")
        self._apply_speed_frame_range_inputs()

    def _set_speed_end_to_current(self):
        if not self.seekable:
            return
        current = int(max(0, self.current_frame))
        self.speed_end_frame_var.set(str(current))
        try:
            start_f = int(self.speed_start_frame_var.get())
        except ValueError:
            start_f = current
        if current < start_f:
            self.speed_start_frame_var.set(str(current))
        self.status_var.set(f"Speed-Ende gesetzt: {current}")
        self._apply_speed_frame_range_inputs()

    def _reset_speed_range(self):
        if not self.seekable:
            return
        self.speed_start_frame_var.set("0")
        self.speed_end_frame_var.set(str(max(0, self.total_frames - 1)))
        self.status_var.set("Speed-Fenster auf ganzes Video gesetzt.")
        self._apply_speed_frame_range_inputs()

    def _set_start_to_current(self):
        if not self.seekable:
            return
        current = int(max(0, self.current_frame))
        self.start_frame_var.set(str(current))
        try:
            end_f = int(self.end_frame_var.get())
        except ValueError:
            end_f = current
        if current > end_f:
            self.end_frame_var.set(str(current))
        self.status_var.set(f"Start-Frame gesetzt: {current}")
        self._apply_frame_range_inputs()

    def _set_end_to_current(self):
        if not self.seekable:
            return
        current = int(max(0, self.current_frame))
        self.end_frame_var.set(str(current))
        try:
            start_f = int(self.start_frame_var.get())
        except ValueError:
            start_f = current
        if current < start_f:
            self.start_frame_var.set(str(current))
        self.status_var.set(f"End-Frame gesetzt: {current}")
        self._apply_frame_range_inputs()

    def _reset_loop_range(self):
        if not self.seekable:
            return
        self.start_frame_var.set("0")
        self.end_frame_var.set(str(max(0, self.total_frames - 1)))
        self.status_var.set("Loop-Bereich auf ganzes Video gesetzt.")
        self._apply_frame_range_inputs()

    def _update_metrics_from_result(self, metrics, score):
        if not metrics:
            self.metric_speed.set("-")
            self.metric_trunk.set("-")
            self.metric_shoulder.set("-")
            self.metric_elbow.set("-")
            self.metric_knee.set("-")
            self.metric_score.set("- / 100")
            return
        self.metric_speed.set(f"{metrics.wrist_speed_kmph:.2f}")
        self.metric_trunk.set(f"{metrics.trunk_inclination_deg:.2f}")
        self.metric_shoulder.set(f"{metrics.shoulder_angle_deg:.2f}")
        self.metric_elbow.set(f"{metrics.elbow_angle_deg:.2f}")
        self.metric_knee.set(f"{metrics.knee_angle_deg:.2f}")
        self.metric_score.set(f"{score} / 100")

    def _process_frame_and_refresh(self, frame, frame_idx: int):
        if not self.engine:
            return
        speed_start_f, speed_end_f = self._get_speed_frame_range()
        metrics, _, _, debug_frames, score = self.engine.process_frame_with_pose(
            frame,
            frame_idx / self.fps,
            self.kpt_thr_var.get(),
            self.ball_s_max_var.get(),
            self.ball_v_min_var.get(),
            self.hough_p1_var.get(),
            self.hough_p2_var.get(),
            self.roi_polygon,
            self.target_speed_var.get(),
            self.use_mog2_var.get(),
            self.ball_min_rad_var.get(),
            self.ball_max_rad_var.get(),
            body_height_m=self._get_body_height_m(),
            frame_index=frame_idx,
            speed_start_frame=speed_start_f,
            speed_end_frame=speed_end_f,
        )
        self.current_frame = frame_idx
        self._update_metrics_from_result(metrics, score)
        self._update_calibration_status()
        self.last_frames = debug_frames

    def _show_frame_at(self, frame_idx: int):
        if not self.cap or not self.seekable:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if ok:
            if not self.is_analyzing:
                display_frame = frame.copy()
                if self.engine and getattr(self.engine, "player_bbox", None):
                    x, y, w, h = self.engine.player_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Analyse-Bereich",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                self.last_frames = {
                    "orig": display_frame,
                    "mask": np.zeros_like(display_frame),
                    "skeleton": display_frame.copy(),
                    "final": display_frame.copy(),
                }
                self.current_frame = frame_idx
            else:
                self._process_frame_and_refresh(frame, frame_idx)
            self._refresh_preview()
        
    def _on_timeline_seek(self, event):
        if self.cap and self.seekable: 
            self._step(0)

    def _step(self, delta: int):
        if not self.cap or not self.seekable: 
            return
        self.paused = True
        self.btn_pause.configure(text="Play")
        if delta == 0:
            target = int(self.timeline_var.get())
        else:
            target = self.current_frame + delta
        self._show_frame_at(target)

    def _start_ball_tracker(self):
        if not self.last_frames or not self.engine:
            messagebox.showwarning("Achtung", "Bitte starte das Video und pausiere es zuerst.")
            return
        self.paused = True
        self.btn_pause.configure(text="Play")
        threading.Thread(target=self._tracker_thread, daemon=True).start()

    def _tracker_thread(self):
        clone = self.last_frames["orig"].copy()
        win_name = "Ball markieren (ENTER zum Bestaetigen)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1) 
        
        bbox = cv2.selectROI(win_name, clone, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win_name)
        
        if bbox and bbox[2] > 0 and bbox[3] > 0:
            self.engine.init_ball_tracker(self.last_frames["orig"], bbox)
            self.root.after(0, lambda: messagebox.showinfo("Tracker", "Ball wird nun manuell verfolgt!"))

    def _reset_ball_tracker(self):
        if self.engine:
            self.engine.reset_ball_tracker()
            messagebox.showinfo("Tracker", "Tracker deaktiviert. Nutze wieder automatische Filter.")

    def _start_ball_color_sampling(self):
        if not self.last_frames:
            messagebox.showwarning("Achtung", "Bitte starte das Video und pausiere es zuerst.")
            return
        self.paused = True
        self.btn_pause.configure(text="Play")
        threading.Thread(target=self._ball_color_thread, daemon=True).start()

    def _ball_color_thread(self):
        frame = self.last_frames["orig"].copy()
        preview = frame.copy()
        points = []
        win_name = "Ballfarbe: Klicks setzen, ENTER bestaetigen, ESC abbrechen"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(preview, (x, y), 4, (0, 255, 255), -1)
                cv2.imshow(win_name, preview)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(win_name, on_click)

        while True:
            cv2.imshow(win_name, preview)
            key = cv2.waitKey(15) & 0xFF
            if key == 13:  # ENTER
                break
            if key == 27:  # ESC
                points = []
                break
        cv2.destroyWindow(win_name)

        if len(points) < 3:
            self.root.after(0, lambda: messagebox.showwarning("Farbfilter", "Bitte mindestens 3 Punkte auf dem Ball setzen."))
            return

        h, w = frame.shape[:2]
        samples = []
        for x, y in points:
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            b, g, r = frame[y, x]
            samples.append((int(b), int(g), int(r)))

        self.ball_color_samples = samples
        if self.engine:
            self.engine.set_ball_color_samples(samples)
        self.root.after(0, lambda: self.lbl_ball_color_status.configure(text=f"Farbfilter aktiv ({len(samples)} Punkte)", foreground="green"))

    def _clear_ball_color_filter(self):
        self.ball_color_samples = []
        if self.engine:
            self.engine.clear_ball_color_filter()
        self.lbl_ball_color_status.configure(text="Kein Farbfilter", foreground="gray")

    def _start_manual_reference(self):
        if not self.last_frames:
            messagebox.showwarning("Achtung", "Bitte starte das Video und pausiere es zuerst.")
            return
        self.paused = True
        self.btn_pause.configure(text="Play")
        threading.Thread(target=self._manual_reference_thread, daemon=True).start()

    def _manual_reference_thread(self):
        frame = self.last_frames["orig"].copy()
        preview = frame.copy()
        points = []
        win_name = "Referenzlinie: 2 Punkte klicken, ENTER bestaetigen"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(preview, (x, y), 6, (255, 0, 255), 2)
                if len(points) == 2:
                    cv2.line(preview, points[0], points[1], (255, 0, 255), 2)
                cv2.imshow(win_name, preview)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(win_name, on_click)

        while True:
            cv2.imshow(win_name, preview)
            key = cv2.waitKey(15) & 0xFF
            if key == 13:  # ENTER
                break
            if key == 27:  # ESC
                points = []
                break
        cv2.destroyWindow(win_name)

        if len(points) != 2:
            return

        ref_len = self._get_body_height_m()
        if ref_len is None:
            self.root.after(0, lambda: messagebox.showerror("Kalibrierung", "Bitte zuerst eine gültige Referenzlänge eingeben."))
            return

        self.manual_reference = (points[0], points[1], ref_len)
        self.root.after(0, lambda: self.calibration_mode_var.set("manual_line"))
        self.root.after(0, self._on_calibration_mode_change)
        if self.engine:
            self.engine.set_calibration_mode("manual_line")
            if self.engine.set_manual_reference(points[0], points[1], ref_len):
                self.root.after(0, self._update_calibration_status)

    def _start_roi_marking(self):
        if not self.last_frames: 
            messagebox.showwarning("Achtung", "Bitte starte das Video und pausiere es zuerst.")
            return
        self.paused = True
        self.btn_pause.configure(text="Play")
        # Stabiler im Hauptthread (wie bei Spieler-ROI), damit OpenCV-Fenster konsistent reagiert.
        self._roi_thread()

    def _roi_thread(self):
        clone = self.last_frames["orig"].copy()
        pts = []
        win_name = "Flugfeld (ENTER zum Beenden)"
        
        def draw_poly(e, x, y, f, p):
            if e == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                cv2.circle(clone, (x, y), 5, (255, 0, 0), -1)
                if len(pts) > 1: 
                    cv2.line(clone, pts[-2], pts[-1], (255, 0, 0), 2)
                cv2.imshow(win_name, clone)
                
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(win_name, draw_poly)
        
        while True:
            cv2.imshow(win_name, clone)
            if cv2.waitKey(10) & 0xFF in [13, 27]:
                break
        cv2.destroyWindow(win_name)

        # Komfort: 2 Klicks als Rechteck-Ecken interpretieren.
        if len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            xa, xb = sorted((int(x1), int(x2)))
            ya, yb = sorted((int(y1), int(y2)))
            if (xb - xa) >= 5 and (yb - ya) >= 5:
                pts = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]

        if len(pts) >= 3:
            self.roi_polygon = pts
            self.root.after(0, lambda: self.lbl_roi_status.configure(text="Flugfeld aktiv", foreground="blue"))
        else:
            self.roi_polygon = []
            self.root.after(0, lambda: self.lbl_roi_status.configure(text="Kein Flugfeld (gesamtes Bild)", foreground="green"))

    def _reset_roi(self):
        self.roi_polygon = []
        self.lbl_roi_status.configure(text="Kein Flugfeld (gesamtes Bild)", foreground="green")

    def _export_video(self):
        if not self.seekable or not self.file_var.get():
            messagebox.showerror("Fehler", "Export geht nur mit Videodateien (nicht bei Webcam-Livebild).")
            return

        source_path = self.file_var.get().strip()
        if not source_path:
            messagebox.showerror("Fehler", "Kein Quellvideo gefunden.")
            return

        name, _ = os.path.splitext(source_path)
        name = re.sub(r"(?:_analysiert)+$", "", name, flags=re.IGNORECASE)
        save_path = f"{name}_analysiert.avi"

        threading.Thread(target=self._export_thread, args=(save_path,), daemon=True).start()

    def _export_thread(self, save_path):
        self.root.after(0, lambda: self.status_var.set("Export... Phase 1/2: Analysiere Video..."))
        
        cap = cv2.VideoCapture(self.file_var.get().strip())
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Fehler", "Quellvideo konnte für Export nicht geöffnet werden."))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        export_seconds = 3.0
        target_frames = max(1, int(round(fps * export_seconds)))
        speed_start_f, speed_end_f = self._get_speed_frame_range()
        start_f, end_f = self._get_frame_range()
        
        # Engine für Phase 1 (Daten sammeln)
        export_engine = AnalysisEngine(
            fps=fps,
            reference_seq=self.reference_seq,
            pose_backend=self.pose_backend_var.get(),
            calibration_mode=self.calibration_mode_var.get(),
        )
        if self.ball_color_samples:
            export_engine.set_ball_color_samples(self.ball_color_samples)
        if self.calibration_mode_var.get() == "manual_line" and self.manual_reference is not None:
            pt1, pt2, ref_len = self.manual_reference
            export_engine.set_manual_reference(pt1, pt2, ref_len)
        
        # Übertrage Kalibrierung und visuelle Punkte auf die Export-Engine
        if self.engine:
            if self.engine.pixels_per_meter:
                export_engine.pixels_per_meter = self.engine.pixels_per_meter
            if hasattr(self.engine, 'calib_points'):
                export_engine.calib_points = self.engine.calib_points
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        source_frame_indices = []
        
        # PHASE 1: Analyse des gewählten Bereichs
        while cap.isOpened():
            frame_abs_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_abs_idx > end_f:
                break
            ok, frame = cap.read()
            if not ok:
                break
            src_idx = len(source_frame_indices)
            source_frame_indices.append(frame_abs_idx)
                
            export_engine.process_frame_with_pose(
                frame, src_idx / fps, self.kpt_thr_var.get(), 
                self.ball_s_max_var.get(), self.ball_v_min_var.get(), self.hough_p1_var.get(), 
                self.hough_p2_var.get(), self.roi_polygon, self.target_speed_var.get(), 
                self.use_mog2_var.get(), self.ball_min_rad_var.get(), self.ball_max_rad_var.get(),
                body_height_m=self._get_body_height_m(),
                frame_index=frame_abs_idx,
                speed_start_frame=speed_start_f,
                speed_end_frame=speed_end_f,
            )
            if len(source_frame_indices) % 10 == 0:
                self.root.after(0, lambda idx=frame_abs_idx: self.status_var.set(f"Phase 1/2: Analysiere Frame {idx}..."))
        analyzed_frames = len(source_frame_indices)
        if analyzed_frames == 0:
            cap.release()
            self.root.after(0, lambda: self.status_var.set("Bereit"))
            self.root.after(0, lambda: messagebox.showerror("Fehler", "Kein Frame im gewählten Bereich für Export gefunden."))
            return
                
        # Nach Phase 1: Bericht generieren und im GUI anzeigen
        report_text = build_detailed_report(export_engine.per_track_state[1].metrics, self.target_speed_var.get())
        self.root.after(0, lambda: self.txt_report.delete("1.0", tk.END))
        self.root.after(0, lambda: self.txt_report.insert("1.0", report_text))
        
        # PHASE 2: Video rendern (mit fertigem Text)
        self.root.after(0, lambda: self.status_var.set("Export... Phase 2/2: Erstelle AVI-Video..."))
        
        text_w = 600
        out_w = w + text_w
        out_h = max(h, 720) 
        text_panel = create_text_panel(report_text, text_w, out_h)
        
        # Codec auf XVID (AVI) geändert
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            cap.release()
            self.root.after(0, lambda: self.status_var.set("Bereit"))
            self.root.after(0, lambda: messagebox.showerror("Fehler", "Zieldatei konnte nicht geschrieben werden."))
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        export_engine_render = AnalysisEngine(
            fps=fps,
            reference_seq=self.reference_seq,
            pose_backend=self.pose_backend_var.get(),
            calibration_mode=self.calibration_mode_var.get(),
        )
        if self.ball_color_samples:
            export_engine_render.set_ball_color_samples(self.ball_color_samples)
        if self.calibration_mode_var.get() == "manual_line" and self.manual_reference is not None:
            pt1, pt2, ref_len = self.manual_reference
            export_engine_render.set_manual_reference(pt1, pt2, ref_len)
        if self.engine and self.engine.pixels_per_meter:
            export_engine_render.pixels_per_meter = self.engine.pixels_per_meter
        if self.engine and hasattr(self.engine, 'calib_points'):
            export_engine_render.calib_points = self.engine.calib_points

        if analyzed_frames == 1:
            repeat_counts = [target_frames]
        else:
            repeat_counts = [0] * analyzed_frames
            denom = max(1, target_frames - 1)
            for out_idx in range(target_frames):
                src_idx = int(round((out_idx * (analyzed_frames - 1)) / denom))
                repeat_counts[src_idx] += 1

        written_frames = 0
        last_combined = None
        src_idx = 0
        while src_idx < analyzed_frames and written_frames < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
                
            _, _, _, debug_frames, _ = export_engine_render.process_frame_with_pose(
                frame, src_idx / fps, self.kpt_thr_var.get(), 
                self.ball_s_max_var.get(), self.ball_v_min_var.get(), self.hough_p1_var.get(), 
                self.hough_p2_var.get(), self.roi_polygon, self.target_speed_var.get(), 
                self.use_mog2_var.get(), self.ball_min_rad_var.get(), self.ball_max_rad_var.get(),
                body_height_m=self._get_body_height_m(),
                frame_index=source_frame_indices[src_idx],
                speed_start_frame=speed_start_f,
                speed_end_frame=speed_end_f,
            )
            
            final_f = debug_frames["final"]
            if final_f.shape[0] != out_h:
                 final_f = cv2.resize(final_f, (w, out_h))
            
            combined = np.hstack((final_f, text_panel))
            last_combined = combined
            repeats = repeat_counts[src_idx]
            for _ in range(repeats):
                out.write(combined)
                written_frames += 1
                if written_frames % 10 == 0:
                    self.root.after(0, lambda idx=written_frames: self.status_var.set(f"Phase 2/2: Schreibe Frame {idx}..."))
                if written_frames >= target_frames:
                    break
            src_idx += 1

        # Fallback nur bei Lesefehlern: Ausgabe trotzdem auf exakt 3 Sekunden auffüllen.
        if written_frames < target_frames and last_combined is not None:
            while written_frames < target_frames:
                out.write(last_combined)
                written_frames += 1
            
        cap.release()
        out.release()
        self.root.after(0, lambda: self.status_var.set("Bereit"))
        self.root.after(
            0,
            lambda: messagebox.showinfo(
                "Erfolg",
                f"Video erfolgreich exportiert ({export_seconds:.0f}s):\n{save_path}",
            ),
        )

    def start(self):
        if self.running:
            self.stop()
        try: 
            self._open_source()
        except Exception as exc: 
            messagebox.showerror("Fehler", str(exc))
            return
        if getattr(self.engine.pose_estimator, "mode", "none") == "none":
            self.status_var.set("Kein Pose-Backend aktiv. Bitte Backend prüfen.")
        else:
            self.status_var.set("Vorschau läuft (ohne KI). Für Pose/Winkel: '🚀 Analyse Starten'.")
        self.running = True
        self.loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.loop_thread.start()
        self.root.after(50, self._refresh_preview)

    def stop(self):
        self.running = False
        self.is_analyzing = False
        self.paused = False
        self.preanalysis_pending = False
        self.preanalysis_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.loop_thread and self.loop_thread.is_alive() and self.loop_thread is not threading.current_thread():
            self.loop_thread.join(timeout=1.0)
        self.loop_thread = None
        self.btn_pause.configure(text="Pause")
        self.status_var.set("Bereit")

    def _run_preanalysis_pass(self):
        if not self.seekable or not self.engine or not self.file_var.get().strip():
            return

        self._apply_runtime_preferences_to_engine()
        self.preanalysis_running = True
        self.root.after(0, lambda: self.status_var.set("Pass 1/2: Voranalyse läuft..."))

        start_f, end_f = self._get_frame_range()
        video_path = self.file_var.get().strip()

        if self.calibration_mode_var.get() == "body_height":
            self.engine.reset_calibration(keep_mode=True)
        self.engine.clear_ball_cache()
        self.engine.ball_detector.reset_background_model()

        cap_pre = cv2.VideoCapture(video_path)
        if not cap_pre.isOpened():
            self.preanalysis_running = False
            self.preanalysis_pending = False
            self.root.after(0, lambda: self.status_var.set("Voranalyse übersprungen (Datei nicht lesbar)"))
            return

        cap_pre.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_idx = start_f
        processed = 0

        while cap_pre.isOpened() and self.running:
            ok, frame = cap_pre.read()
            if not ok or frame_idx > end_f:
                break

            self.engine.preanalyze_frame(
                frame=frame,
                frame_index=frame_idx,
                kpt_thr=self.kpt_thr_var.get(),
                s_max=self.ball_s_max_var.get(),
                v_min=self.ball_v_min_var.get(),
                p1=self.hough_p1_var.get(),
                p2=self.hough_p2_var.get(),
                roi_polygon=self.roi_polygon,
                use_mog2=self.use_mog2_var.get(),
                min_rad=self.ball_min_rad_var.get(),
                max_rad=self.ball_max_rad_var.get(),
                body_height_m=self._get_body_height_m(),
            )
            frame_idx += 1
            processed += 1
            if processed % 20 == 0:
                self.root.after(0, lambda idx=frame_idx: self.status_var.set(f"Pass 1/2: Voranalyse Frame {idx}"))

        cap_pre.release()
        self.preanalysis_running = False
        self.preanalysis_pending = False

        if self.engine:
            self.engine.reset_runtime_state()

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        self.current_frame = start_f
        self.root.after(
            0,
            lambda: self.status_var.set(
                f"Pass 2/2: Wiedergabe (Ball-Cache: {len(self.engine.ball_cache_by_frame) if self.engine else 0})"
            ),
        )
        self._update_calibration_status()

    def _loop(self):
        while self.running and self.cap is not None:
            if self.preanalysis_pending and self.seekable and self.is_analyzing and not self.preanalysis_running:
                self._run_preanalysis_pass()

            if self.paused: 
                time.sleep(0.05)
                continue

            start_f, end_f = self._get_frame_range()
            frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.seekable else self.current_frame

            if self.seekable and frame_pos < start_f:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                self.current_frame = start_f
                continue

            if self.seekable and frame_pos > end_f:
                if self.is_analyzing and not getattr(self, "report_generated_auto", False):
                    self.root.after(0, self._generate_report)
                    self.report_generated_auto = True
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                if self.is_analyzing and self.engine:
                    self.engine.reset_runtime_state()
                self.current_frame = start_f
                continue

            t_start = time.perf_counter()
            ok, frame = self.cap.read()
            
            if not ok: 
                if self.seekable:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f))
                    if self.is_analyzing and self.engine:
                        self.engine.reset_runtime_state()
                    self.current_frame = max(0, start_f)
                    continue
                self.running = False
                break

            frame_idx = frame_pos if self.seekable else (self.current_frame + 1)
            if not self.is_analyzing:
                display_frame = frame.copy()
                if self.engine and getattr(self.engine, "player_bbox", None):
                    x, y, w, h = self.engine.player_bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Analyse-Bereich",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                self.last_frames = {
                    "orig": display_frame,
                    "mask": np.zeros_like(display_frame),
                    "skeleton": display_frame.copy(),
                    "final": display_frame.copy(),
                }
                self.current_frame = frame_idx
            elif self.engine:
                self._process_frame_and_refresh(frame, frame_idx)
                
            t_dt = (1.0 / self.fps) - (time.perf_counter() - t_start)
            if t_dt > 0: 
                time.sleep(t_dt)
                
        if self.cap: 
            self.cap.release()
            self.cap = None

    def _refresh_preview(self):
        if self.last_frames:
            try:
                curr_tab = self.view_notebook.index("current")
            except tk.TclError:
                curr_tab = 0
            
            tgt_w, tgt_h = 960, 540
            
            if curr_tab == 0:
                f1 = cv2.resize(self.last_frames["orig"], (tgt_w//2, tgt_h//2))
                mask_src = self.last_frames["mask"]
                if len(mask_src.shape) == 2:
                    mask_src = cv2.cvtColor(mask_src, cv2.COLOR_GRAY2BGR)
                f2 = cv2.resize(mask_src, (tgt_w//2, tgt_h//2))
                f3 = cv2.resize(self.last_frames["skeleton"], (tgt_w//2, tgt_h//2))
                f4 = cv2.resize(self.last_frames["final"], (tgt_w//2, tgt_h//2))
                img_bgr = np.vstack((np.hstack((f1, f2)), np.hstack((f3, f4))))
            elif curr_tab == 1:
                img_bgr = cv2.resize(self.last_frames["orig"], (tgt_w, tgt_h))
            elif curr_tab == 2:
                mask_bgr = cv2.cvtColor(self.last_frames["mask"], cv2.COLOR_GRAY2BGR) if len(self.last_frames["mask"].shape) == 2 else self.last_frames["mask"]
                img_bgr = cv2.resize(mask_bgr, (tgt_w, tgt_h))
            elif curr_tab == 3:
                img_bgr = cv2.resize(self.last_frames["skeleton"], (tgt_w, tgt_h))
            elif curr_tab == 4:
                img_bgr = cv2.resize(self.last_frames["final"], (tgt_w, tgt_h))
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            if self.pil_image:
                self._photo = self.pil_imagetk.PhotoImage(image=self.pil_image.fromarray(img_rgb))
            else:
                self._photo = tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", img_rgb)[1].tobytes()), format="png")
            
            self.tab_labels[curr_tab].configure(image=self._photo, text="")
            
        if self.seekable: 
            self.lbl_timeline.configure(text=f"Frame: {self.current_frame} / {max(0, self.total_frames - 1)}")
            self.timeline_var.set(self.current_frame)
            
        if self.running: 
            self.root.after(50, self._refresh_preview)

    def _on_close(self): 
        self.stop()
        self.root.destroy()
        
    def run(self): 
        self.root.mainloop()

    def export_trimmed_video(self):
        """Schneidet das Video auf Start/Ende zu und speichert es als _opt."""
        if not self.cap or not getattr(self, "video_path", None):
            return None

        name, _ = os.path.splitext(self.video_path)
        name = re.sub(r"(?:_opt)+$", "", name, flags=re.IGNORECASE)
        opt_path = f"{name}_opt.avi"

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(opt_path, fourcc, fps, (w, h))
        if not out.isOpened():
            return None

        start_f, end_f = self._get_frame_range()
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        current_f = start_f

        while current_f <= end_f:
            ret, frame = self.cap.read()
            if not ret:
                break
            out.write(frame)
            current_f += 1

        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        return opt_path

    def start_full_analysis(self):
        """Exportiert erst das Video, lädt es neu und startet dann die Analyse."""
        if not self.cap or not self.seekable:
            messagebox.showwarning("Achtung", "Bitte zuerst ein Video laden und mit Start abspielen.")
            return

        if not self.engine:
            messagebox.showerror("Fehler", "Analyse-Engine ist nicht initialisiert. Bitte Video neu starten.")
            return

        pose_mode = getattr(self.engine.pose_estimator, "mode", "none")
        if pose_mode == "none":
            proceed_without_pose = messagebox.askyesno(
                "Pose-Backend fehlt",
                "Es ist kein aktives Pose-Backend verfügbar.\n"
                "Soll die Analyse trotzdem mit Ball-Tracking (ohne Skelett/Winkel) gestartet werden?",
            )
            if not proceed_without_pose:
                return

        current_video = getattr(self, "video_path", "") or self.file_var.get().strip()
        current_name = os.path.splitext(os.path.basename(current_video))[0].lower()
        is_already_opt = current_name.endswith("_opt")
        self.paused = True
        self.btn_pause.configure(text="Play")

        # Bereits optimierte Datei nicht erneut als *_opt exportieren.
        if not is_already_opt:
            messagebox.showinfo(
                "Video wird vorbereitet",
                "Das Video wird jetzt auf den Wurfmoment zugeschnitten. Das dauert einen kleinen Moment...",
            )

            opt_path = self.export_trimmed_video()
            if not opt_path:
                messagebox.showerror("Fehler", "Das zugeschnittene Video konnte nicht erstellt werden.")
                return

            self.video_path = opt_path
            self.file_var.set(opt_path)
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Fehler", f"Das neue Video konnte nicht geladen werden:\n{self.video_path}")
                return

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps or 30.0
            self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.timeline_slider.configure(to=max(0, self.total_frames - 1))
            self.start_frame_var.set("0")
            self.end_frame_var.set(str(max(0, self.total_frames - 1)))
            self.timeline_var.set(0)
            self.current_frame = 0
            self._apply_speed_frame_range_inputs(update_status=False)

        self.is_analyzing = True
        self.report_generated_auto = False
        self.preanalysis_pending = self.seekable and self.use_two_pass_var.get()
        if self.engine:
            self.engine.reset_runtime_state()
            self.engine.clear_ball_cache()
            self.engine.ball_detector.reset_background_model()
        self.paused = False
        self.btn_pause.configure(text="Analyse läuft...")
        if pose_mode == "none":
            self.status_var.set("Analyse läuft (Ball-only, ohne Pose/Winkel).")
        else:
            self.status_var.set(f"Analyse läuft (Pose: {pose_mode}).")

    def _start_player_bbox_marking(self):
        """Öffnet ein Fenster, um die Region of Interest (ROI) zu markieren."""
        if not self.last_frames or "orig" not in self.last_frames:
            messagebox.showwarning("Achtung", "Bitte lade ein Video und pausiere an einer guten Stelle.")
            return

        self.paused = True
        self.btn_pause.configure(text="Play")
        # Wichtiger Stabilitätsfix: selectROI im UI-Thread ausführen.
        # OpenCV HighGUI ist in manchen Builds nicht thread-safe und kann sonst den Prozess beenden.
        self._player_bbox_thread()

    def _player_bbox_thread(self):
        win_name = "Spieler markieren (ENTER zum Bestaetigen)"
        try:
            clone = self.last_frames["orig"].copy()
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

            bbox = cv2.selectROI(win_name, clone, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(win_name)
        except Exception as exc:
            try:
                cv2.destroyWindow(win_name)
            except Exception:
                pass
            self.root.after(0, lambda: messagebox.showerror("ROI-Fehler", f"Spieler-ROI konnte nicht gesetzt werden:\n{exc}"))
            return

        if bbox and bbox[2] > 0 and bbox[3] > 0:
            if self.engine:
                self.engine.set_player_bbox(bbox)
            self.root.after(0, lambda: messagebox.showinfo("Erfolg", "Spieler markiert. Die Analyse ist nun massiv beschleunigt!"))
        else:
            if self.engine:
                self.engine.set_player_bbox(None)
