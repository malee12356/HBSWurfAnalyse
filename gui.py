import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import base64
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
    def __init__(self, parent, text="", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.show = tk.IntVar(value=1)
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill="x", expand=1)
        self.toggle_button = ttk.Checkbutton(self.title_frame, width=2, text='▼', command=self.toggle, variable=self.show, style='Toolbutton')
        self.toggle_button.pack(side="left")
        ttk.Label(self.title_frame, text=text, font=("Helvetica", 10, "bold")).pack(side="left", fill="x", expand=1)
        self.sub_frame = ttk.Frame(self, relief="sunken", borderwidth=1)
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
        self.ball_min_rad_var = tk.IntVar(value=15)
        self.ball_max_rad_var = tk.IntVar(value=35)
        
        self.real_length_var = tk.DoubleVar(value=1.65)
        self.calibration_mode_var = tk.StringVar(value="body_height")
        self.use_two_pass_var = tk.BooleanVar(value=True)
        self.target_speed_var = tk.DoubleVar(value=90.0)
        
        self.start_frame_var = tk.StringVar(value="0")
        self.end_frame_var = tk.StringVar(value="999999")
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
        
        self.reference_seq = load_reference_sequence(reference_json)
        self.last_frames = None
        self.ball_color_samples = []
        self.manual_reference = None

        self._photo = None
        self.pil_image = optional_import("PIL.Image")
        self.pil_imagetk = optional_import("PIL.ImageTk")
        
        self._build_layout()

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

        tf_ki = ToggledFrame(self.frame_settings, text="1. Skelett (KI)")
        tf_ki.pack(fill=tk.X, pady=5)
        ttk.Label(tf_ki.sub_frame, text="Backend:").pack(anchor="w")
        ttk.Combobox(tf_ki.sub_frame, textvariable=self.pose_backend_var, values=("auto", "mediapipe", "mmpose"), state="readonly").pack(fill=tk.X, pady=(0,5))
        
        lbl_kpt = ttk.Label(tf_ki.sub_frame, text=f"Sicherheit: {self.kpt_thr_var.get():.2f}")
        lbl_kpt.pack(anchor="w")
        ttk.Scale(tf_ki.sub_frame, from_=0.0, to=1.0, variable=self.kpt_thr_var, command=lambda v: lbl_kpt.configure(text=f"Sicherheit: {float(v):.2f}")).pack(fill=tk.X)
        
        tf_ball = ToggledFrame(self.frame_settings, text="2. Ball Filter & Tracker")
        tf_ball.pack(fill=tk.X, pady=5)
        ttk.Button(tf_ball.sub_frame, text="🎯 Ball manuell markieren", command=self._start_ball_tracker).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tf_ball.sub_frame, text="Tracker zurücksetzen", command=self._reset_ball_tracker).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tf_ball.sub_frame, text="🎨 Ballfarbe sampeln", command=self._start_ball_color_sampling).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(tf_ball.sub_frame, text="Farbfilter löschen", command=self._clear_ball_color_filter).pack(fill=tk.X, pady=(0, 5))
        self.lbl_ball_color_status = ttk.Label(tf_ball.sub_frame, text="Kein Farbfilter", foreground="gray")
        self.lbl_ball_color_status.pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(tf_ball.sub_frame, text="MOG2 Filter an", variable=self.use_mog2_var).pack(anchor="w", pady=(0,5))
        lbl_min_rad = ttk.Label(tf_ball.sub_frame, text=f"Min. Radius: {self.ball_min_rad_var.get()} px")
        lbl_min_rad.pack(anchor="w")
        ttk.Scale(tf_ball.sub_frame, from_=5, to=50, variable=self.ball_min_rad_var, command=lambda v: lbl_min_rad.configure(text=f"Min. Radius: {int(float(v))} px")).pack(fill=tk.X)
        lbl_max_rad = ttk.Label(tf_ball.sub_frame, text=f"Max. Radius: {self.ball_max_rad_var.get()} px")
        lbl_max_rad.pack(anchor="w")
        ttk.Scale(tf_ball.sub_frame, from_=10, to=80, variable=self.ball_max_rad_var, command=lambda v: lbl_max_rad.configure(text=f"Max. Radius: {int(float(v))} px")).pack(fill=tk.X)
        lbl_s_max = ttk.Label(tf_ball.sub_frame, text=f"Max. Sättigung: {self.ball_s_max_var.get()}")
        lbl_s_max.pack(anchor="w", pady=(5,0))
        ttk.Scale(tf_ball.sub_frame, from_=10, to=255, variable=self.ball_s_max_var, command=lambda v: lbl_s_max.configure(text=f"Max. Sättigung: {int(float(v))}")).pack(fill=tk.X)
        lbl_v_min = ttk.Label(tf_ball.sub_frame, text=f"Min. Helligkeit: {self.ball_v_min_var.get()}")
        lbl_v_min.pack(anchor="w")
        ttk.Scale(tf_ball.sub_frame, from_=0, to=200, variable=self.ball_v_min_var, command=lambda v: lbl_v_min.configure(text=f"Min. Helligkeit: {int(float(v))}")).pack(fill=tk.X)
        
        tf_roi = ToggledFrame(self.frame_settings, text="3. Sichtfeld (ROI)")
        tf_roi.pack(fill=tk.X, pady=5)
        ttk.Button(tf_roi.sub_frame, text="Bereich markieren", command=self._start_roi_marking).pack(fill=tk.X, pady=2)
        ttk.Button(tf_roi.sub_frame, text="Zurücksetzen", command=self._reset_roi).pack(fill=tk.X)
        self.lbl_roi_status = ttk.Label(tf_roi.sub_frame, text="Gesamtes Bild", foreground="green")
        self.lbl_roi_status.pack(anchor="w", pady=(5, 0))

        tf_calib = ToggledFrame(self.frame_settings, text="4. Kalibrierung")
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
        
        tf_time = ToggledFrame(self.frame_settings, text="5. Loop")
        tf_time.pack(fill=tk.X, pady=5)
        row1 = ttk.Frame(tf_time.sub_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.start_frame_var, width=5).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row1, text="Ende:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.end_frame_var, width=5).pack(side=tk.LEFT)
        ttk.Checkbutton(tf_time.sub_frame, text="Loop in Bereich", variable=self.loop_var).pack(anchor="w", pady=(5,0))
        ttk.Checkbutton(tf_time.sub_frame, text="2-Pass Voranalyse (Datei)", variable=self.use_two_pass_var).pack(anchor="w", pady=(5,0))

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
            self.file_var.set(path)
            self.source_var.set("file")

    def _open_source(self):
        ensure_cv2()
        self.cap, source_name = select_video_capture(self.source_var.get(), self.file_var.get().strip(), self.camera_var.get())
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.seekable = self.source_var.get() == "file"
        
        if self.seekable:
            self.total_frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.timeline_slider.configure(to=max(0, self.total_frames - 1))
            self.end_frame_var.set(str(max(0, self.total_frames - 1)))
            
        self._update_controls()
        self.report_generated_auto = False
        self.preanalysis_pending = self.seekable and self.use_two_pass_var.get()
        self.preanalysis_running = False
        self.current_frame = 0
        self.engine = AnalysisEngine(
            fps=self.fps,
            reference_seq=self.reference_seq,
            pose_backend=self.pose_backend_var.get(),
            calibration_mode=self.calibration_mode_var.get(),
        )
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

    def _update_metrics_from_result(self, metrics, score):
        if not metrics:
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
        threading.Thread(target=self._roi_thread, daemon=True).start()

    def _roi_thread(self):
        clone = self.last_frames["orig"].copy()
        pts = []
        win_name = "ROI (ENTER zum Beenden)"
        
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
        
        if len(pts) >= 3:
            self.roi_polygon = pts
            self.root.after(0, lambda: self.lbl_roi_status.configure(text="Bereich aktiv", foreground="blue"))
        else:
            self.roi_polygon = []
            self.root.after(0, lambda: self.lbl_roi_status.configure(text="Gesamtes Bild", foreground="green"))

    def _reset_roi(self):
        self.roi_polygon = []
        self.lbl_roi_status.configure(text="Gesamtes Bild", foreground="green")

    def _export_video(self):
        if not self.seekable or not self.file_var.get():
            messagebox.showerror("Fehler", "Export geht nur mit Videodateien (nicht bei Webcam-Livebild).")
            return
            
        # Geändert von .mp4 auf .avi
        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI Video", "*.avi")])
        if not save_path:
            return
            
        threading.Thread(target=self._export_thread, args=(save_path,), daemon=True).start()

    def _export_thread(self, save_path):
        self.root.after(0, lambda: self.status_var.set("Export... Phase 1/2: Analysiere Video..."))
        
        cap = cv2.VideoCapture(self.file_var.get().strip())
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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
            
        try: start_f = int(self.start_frame_var.get())
        except ValueError: start_f = 0
            
        try: end_f = int(self.end_frame_var.get())
        except ValueError: end_f = 999999
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_idx = 0
        
        # PHASE 1: Komplette Video-Analyse (damit der Bericht alle Daten hat)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame_idx > (end_f - start_f):
                break
                
            export_engine.process_frame_with_pose(
                frame, frame_idx / fps, self.kpt_thr_var.get(), 
                self.ball_s_max_var.get(), self.ball_v_min_var.get(), self.hough_p1_var.get(), 
                self.hough_p2_var.get(), self.roi_polygon, self.target_speed_var.get(), 
                self.use_mog2_var.get(), self.ball_min_rad_var.get(), self.ball_max_rad_var.get(),
                body_height_m=self._get_body_height_m(),
                frame_index=start_f + frame_idx,
            )
            frame_idx += 1
            if frame_idx % 10 == 0:
                self.root.after(0, lambda idx=frame_idx: self.status_var.set(f"Phase 1/2: Analysiere Frame {idx}..."))
                
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

        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame_idx > (end_f - start_f):
                break
                
            _, _, _, debug_frames, _ = export_engine_render.process_frame_with_pose(
                frame, frame_idx / fps, self.kpt_thr_var.get(), 
                self.ball_s_max_var.get(), self.ball_v_min_var.get(), self.hough_p1_var.get(), 
                self.hough_p2_var.get(), self.roi_polygon, self.target_speed_var.get(), 
                self.use_mog2_var.get(), self.ball_min_rad_var.get(), self.ball_max_rad_var.get(),
                body_height_m=self._get_body_height_m(),
                frame_index=start_f + frame_idx,
            )
            
            final_f = debug_frames["final"]
            if final_f.shape[0] != out_h:
                 final_f = cv2.resize(final_f, (w, out_h))
            
            combined = np.hstack((final_f, text_panel))
            out.write(combined)
            frame_idx += 1
            if frame_idx % 10 == 0:
                self.root.after(0, lambda idx=frame_idx: self.status_var.set(f"Phase 2/2: Schreibe Frame {idx}..."))
            
        cap.release()
        out.release()
        self.root.after(0, lambda: self.status_var.set("Bereit"))
        self.root.after(0, lambda: messagebox.showinfo("Erfolg", f"Video erfolgreich als AVI exportiert!\n{save_path}"))

    def start(self):
        if self.running: 
            return
        try: 
            self._open_source()
        except Exception as exc: 
            messagebox.showerror("Fehler", str(exc))
            return
        self.status_var.set("Bereit")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.root.after(50, self._refresh_preview)

    def stop(self):
        self.running = False
        self.paused = False
        self.preanalysis_pending = False
        self.preanalysis_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_pause.configure(text="Pause")

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
        if self.preanalysis_pending and self.seekable:
            self._run_preanalysis_pass()

        while self.running and self.cap is not None:
            if self.paused: 
                time.sleep(0.05)
                continue

            start_f, end_f = self._get_frame_range()
            frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.seekable else self.current_frame

            if self.loop_var.get() and self.seekable and (frame_pos > end_f or frame_pos < start_f):
                if not getattr(self, "report_generated_auto", False):
                    self.root.after(0, self._generate_report)
                    self.report_generated_auto = True
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                if self.engine:
                    self.engine.reset_runtime_state()
                self.current_frame = start_f
                continue

            t_start = time.perf_counter()
            ok, frame = self.cap.read()
            
            if not ok: 
                if not getattr(self, "report_generated_auto", False):
                    self.root.after(0, self._generate_report)
                    self.report_generated_auto = True
                    
                if self.loop_var.get() and self.seekable:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f))
                    if self.engine:
                        self.engine.reset_runtime_state()
                    continue
                else:
                    self.running = False
                    break

            if self.engine:
                frame_idx = frame_pos if self.seekable else (self.current_frame + 1)
                self._process_frame_and_refresh(frame, frame_idx)
                
            t_dt = (1.0 / self.fps) - (time.perf_counter() - t_start)
            if t_dt > 0: 
                time.sleep(t_dt)
                
        if self.cap: 
            self.cap.release()

    def _refresh_preview(self):
        if self.last_frames:
            try:
                curr_tab = self.view_notebook.index("current")
            except tk.TclError:
                curr_tab = 0
            
            tgt_w, tgt_h = 960, 540
            
            if curr_tab == 0:
                f1 = cv2.resize(self.last_frames["orig"], (tgt_w//2, tgt_h//2))
                f2 = cv2.resize(self.last_frames["mask"], (tgt_w//2, tgt_h//2))
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
