import sys
import os
import time
import threading
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                           QWidget, QLabel, QFileDialog, QSlider, QSpinBox, QListWidget,
                           QListWidgetItem, QMessageBox, QProgressBar, QCheckBox, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QUrl, pyqtSignal, QThread, QMetaObject
from PyQt5.QtGui import QFont, QIcon
import librosa
import soundfile as sf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pygame
import tempfile
import shutil
import math # Keep math for potential future use, though ceil isn't needed now

# --- pydub Import Handling ---
PYDUB_AVAILABLE = False
FFMPEG_WARNING_ISSUED = False
try:
    from pydub import AudioSegment as PydubAudioSegment
    from pydub.silence import detect_nonsilent
    # Check if ffmpeg/ffprobe are accessible by trying a basic operation
    try:
        _test_segment = PydubAudioSegment.silent(duration=10)
        PYDUB_AVAILABLE = True
    except Exception as pydub_init_err:
        print(f"Warning: pydub imported but may lack dependencies (like ffmpeg): {pydub_init_err}")
        if "ffmpeg" in str(pydub_init_err).lower() or "ffprobe" in str(pydub_init_err).lower():
             print("*"*60)
             print(" تحذير: مكتبة pydub تتطلب ffmpeg/ffprobe. ")
             print(" قد لا تعمل وظيفة تقسيم الصوت بشكل صحيح.")
             print(" الرجاء تثبيت ffmpeg وإضافته إلى مسار النظام (PATH).")
             print("*"*60)
             FFMPEG_WARNING_ISSUED = True
        # Allow running but splitting might fail later
        PYDUB_AVAILABLE = False # Treat as unavailable if basic operation fails

except ImportError:
    print("*"*60)
    print(" تحذير: مكتبة pydub غير موجودة. ")
    print(" وظيفة تقسيم الصوت لن تعمل بدونها. ")
    print(" الرجاء تثبيتها باستخدام: pip install pydub ")
    print("*"*60)
    PYDUB_AVAILABLE = False
except Exception as e:
    print(f"خطأ غير متوقع عند استيراد pydub: {e}")
    PYDUB_AVAILABLE = False


# --- فئة مقطع الصوت (AudioSegment Class) ---
# (No significant changes needed here, keeping as is)
class AudioSegment:
    def __init__(self, start_time, end_time, audio_data, sr):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.audio_data = audio_data # بيانات الصوت الخام لهذا المقطع
        self.sr = sr # معدل التعيين
        self.temp_file = None # المسار إلى الملف المؤقت (إذا تم إنشاؤه)

    def create_temp_file(self):
        """ينشئ ملف WAV مؤقت لتشغيل هذا المقطع."""
        if self.temp_file and os.path.exists(self.temp_file):
            return self.temp_file # إذا كان موجوداً بالفعل، أعد استخدامه

        try:
            # Ensure the directory for temp files exists
            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)

            # Create temp file within the temp directory
            fd, temp_path = tempfile.mkstemp(suffix='.wav', dir=temp_dir)
            os.close(fd)

            # Write audio data
            sf.write(temp_path, self.audio_data, self.sr)
            self.temp_file = temp_path
            # print(f"Created temp file: {temp_path}") # Debugging
            return temp_path
        except Exception as e:
            print(f"خطأ في إنشاء الملف المؤقت: {e}")
            # Attempt to clean up if writing failed
            if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass
            self.temp_file = None
            return None

    def release_temp_file(self):
        """يحذف الملف المؤقت لتحرير المساحة."""
        if self.temp_file and os.path.exists(self.temp_file):
            temp_to_remove = self.temp_file
            self.temp_file = None # Set to None first
            try:
                # Attempt to remove the file
                os.remove(temp_to_remove)
                # print(f"تم حذف الملف المؤقت: {temp_to_remove}") # Debugging
            except PermissionError:
                 print(f"Warning: Could not delete temp file (possibly still in use): {temp_to_remove}")
                 # Schedule for later deletion? Or rely on OS cleanup.
            except FileNotFoundError:
                 pass # Already deleted
            except Exception as e:
                print(f"خطأ أثناء حذف الملف المؤقت {temp_to_remove}: {e}")
                # pass # Ignore other errors for now


    def __del__(self):
        """التنظيف عند حذف الكائن."""
        self.release_temp_file()


# --- فئة معالجة الصوت في خيط منفصل (Revised) ---
class AudioProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_done = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, audio_file, silence_threshold_db, min_silence_duration_sec,
                 enable_max_duration, max_segment_duration_sec):
        super().__init__()
        self.audio_file = audio_file
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_len_ms = int(min_silence_duration_sec * 1000)
        self.enable_max_duration = enable_max_duration
        self.max_segment_duration_ms = int(max_segment_duration_sec * 1000) if enable_max_duration else float('inf')

        # Ensure min_silence_len_ms is reasonable for pydub
        if self.min_silence_len_ms < 25: # pydub might struggle with very short values
             print(f"تحذير: مدة الصمت الدنيا ({self.min_silence_len_ms}ms) صغيرة جدًا, تم تعديلها إلى 25ms.")
             self.min_silence_len_ms = 25

    def run(self):
        global FFMPEG_WARNING_ISSUED
        if not PYDUB_AVAILABLE:
            error_msg = "مكتبة pydub غير متاحة أو فشلت التهيئة الأولية.\nيرجى تثبيتها (`pip install pydub`)."
            if FFMPEG_WARNING_ISSUED:
                 error_msg += "\n\nتأكد أيضاً من تثبيت ffmpeg وإضافته إلى مسار النظام (PATH)."
            self.error_occurred.emit(error_msg)
            return

        try:
            self.progress_updated.emit(5)
            print(f"بدء المعالجة: {self.audio_file}")
            print(f"إعدادات الصمت: عتبة={self.silence_threshold_db}dBFS, مدة دنيا={self.min_silence_len_ms}ms")
            if self.enable_max_duration:
                print(f"إعدادات المدة القصوى: مفعلة, حد أقصى={self.max_segment_duration_ms / 1000.0:.2f}s")
            else:
                print("إعدادات المدة القصوى: غير مفعلة")

            # 1. تحميل الملف الصوتي باستخدام pydub
            self.progress_updated.emit(15)
            try:
                file_extension = os.path.splitext(self.audio_file)[1].lower().replace('.', '')
                if not file_extension: file_extension = 'mp3' # Default guess
                print(f"محاولة تحميل باستخدام pydub (صيغة مقترحة: {file_extension})...")
                audio = PydubAudioSegment.from_file(self.audio_file, format=file_extension)
            except Exception as load_err_1:
                 print(f"فشل التحميل بالصيغة المقترحة ({load_err_1}). محاولة الاكتشاف التلقائي...")
                 try:
                     audio = PydubAudioSegment.from_file(self.audio_file)
                 except Exception as load_err_2:
                     print(f"خطأ فادح أثناء تحميل الملف باستخدام pydub: {load_err_2}")
                     error_detail = str(load_err_2)
                     if "ffmpeg" in error_detail.lower() or "ffprobe" in error_detail.lower() or "specified path" in error_detail.lower():
                         FFMPEG_WARNING_ISSUED = True
                         self.error_occurred.emit("خطأ: لم يتم العثور على برنامج ffmpeg/ffprobe أو الوصول إليه.\n"
                                                  "مكتبة pydub تتطلبه لمعالجة الملفات الصوتية.\n"
                                                  "يرجى تثبيت ffmpeg وإضافته إلى مسار النظام (PATH).")
                     else:
                         self.error_occurred.emit(f"فشل تحميل الملف الصوتي بـ pydub:\n{error_detail}")
                     return

            audio_duration_ms = len(audio)
            print(f"تم تحميل الملف (pydub). المدة: {audio_duration_ms / 1000.0:.2f} ثانية. معدل الإطارات: {audio.frame_rate}Hz.")
            self.progress_updated.emit(30)

            # 2. اكتشاف الأجزاء غير الصامتة
            print(f"اكتشاف الأجزاء غير الصامتة (min_silence_len={self.min_silence_len_ms}ms, silence_thresh={self.silence_threshold_db}dBFS)...")
            nonsilent_ranges_ms = detect_nonsilent(
                audio,
                min_silence_len=self.min_silence_len_ms,
                silence_thresh=self.silence_threshold_db,
                seek_step=1 # Check every ms
            )

            if not nonsilent_ranges_ms:
                print("لم يتم العثور على أجزاء غير صامتة بالإعدادات الحالية.")
                # Check if the file is actually empty or completely silent according to pydub
                if audio.dBFS > self.silence_threshold_db : # Check if overall level is above threshold
                     print("الملف يحتوي على صوت ولكنه لم يقسم. قد تكون مدة الصمت الدنيا طويلة جداً أو العتبة غير مناسبة.")
                     print("اعتبار الملف كاملاً كمقطع واحد.")
                     nonsilent_ranges_ms = [[0, audio_duration_ms]] # Treat the whole file as one segment
                else:
                     print("الملف يبدو صامتاً تماماً أو فارغاً.")
                     self.processing_done.emit([])
                     self.progress_updated.emit(100)
                     return # Exit if truly silent/empty


            print(f"تم العثور على {len(nonsilent_ranges_ms)} جزء غير صامت مبدئي.")
            self.progress_updated.emit(50)

            # 3. (جديد) تطبيق قيد المدة القصوى (إذا كان مفعلاً)
            processed_ranges_ms = []
            if self.enable_max_duration and self.max_segment_duration_ms != float('inf') and self.max_segment_duration_ms > 0:
                print(f"تطبيق قيد المدة القصوى ({self.max_segment_duration_ms / 1000.0:.2f}s)...")
                for start_ms, end_ms in nonsilent_ranges_ms:
                    duration_ms = end_ms - start_ms
                    if duration_ms > self.max_segment_duration_ms:
                        # Split this long chunk
                        current_start = start_ms
                        while current_start < end_ms:
                            current_end = min(end_ms, current_start + self.max_segment_duration_ms)
                            if current_end > current_start: # Avoid zero-length segments
                                processed_ranges_ms.append([current_start, current_end])
                            current_start = current_end
                    elif duration_ms > 0: # Add non-empty chunks directly
                        processed_ranges_ms.append([start_ms, end_ms])
                print(f"عدد الأجزاء بعد تطبيق المدة القصوى: {len(processed_ranges_ms)}")
            else:
                 # Only keep non-empty ranges if max duration is not applied
                 processed_ranges_ms = [[start, end] for start, end in nonsilent_ranges_ms if end > start]
                 print("تجاوز خطوة المدة القصوى.")

            self.progress_updated.emit(60)

            if not processed_ranges_ms:
                 print("لا توجد مقاطع بعد المعالجة الأولية وتطبيق القيود.")
                 self.processing_done.emit([])
                 self.progress_updated.emit(100)
                 return

            # 4. تحميل بيانات الصوت الخام باستخدام librosa (للحصول على مصفوفة numpy)
            print("تحميل بيانات الصوت كـ numpy array باستخدام librosa...")
            try:
                 # Load with the sample rate detected by pydub if possible for consistency
                 target_sr = audio.frame_rate
                 print(f"محاولة التحميل بـ librosa بمعدل عينة: {target_sr} Hz")
                 y, sr = librosa.load(self.audio_file, sr=target_sr)
                 # Double check the loaded sample rate
                 if sr != target_sr:
                      print(f"تحذير: librosa حمل بمعدل عينة مختلف ({sr}Hz) عن المطلوب ({target_sr}Hz).")
                      # Could potentially resample here, but let's proceed and see.
                      # The calculations below will use the actual `sr` loaded by librosa.
            except Exception as e:
                 print(f"فشل تحميل librosa بمعدل العينة المستهدف. محاولة التحميل بمعدل أصلي (sr=None)...")
                 try:
                     y, sr = librosa.load(self.audio_file, sr=None)
                     print(f"تم التحميل بمعدل العينة الأصلي: {sr}Hz")
                 except Exception as e_fallback:
                     self.error_occurred.emit(f"فشل تحميل بيانات الصوت الخام باستخدام librosa:\n{e_fallback}")
                     return

            # Verify audio duration consistency (optional but good check)
            librosa_duration_sec = len(y) / sr
            pydub_duration_sec = audio_duration_ms / 1000.0
            if abs(librosa_duration_sec - pydub_duration_sec) > 0.1: # Allow small difference
                print(f"تحذير: اختلاف ملحوظ في المدة المحسوبة بين pydub ({pydub_duration_sec:.2f}s) و librosa ({librosa_duration_sec:.2f}s).")

            self.progress_updated.emit(75)

            # 5. إنشاء قائمة المقاطع النهائية (AudioSegment)
            final_audio_segments = []
            print("إنشاء كائنات المقاطع النهائية...")
            num_ranges = len(processed_ranges_ms)
            for i, (start_ms, end_ms) in enumerate(processed_ranges_ms):
                start_time_sec = start_ms / 1000.0
                end_time_sec = end_ms / 1000.0

                # Calculate corresponding sample indices in the numpy array 'y'
                # Use max/min and ensure start < end
                start_sample = max(0, min(len(y) - 1, int(start_time_sec * sr)))
                end_sample = max(0, min(len(y), int(end_time_sec * sr)))

                # Extract audio data for the segment
                if start_sample < end_sample:
                    segment_audio_data = y[start_sample:end_sample]
                    if np.any(segment_audio_data): # Ensure there's actual audio data
                        segment = AudioSegment(start_time_sec, end_time_sec, segment_audio_data, sr)
                        final_audio_segments.append(segment)
                        # print(f"  ++ مقطع {len(final_audio_segments)}: [{start_time_sec:.3f}s - {end_time_sec:.3f}s]") # Debug
                    else:
                         print(f"  -- تجاهل مقطع فارغ (بعد الاستخراج): [{start_time_sec:.3f}s - {end_time_sec:.3f}s], samples [{start_sample}-{end_sample}]")
                else:
                    print(f"  -- تجاهل مقطع غير صالح (samples): start={start_sample}, end={end_sample}")

                # Update progress
                progress = 75 + int(((i + 1) / num_ranges) * 25)
                self.progress_updated.emit(progress)

            print(f"اكتملت المعالجة. العدد النهائي للمقاطع: {len(final_audio_segments)}")
            if not final_audio_segments and len(processed_ranges_ms) > 0:
                 print("تحذير: تم العثور على نطاقات زمنية لكن لم يتم إنشاء مقاطع صوتية نهائية. تحقق من حسابات الفهرسة أو بيانات librosa.")


            self.progress_updated.emit(100)
            self.processing_done.emit(final_audio_segments)

        except FileNotFoundError:
             self.error_occurred.emit(f"لم يتم العثور على الملف: {self.audio_file}")
        except Exception as e:
            import traceback
            print(f"حدث خطأ غير متوقع في خيط المعالجة: {e}")
            traceback.print_exc()
            error_detail = str(e)
            if "ffmpeg" in error_detail.lower() or "ffprobe" in error_detail.lower():
                 FFMPEG_WARNING_ISSUED = True
                 self.error_occurred.emit("خطأ متعلق بـ ffmpeg/ffprobe. تأكد من تثبيته وإضافته إلى PATH.")
            elif "specified path" in error_detail.lower(): # Often related to ffmpeg path
                 FFMPEG_WARNING_ISSUED = True
                 self.error_occurred.emit("خطأ في مسار ffmpeg/ffprobe. تأكد من صحة التثبيت والإعداد.")
            else:
                 self.error_occurred.emit(f"خطأ غير متوقع أثناء المعالجة:\n{error_detail}")

# --- فئة عرض الشكل الموجي (AudioWaveform Class - Fixed Colors) ---
class AudioWaveform(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken) # Add a frame style
        self.setMinimumHeight(120) # Slightly taller
        self.figure = Figure(figsize=(5, 1.2), dpi=100) # Adjusted figsize slightly
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus) # Allow canvas to receive focus if needed

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.segments = []
        self.audio_data = None
        self.sr = None
        self.selected_segment = -1
        self.full_duration = 0

    def set_audio(self, audio_file):
        """تحميل بيانات الصوت ورسم الموجة الأولية."""
        try:
            # Use a default SR for faster loading of waveform preview if needed,
            # but sr=None is better for accuracy if performance allows.
            y, sr = librosa.load(audio_file, sr=None) # Load original SR
            self.audio_data = y
            self.sr = sr
            self.full_duration = librosa.get_duration(y=y, sr=sr)
            self.segments = [] # Clear old segments
            self.selected_segment = -1
            self.plot_waveform() # Plot initial waveform
        except Exception as e:
            print(f"Error loading audio for waveform: {e}")
            self.audio_data = None
            self.sr = None
            self.full_duration = 0
            self.figure.clear() # Clear the plot on failure
            self.canvas.draw()


    def plot_waveform(self):
        """يرسم الشكل الموجي للصوت مع تحديد المقاطع."""
        if self.audio_data is None or len(self.audio_data) == 0:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'لا توجد بيانات صوتية للعرض', horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Calculate time axis
        times = np.linspace(0, self.full_duration, num=len(self.audio_data))

        # Plot the main waveform
        ax.plot(times, self.audio_data, color='#3498db', linewidth=0.6, label='Waveform') # Slightly thicker line

        # Add vertical lines for segment boundaries
        segment_boundaries = set()
        for segment in self.segments:
            segment_boundaries.add(segment.start_time)
            segment_boundaries.add(segment.end_time)

        # ***** Correct RGBA Color Format *****
        # boundary_color = 'rgba(0, 128, 0, 0.7)' # Incorrect format
        boundary_color = (0.0, 0.502, 0.0, 0.7) # Correct tuple format (Green, semi-transparent)
        # ************************************
        for boundary in sorted(list(segment_boundaries)):
             # Ensure boundary is within plot limits
             if 0 <= boundary <= self.full_duration:
                 ax.axvline(x=boundary, color=boundary_color, linestyle='--', linewidth=0.9) # Slightly thicker line

        # Highlight the selected segment (if any)
        if 0 <= self.selected_segment < len(self.segments):
            selected = self.segments[self.selected_segment]
            # ***** Correct RGBA Color Format *****
            # highlight_color_str = 'rgba(255, 0, 0, 0.3)' # Incorrect format
            highlight_color_tuple = (1.0, 0.0, 0.0, 0.3)   # Correct tuple format (Red, semi-transparent)
            # ************************************
            # Ensure span is within plot limits
            start_span = max(0, selected.start_time)
            end_span = min(self.full_duration, selected.end_time)
            if start_span < end_span:
                 ax.axvspan(start_span, end_span, color=highlight_color_tuple, label=f'Segment {self.selected_segment+1}')

        # Formatting the plot
        max_amplitude = np.max(np.abs(self.audio_data)) if len(self.audio_data) > 0 else 1.0
        if max_amplitude == 0: max_amplitude = 1.0 # Avoid zero ylim range for silent audio
        ax.set_ylim(-max_amplitude * 1.1, max_amplitude * 1.1) # Add a small margin
        ax.set_xlim(0, self.full_duration)
        ax.set_xlabel('الوقت (ثواني)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Plot of Audio Waveform')
        ax.grid(True, linestyle=':', alpha=0.5) # Slightly more visible grid

        # Reduce margins for better space usage
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.2)

        try:
            self.canvas.draw() # Redraw the canvas
        except Exception as e:
            print(f"خطأ أثناء رسم المخطط: {e}")

    def update_segments(self, segments, selected_idx=-1):
        """تحديث قائمة المقاطع والمقطع المحدد وإعادة الرسم."""
        # Release old temp files *before* replacing the list
        # for old_segment in self.segments:
        #     old_segment.release_temp_file()
        # It's safer to let __del__ handle this or manage it explicitly elsewhere

        self.segments = segments
        self.selected_segment = selected_idx
        self.plot_waveform() # Redraw with new segments


# --- فئة مشغل الصوت (AudioPlayer Class) ---
# (No significant changes needed here, keeping as is)
class AudioPlayer:
    def __init__(self):
        self.mixer_initialized = False
        try:
            pygame.mixer.init() # Initialize pygame mixer
            self.mixer_initialized = True
            print("Pygame mixer initialized successfully.")
        except pygame.error as e:
            print(f"خطأ في تهيئة pygame.mixer: {e}")
            # Show warning later in the GUI if needed
            QMessageBox.warning(None, "خطأ في الصوت",
                                "لم يتمكن من تهيئة نظام الصوت (pygame.mixer).\n"
                                "قد تحتاج إلى تثبيت مكتبات SDL أو التأكد من عمل جهاز الصوت.\n"
                                "التشغيل لن يعمل.")
        self.current_segment_ref = None # Reference to the playing AudioSegment object
        self.is_playing = False
        self.repeat_count = 0
        self.remaining_repeats = 0
        self.on_playback_finished_callback = None # Renamed for clarity

    def play_segment(self, segment: AudioSegment, repeat_count=0, on_finished=None):
        """Plays a specific audio segment with optional repeat."""
        if not self.mixer_initialized:
            print("Audio player mixer not initialized.")
            if on_finished: on_finished()
            return False

        if self.is_playing:
            self.stop() # Stop previous playback

        temp_file_path = segment.create_temp_file()
        if not temp_file_path:
            print(f"فشل في إنشاء ملف مؤقت للمقطع {segment.start_time:.2f}s.")
            if on_finished: on_finished()
            return False

        try:
            print(f"Starting playback: {temp_file_path}, Repeat: {repeat_count}")
            pygame.mixer.music.load(temp_file_path)
            # Play once initially. Repeats handled by update()
            pygame.mixer.music.play(loops=0)

            self.current_segment_ref = segment # Keep reference
            self.is_playing = True
            self.repeat_count = max(0, repeat_count)
            self.remaining_repeats = self.repeat_count
            self.on_playback_finished_callback = on_finished
            return True
        except pygame.error as e:
            print(f"Pygame error trying to play {temp_file_path}: {e}")
            self.is_playing = False
            self.current_segment_ref = None
            self.remaining_repeats = 0
            # Attempt to unload if load succeeded but play failed
            try: pygame.mixer.music.unload()
            except: pass
            if on_finished: on_finished()
            return False
        except Exception as e:
             print(f"Unexpected error during playback setup: {e}")
             self.is_playing = False
             self.current_segment_ref = None
             self.remaining_repeats = 0
             try: pygame.mixer.music.unload()
             except: pass
             if on_finished: on_finished()
             return False

    def stop(self):
        """Stops the current playback immediately."""
        if not self.mixer_initialized: return
        if self.is_playing or pygame.mixer.music.get_busy():
            print("Stopping playback.")
            pygame.mixer.music.stop()
            pygame.mixer.music.unload() # Release the file lock

        # Release temp file associated with the stopped segment
        # if self.current_segment_ref:
             # self.current_segment_ref.release_temp_file() # Let __del__ handle this? Risky if ref still exists elsewhere.
             # It's safer to clean up temp files explicitly when segments are replaced or app closes.

        self.is_playing = False
        self.remaining_repeats = 0
        # Do not call the finished callback here, as stop was manual
        self.on_playback_finished_callback = None # Clear callback on manual stop
        self.current_segment_ref = None


    def is_busy(self):
        """Check if the player is currently playing audio."""
        if not self.mixer_initialized: return False
        # Considered busy if the music module is active
        return self.is_playing and pygame.mixer.music.get_busy()

    def update(self):
        """Should be called periodically to check playback status and handle repeats/finish."""
        if not self.mixer_initialized or not self.is_playing:
            return

        # Check if the music has stopped playing
        if not pygame.mixer.music.get_busy():
            if self.remaining_repeats > 0:
                self.remaining_repeats -= 1
                print(f"Repeating playback, repeats left: {self.remaining_repeats}")
                try:
                    # Replay the currently loaded music
                    pygame.mixer.music.play(loops=0)
                except pygame.error as e:
                    print(f"Pygame error during repeat playback: {e}")
                    # Stop everything on error during repeat
                    current_callback = self.on_playback_finished_callback # Save callback
                    self.stop()
                    if current_callback:
                         current_callback() # Notify original caller of failure/stop
            else:
                # Playback finished (including all repeats)
                print("Playback finished (including repeats).")
                current_callback = self.on_playback_finished_callback # Save callback before stop clears it
                segment_ref = self.current_segment_ref # Save ref before stop clears it

                self.stop() # Clean up player state

                # Now call the callback if it exists
                if current_callback:
                    current_callback()


    def cleanup(self):
        """Cleans up pygame mixer resources."""
        if self.mixer_initialized:
            print("Cleaning up pygame.mixer...")
            self.stop() # Stop any playback
            pygame.mixer.quit() # Uninitialize mixer
            self.mixer_initialized = False
            print("Pygame mixer cleanup complete.")


# --- النافذة الرئيسية للتطبيق (MainWindow Class - Revised UI) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("مقسم ومكرر الصوت") # Renamed title
        self.setMinimumSize(850, 700) # Adjusted size

        # State variables
        self.audio_file = None
        self.all_segments = [] # Renamed for clarity
        self.current_segment_index = -1
        self.processing_thread = None
        self.audio_player = AudioPlayer()
        # Removed self.is_auto_playing, logic now depends directly on checkbox state

        # Timer for player updates
        self.update_timer_interval = 150 # ms (check less frequently is fine)
        self.update_timer_id = self.startTimer(self.update_timer_interval)

        self.setup_ui()
        self.ensure_temp_dir() # Make sure temp dir exists on startup

    def ensure_temp_dir(self):
         """Creates the temp directory if it doesn't exist."""
         try:
             temp_dir = tempfile.gettempdir()
             os.makedirs(temp_dir, exist_ok=True)
             print(f"Temp directory ensured: {temp_dir}")
         except Exception as e:
              print(f"Error ensuring temp directory: {e}")
              QMessageBox.warning(self, "خطأ في الدليل المؤقت",
                                 f"لا يمكن إنشاء أو الوصول إلى دليل الملفات المؤقتة:\n{temp_dir}\nقد تفشل بعض العمليات.")

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)

        # Apply font and layout direction
        arabic_font = QFont("Segoe UI", 10) # Common modern font
        self.setFont(arabic_font)
        self.setLayoutDirection(Qt.RightToLeft)
        main_widget.setLayoutDirection(Qt.RightToLeft)

        # --- 1. File Selection Group ---
        file_group = QGroupBox("اختيار الملف الصوتي")
        file_group.setLayoutDirection(Qt.RightToLeft)
        file_layout = QHBoxLayout()
        self.file_label = QLabel("لم يتم اختيار ملف")
        self.file_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.file_label.setWordWrap(True) # Allow wrapping for long paths
        self.file_button = QPushButton("اختيار ملف...")
        self.file_button.setIcon(QIcon.fromTheme("document-open", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-open-16.png"))) # Fallback icon
        self.file_button.setToolTip("اختر ملف صوتي للمعالجة (MP3, WAV, OGG, FLAC, M4A)")
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label, 1) # Give label more space
        file_group.setLayout(file_layout)

        # --- 2. Segmentation Settings Group ---
        settings_group = QGroupBox("إعدادات التقسيم")
        settings_group.setLayoutDirection(Qt.RightToLeft)
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(8)

        # -- Silence Threshold --
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("عتبة الصمت (dBFS):")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(-80, -10) # Wider range: -80 (very sensitive) to -10 (less sensitive)
        self.threshold_slider.setValue(-45)      # Default value
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setToolTip("أقل مستوى للصوت ليعتبر 'صوت'. القيم الأقل (أكثر سلبية) أكثر حساسية للضوضاء المنخفضة.")
        self.threshold_value_label = QLabel(f"{self.threshold_slider.value()} dB")
        self.threshold_value_label.setMinimumWidth(55)
        self.threshold_slider.valueChanged.connect(lambda val: self.threshold_value_label.setText(f"{val} dB"))
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider, 1)
        threshold_layout.addWidget(self.threshold_value_label)

        # -- Min Silence Duration --
        duration_layout = QHBoxLayout()
        duration_label = QLabel("أقل مدة للصمت (ثانية):")
        # Use QDoubleSpinBox for finer control over seconds
        self.min_silence_spinbox = QSpinBox() # Keep using spinbox for ms internally? No, let's use seconds.
        # Let's use a slider still, but maybe map values differently or use spinbox.
        # Sticking with slider for now, range 1 to 30 maps to 0.1s to 3.0s
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(1, 50) # 0.1s to 5.0s
        self.duration_slider.setValue(7)     # Default 0.7s
        self.duration_slider.setTickPosition(QSlider.TicksBelow)
        self.duration_slider.setTickInterval(5)
        self.duration_slider.setToolTip("أقصر مدة صمت (بالثواني) يجب أن تستمر ليتم اعتبارها فاصل بين المقاطع.")
        self.duration_value_label = QLabel(f"{self.duration_slider.value() / 10.0:.1f} ث")
        self.duration_value_label.setMinimumWidth(60)
        self.duration_slider.valueChanged.connect(lambda val: self.duration_value_label.setText(f"{val / 10.0:.1f} ث"))
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_slider, 1)
        duration_layout.addWidget(self.duration_value_label)

        # -- Max Segment Duration (New) --
        max_duration_layout = QHBoxLayout()
        self.max_duration_checkbox = QCheckBox("تحديد مدة قصوى للمقطع:")
        self.max_duration_checkbox.setToolTip("إذا تم تفعيل هذا الخيار، سيتم تقسيم المقاطع الطويلة جداً\nحتى لو لم تحتوِ على صمت كافٍ.")
        self.max_duration_spinbox = QSpinBox()
        self.max_duration_spinbox.setRange(5, 180) # 5 seconds to 3 minutes
        self.max_duration_spinbox.setValue(30)     # Default 30 seconds
        self.max_duration_spinbox.setSuffix(" ثانية")
        self.max_duration_spinbox.setEnabled(False) # Disabled by default
        self.max_duration_spinbox.setMinimumWidth(80)
        self.max_duration_checkbox.toggled.connect(self.max_duration_spinbox.setEnabled) # Enable/disable spinbox
        max_duration_layout.addWidget(self.max_duration_checkbox)
        max_duration_layout.addWidget(self.max_duration_spinbox)
        max_duration_layout.addStretch()

        # -- Process Button & Progress Bar --
        process_layout = QHBoxLayout()
        self.process_button = QPushButton("معالجة وتقسيم")
        self.process_button.setIcon(QIcon.fromTheme("media-record", QIcon(":/qt-project.org/styles/commonstyle/images/media-record-32.png")))
        self.process_button.setEnabled(False) # Disabled until file selected
        self.process_button.setFixedHeight(30) # Make button slightly taller
        self.process_button.clicked.connect(self.process_audio)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFixedHeight(30)
        process_layout.addWidget(self.process_button, 1)
        process_layout.addWidget(self.progress_bar, 2)

        settings_layout.addLayout(threshold_layout)
        settings_layout.addLayout(duration_layout)
        settings_layout.addLayout(max_duration_layout) # Add new max duration controls
        settings_layout.addLayout(process_layout)
        settings_group.setLayout(settings_layout)

        # --- 3. Playback Control Group ---
        playback_group = QGroupBox("التحكم بالتشغيل")
        playback_group.setLayoutDirection(Qt.RightToLeft)
        playback_main_layout = QHBoxLayout() # Main layout for this group

        # -- Left side: Repeat and Auto-play options --
        options_layout = QVBoxLayout()
        options_layout.setSpacing(10)

        # Repeat controls
        repeat_groupbox = QGroupBox("التكرار")
        repeat_groupbox.setLayoutDirection(Qt.RightToLeft)
        repeat_layout = QVBoxLayout()
        self.enable_repeat_checkbox = QCheckBox("تكرار المقطع المحدد")
        self.enable_repeat_checkbox.setChecked(True)
        repeat_spin_layout = QHBoxLayout()
        repeat_label = QLabel("عدد مرات التشغيل الإجمالي:") # Changed label for clarity
        self.repeat_count_spinbox = QSpinBox()
        self.repeat_count_spinbox.setRange(1, 50) # Play 1 time (no repeat) up to 50 times total
        self.repeat_count_spinbox.setValue(3)    # Default: play 3 times total
        self.repeat_count_spinbox.setToolTip("العدد الإجمالي لمرات تشغيل المقطع المحدد (1 يعني بدون تكرار).")
        self.repeat_count_spinbox.setMinimumWidth(60)
        repeat_spin_layout.addWidget(repeat_label)
        repeat_spin_layout.addWidget(self.repeat_count_spinbox)
        repeat_spin_layout.addStretch()
        repeat_layout.addWidget(self.enable_repeat_checkbox)
        repeat_layout.addLayout(repeat_spin_layout)
        repeat_groupbox.setLayout(repeat_layout)

        # Auto-play next control
        self.auto_play_next_checkbox = QCheckBox("تشغيل المقطع التالي تلقائياً")
        self.auto_play_next_checkbox.setChecked(False) # Default off
        self.auto_play_next_checkbox.setToolTip("عند انتهاء المقطع الحالي، يتم تشغيل المقطع التالي في القائمة تلقائياً.")

        options_layout.addWidget(repeat_groupbox)
        options_layout.addWidget(self.auto_play_next_checkbox)
        options_layout.addStretch()

        # -- Right side: Playback buttons --
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(5)
        buttons_row1 = QHBoxLayout()
        self.play_pause_button = QPushButton("تشغيل")
        self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-start", QIcon(":/qt-project.org/styles/commonstyle/images/media-play-32.png")))
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setToolTip("تشغيل المقطع المحدد (أو المقطع الأول إذا لم يتم تحديد شيء)")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.stop_button = QPushButton("إيقاف")
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop", QIcon(":/qt-project.org/styles/commonstyle/images/media-stop-32.png")))
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("إيقاف التشغيل الحالي")
        self.stop_button.clicked.connect(self.stop_playback)
        buttons_row1.addWidget(self.play_pause_button)
        buttons_row1.addWidget(self.stop_button)

        buttons_row2 = QHBoxLayout()
        self.prev_button = QPushButton("السابق")
        self.prev_button.setIcon(QIcon.fromTheme("media-skip-backward", QIcon(":/qt-project.org/styles/commonstyle/images/media-seek-backward-32.png")))
        self.prev_button.setEnabled(False)
        self.prev_button.setToolTip("الانتقال إلى المقطع السابق وتشغيله")
        self.prev_button.clicked.connect(self.play_previous_segment)
        self.next_button = QPushButton("التالي")
        self.next_button.setIcon(QIcon.fromTheme("media-skip-forward", QIcon(":/qt-project.org/styles/commonstyle/images/media-seek-forward-32.png")))
        self.next_button.setEnabled(False)
        self.next_button.setToolTip("الانتقال إلى المقطع التالي وتشغيله")
        self.next_button.clicked.connect(self.play_next_segment)
        buttons_row2.addWidget(self.prev_button)
        buttons_row2.addWidget(self.next_button)

        buttons_layout.addLayout(buttons_row1)
        buttons_layout.addLayout(buttons_row2)
        buttons_layout.addStretch()


        playback_main_layout.addLayout(options_layout, 1) # Options on the left
        playback_main_layout.addLayout(buttons_layout, 1) # Buttons on the right
        playback_group.setLayout(playback_main_layout)

        # --- 4. Waveform Display ---
        self.waveform_widget = AudioWaveform()

        # --- 5. Segments List ---
        segments_group = QGroupBox("قائمة المقاطع")
        segments_group.setLayoutDirection(Qt.RightToLeft)
        segments_layout = QVBoxLayout()
        self.segments_listwidget = QListWidget()
        self.segments_listwidget.setAlternatingRowColors(True)
        self.segments_listwidget.itemClicked.connect(self.on_segment_selected_from_list)
        self.segments_listwidget.itemDoubleClicked.connect(self.on_segment_double_clicked)
        self.segments_listwidget.setToolTip("انقر لتحديد مقطع، انقر نقراً مزدوجاً لتشغيله.")
        segments_layout.addWidget(self.segments_listwidget)
        segments_group.setLayout(segments_layout)

        # --- Add widgets to main layout ---
        main_layout.addWidget(file_group)
        main_layout.addWidget(settings_group)
        main_layout.addWidget(playback_group)
        main_layout.addWidget(self.waveform_widget, 1) # Give waveform more flexible space
        main_layout.addWidget(segments_group, 2)      # Give segment list more flexible space

        self.setCentralWidget(main_widget)
        self.update_playback_buttons_state() # Initial button state

    # --- UI Interaction and Event Handling ---

    def select_file(self):
        """Opens a dialog to select an audio file."""
        # Use user's Desktop directory as a starting point, or fallback to home dir
        start_dir = os.path.expanduser("~/Desktop")
        if not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف صوتي",
            start_dir, # Start in a common directory
            "ملفات الصوت (*.mp3 *.wav *.ogg *.flac *.m4a);;كل الملفات (*.*)"
        )

        if file_path:
            self.reset_processing_state() # Reset before loading new file
            self.audio_file = file_path
            file_name = os.path.basename(file_path)
            self.file_label.setText(f"{file_name}") # Show only filename for brevity
            self.file_label.setToolTip(file_path) # Show full path in tooltip
            self.process_button.setEnabled(True)
            print(f"File selected: {file_path}")
            # Load and display waveform immediately
            self.waveform_widget.set_audio(file_path)
            self.update_playback_buttons_state()


    def cleanup_old_segments(self):
        """Explicitly releases resources of old segments."""
        print(f"Cleaning up {len(self.all_segments)} old segment(s)...")
        for segment in self.all_segments:
            segment.release_temp_file()
        self.all_segments = [] # Clear the list


    def reset_processing_state(self):
        """Resets state related to audio processing and segments."""
        self.stop_playback() # Stop any current playback
        # Terminate existing thread if running
        if self.processing_thread and self.processing_thread.isRunning():
             print("Terminating previous processing thread...")
             self.processing_thread.terminate()
             self.processing_thread.wait(1000) # Wait a bit for termination

        self.cleanup_old_segments() # Clean up previous segments

        self.audio_file = None
        self.file_label.setText("لم يتم اختيار ملف")
        self.file_label.setToolTip("")
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.segments_listwidget.clear()
        self.waveform_widget.set_audio(None) # Clear waveform display
        self.current_segment_index = -1
        self.update_playback_buttons_state()
        # Do not reset segmentation parameters (threshold, duration, etc.)


    def process_audio(self):
        """Starts the audio segmentation process in a separate thread."""
        if not self.audio_file or not os.path.exists(self.audio_file):
            QMessageBox.warning(self, "ملف غير موجود", "الرجاء اختيار ملف صوتي صالح أولاً.")
            return

        if not PYDUB_AVAILABLE:
             QMessageBox.critical(self, "خطأ في التبعيات",
                                  "مكتبة pydub غير متاحة أو لم يتم تهيئتها بنجاح (قد يكون السبب ffmpeg).\nلا يمكن تقسيم الصوت.")
             return


        self.stop_playback()
        self.cleanup_old_segments() # Clean segments before starting new process

        # Disable UI elements during processing
        self.set_ui_enabled(False)
        self.process_button.setText("جاري المعالجة...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Get settings from UI
        silence_threshold = self.threshold_slider.value()
        min_silence_duration = self.duration_slider.value() / 10.0 # Convert slider value to seconds
        enable_max_dur = self.max_duration_checkbox.isChecked()
        max_segment_dur = self.max_duration_spinbox.value() # Already in seconds

        # Clear previous results
        self.segments_listwidget.clear()
        self.waveform_widget.update_segments([], -1) # Clear segments on waveform
        self.current_segment_index = -1
        self.update_playback_buttons_state()

        # Start the processing thread
        print("Starting AudioProcessingThread...")
        self.processing_thread = AudioProcessingThread(
            self.audio_file,
            silence_threshold,
            min_silence_duration,
            enable_max_dur,
            max_segment_dur
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_done.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.finished.connect(self.on_thread_actually_finished) # For cleanup
        self.processing_thread.start()


    def set_ui_enabled(self, enabled):
        """Enables or disables UI elements during processing."""
        self.file_button.setEnabled(enabled)
        # Keep process button managed separately
        # self.process_button.setEnabled(enabled if self.audio_file else False)
        self.threshold_slider.setEnabled(enabled)
        self.duration_slider.setEnabled(enabled)
        self.max_duration_checkbox.setEnabled(enabled)
        # Only enable spinbox if checkbox is checked AND UI is enabled
        self.max_duration_spinbox.setEnabled(enabled and self.max_duration_checkbox.isChecked())
        self.enable_repeat_checkbox.setEnabled(enabled)
        self.repeat_count_spinbox.setEnabled(enabled)
        self.auto_play_next_checkbox.setEnabled(enabled)
        self.segments_listwidget.setEnabled(enabled)
        # Playback buttons are handled by update_playback_buttons_state


    def update_progress(self, value):
        """Updates the progress bar value."""
        self.progress_bar.setValue(value)


    def on_processing_finished(self, processed_segments):
        """Handles successful completion of the processing thread."""
        print(f"Processing thread finished successfully. Received {len(processed_segments)} segments.")
        self.all_segments = processed_segments # Store the new segments

        self.segments_listwidget.clear() # Clear list widget first
        if not self.all_segments:
            QMessageBox.information(self, "لا توجد مقاطع", "لم يتم العثور على مقاطع صوتية بناءً على الإعدادات الحالية أو أن الملف صامت.")
            self.waveform_widget.update_segments([]) # Make sure waveform is cleared
        else:
            # Populate the list widget
            for i, segment in enumerate(self.all_segments):
                # Format: المقطع 1: [0.00s - 5.32s] (5.32 ث)
                item_text = (f"المقطع {i+1}: "
                             f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                             f"({segment.duration:.2f} ث)")
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i) # Store segment index
                self.segments_listwidget.addItem(item)

            # Update waveform with new segments
            self.waveform_widget.update_segments(self.all_segments, -1) # Show all segments, none selected yet
            print(f"Populated list and waveform with {len(self.all_segments)} segments.")
            # Select the first item automatically? Optional.
            # if self.all_segments:
            #     self.select_segment(0)
            #     self.segments_listwidget.setCurrentRow(0)


    def on_processing_error(self, error_msg):
        """Handles errors reported by the processing thread."""
        print(f"Processing error reported: {error_msg}")
        QMessageBox.critical(self, "خطأ في المعالجة", f"فشلت عملية تقسيم الصوت:\n{error_msg}")
        # No segments were generated or they are invalid
        self.cleanup_old_segments()
        self.segments_listwidget.clear()
        self.waveform_widget.update_segments([]) # Clear waveform
        self.current_segment_index = -1
        # UI state will be reset in on_thread_actually_finished


    def on_thread_actually_finished(self):
        """Called when the processing thread truly finishes (success, error, or termination)."""
        print("Processing thread finished signal received.")
        self.processing_thread = None # Clear thread reference
        self.reset_ui_after_processing()


    def reset_ui_after_processing(self):
        """Resets the UI elements after processing is complete."""
        self.process_button.setText("معالجة وتقسيم")
        # Enable process button only if a valid file is selected
        self.process_button.setEnabled(bool(self.audio_file and os.path.exists(self.audio_file)))
        self.progress_bar.setVisible(False)
        self.set_ui_enabled(True) # Re-enable other UI elements
        self.update_playback_buttons_state() # Update playback buttons based on new state
        print("UI re-enabled after processing.")


    def on_segment_selected_from_list(self, item):
        """Handles selection changes in the segments list."""
        if item: # Ensure item is valid
             segment_index = item.data(Qt.UserRole)
             self.select_segment(segment_index)


    def on_segment_double_clicked(self, item):
        """Handles double-clicks on a segment in the list (play it)."""
        if item:
            segment_index = item.data(Qt.UserRole)
            print(f"Segment {segment_index + 1} double-clicked.")
            self.select_and_play_segment(segment_index)


    def select_segment(self, index):
        """Selects a segment by index and updates UI (waveform, list selection). Does not play."""
        if 0 <= index < len(self.all_segments):
            if index != self.current_segment_index:
                 print(f"Selecting segment {index + 1}.")
                 self.current_segment_index = index
                 self.segments_listwidget.setCurrentRow(index) # Ensure list highlights the correct item
                 self.waveform_widget.update_segments(self.all_segments, index) # Highlight in waveform
                 self.update_playback_buttons_state()
        else:
            print(f"Invalid segment index to select: {index}")
            # Optionally deselect
            # self.current_segment_index = -1
            # self.segments_listwidget.setCurrentRow(-1)
            # self.waveform_widget.update_segments(self.all_segments, -1)
            # self.update_playback_buttons_state()


    def select_and_play_segment(self, segment_index):
        """Selects and plays a specific segment."""
        if not (0 <= segment_index < len(self.all_segments)):
            print(f"Invalid segment index to play: {segment_index}")
            if not self.all_segments:
                 QMessageBox.information(self, "لا توجد مقاطع", "لا يمكن التشغيل، قم بمعالجة الملف أولاً.")
            return

        # First, select the segment visually
        self.select_segment(segment_index)

        segment_to_play = self.all_segments[segment_index]

        # Determine repeat count based on UI
        total_plays = 1 # Default: play once
        if self.enable_repeat_checkbox.isChecked():
            total_plays = self.repeat_count_spinbox.value()

        # pydub's play(loops=N) plays N+1 times. So loops = total_plays - 1
        repeat_loops = max(0, total_plays - 1)

        print(f"Requesting playback of segment {segment_index + 1} ({segment_to_play.duration:.2f}s). Total plays: {total_plays} (loops={repeat_loops}).")

        # Play using the audio player, providing the callback
        success = self.audio_player.play_segment(
            segment_to_play,
            repeat_loops,
            on_finished=self.on_playback_finished # Pass method reference
        )

        if success:
            self.update_playback_buttons_state() # Update buttons to show 'Stop'/'Pause' state
        else:
            QMessageBox.warning(self, "خطأ في التشغيل", f"لم يتمكن من تشغيل المقطع {segment_index + 1}.")
            self.update_playback_buttons_state() # Update buttons state even on failure


    def on_playback_finished(self):
        """Callback executed by AudioPlayer when playback (including repeats) finishes naturally."""
        # Ensure this runs in the main GUI thread
        if QThread.currentThread() != self.thread():
             #print("Deferring on_playback_finished to GUI thread.")
             QMetaObject.invokeMethod(self, "on_playback_finished", Qt.QueuedConnection)
             return

        print("MainWindow received on_playback_finished signal.")

        # Check if auto-play next is enabled
        if self.auto_play_next_checkbox.isChecked():
            print("Auto-play next is enabled.")
            current_idx = self.current_segment_index
            next_idx = current_idx + 1
            if 0 <= current_idx < len(self.all_segments) -1 :
                 print(f"Attempting to auto-play next segment ({next_idx + 1}).")
                 # Check if player is actually stopped before starting next
                 # Small delay might be needed if pygame events are slightly behind
                 # QtCore.QTimer.singleShot(50, lambda: self.select_and_play_segment(next_idx))
                 self.select_and_play_segment(next_idx) # Try immediate first
            else:
                 print("Last segment finished, auto-play stopped.")
                 self.update_playback_buttons_state() # Update buttons to show 'Play' state
        else:
             print("Auto-play next is disabled or finished.")
             self.update_playback_buttons_state() # Update buttons to show 'Play' state


    def play_next_segment(self):
        """Plays the next segment in the list."""
        if not self.all_segments: return
        next_index = self.current_segment_index + 1
        if next_index >= len(self.all_segments):
            next_index = 0 # Wrap around to the first segment? Or stop? Let's stop.
            print("Already at the last segment.")
            self.stop_playback()
            return
        self.select_and_play_segment(next_index)


    def play_previous_segment(self):
        """Plays the previous segment in the list."""
        if not self.all_segments: return
        prev_index = self.current_segment_index - 1
        if prev_index < 0:
            prev_index = len(self.all_segments) - 1 # Wrap around to the last segment? Or stop? Let's stop at first.
            print("Already at the first segment.")
            # Option: Play the first segment again if already there
            # self.select_and_play_segment(0)
            return
        self.select_and_play_segment(prev_index)


    def toggle_play_pause(self):
        """Toggles playback of the currently selected segment."""
        if self.audio_player.is_busy():
            # If playing, stop it (no pause implemented here)
            print("Toggle Play/Pause: Stopping current playback.")
            self.stop_playback()
        else:
            # If not playing, start playing the current or first segment
            print("Toggle Play/Pause: Starting playback.")
            target_index = self.current_segment_index
            if not (0 <= target_index < len(self.all_segments)):
                 if self.all_segments:
                     target_index = 0 # Default to first segment if none selected
                     print("No segment selected, starting from first.")
                 else:
                      QMessageBox.information(self, "لا يوجد مقاطع", "لا توجد مقاطع لتشغيلها. قم بمعالجة الملف أولاً.")
                      return # No segments to play
            self.select_and_play_segment(target_index)


    def stop_playback(self):
        """Stops playback immediately."""
        if self.audio_player.is_playing:
             print("Manual stop requested.")
             self.audio_player.stop() # Call the player's stop method
             # Auto-play should naturally stop because the finish callback won't be called for manual stops
             self.update_playback_buttons_state() # Update UI


    def update_playback_buttons_state(self):
        """Updates the enabled state and text/icon of playback buttons."""
        has_segments = bool(self.all_segments)
        is_playing = self.audio_player.is_busy() # Check if player is actively outputting sound

        # Play/Pause Button
        self.play_pause_button.setEnabled(has_segments)
        if is_playing:
            self.play_pause_button.setText("إيقاف") # Simple Play/Stop toggle
            self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-stop", QIcon(":/qt-project.org/styles/commonstyle/images/media-stop-32.png")))
            self.play_pause_button.setToolTip("إيقاف التشغيل الحالي")
        else:
            self.play_pause_button.setText("تشغيل")
            self.play_pause_button.setIcon(QIcon.fromTheme("media-playback-start", QIcon(":/qt-project.org/styles/commonstyle/images/media-play-32.png")))
            self.play_pause_button.setToolTip("تشغيل المقطع المحدد (أو الأول)")

        # Stop Button (maybe redundant if Play/Pause becomes Play/Stop, but keep for clarity?)
        self.stop_button.setEnabled(is_playing) # Enabled only when playing

        # Previous/Next Buttons
        can_go_prev = has_segments and self.current_segment_index > 0
        can_go_next = has_segments and self.current_segment_index < len(self.all_segments) - 1

        self.prev_button.setEnabled(can_go_prev)
        self.next_button.setEnabled(can_go_next)

    # --- Timer and Close Event ---

    def timerEvent(self, event):
        """Periodically called by Qt's timer."""
        # Update the audio player state (checks for finished playback/repeats)
        self.audio_player.update()

    def closeEvent(self, event):
        """Handles the application closing event."""
        print("Close event triggered. Cleaning up...")
        # Stop the timer
        if self.update_timer_id is not None:
            self.killTimer(self.update_timer_id)
            self.update_timer_id = None

        # Stop any ongoing playback and cleanup player resources
        self.audio_player.cleanup()

        # Terminate processing thread if it's still running
        if self.processing_thread and self.processing_thread.isRunning():
            print("Terminating processing thread on close...")
            self.processing_thread.terminate()
            self.processing_thread.wait(1500) # Wait a bit longer

        # Explicitly clean up segment resources
        self.cleanup_old_segments()
        print("Cleanup finished. Accepting close event.")

        event.accept() # Proceed with closing


# --- Application Entry Point ---
def main():
    # Try initializing pygame main module early to catch some errors
    try:
        pygame.init()
        pygame_initialized = True
    except Exception as e:
        print(f"Error initializing pygame: {e}")
        QMessageBox.critical(None, "خطأ في Pygame",
                             f"لم يتمكن من تهيئة Pygame الرئيسي:\n{e}\nقد لا يعمل مشغل الصوت.")
        pygame_initialized = False


    app = QApplication(sys.argv)

    # Apply Right-to-Left globally if needed (often better per-widget)
    # app.setLayoutDirection(Qt.RightToLeft)

    # Check pydub/ffmpeg status after imports
    if not PYDUB_AVAILABLE and not FFMPEG_WARNING_ISSUED:
         # Show warning if pydub is missing but ffmpeg wasn't the detected issue
         QMessageBox.warning(None, "تحذير: Pydub مفقود",
                             "مكتبة Pydub غير مثبتة أو غير قابلة للاستيراد.\nوظيفة تقسيم الصوت لن تعمل.\n"
                             "يرجى التثبيت: pip install pydub")
    elif FFMPEG_WARNING_ISSUED:
          QMessageBox.warning(None, "تحذير: مشكلة في FFmpeg",
                             "تم اكتشاف مشكلة محتملة في العثور على FFmpeg أو استخدامه.\n"
                             "وظيفة تقسيم الصوت قد لا تعمل بشكل صحيح.\n"
                             "تأكد من تثبيت FFmpeg وإضافته إلى PATH.")


    window = MainWindow()
    window.show()

    exit_code = app.exec_()

    # Quit pygame if it was initialized
    if pygame_initialized:
        pygame.quit()
        print("Pygame quit.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()