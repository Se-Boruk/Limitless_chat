# AI_Assistant_main/main.py

#App frontend lib
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QComboBox, QToolBar,
    QGroupBox, QFormLayout, QSlider, QSplitter, QTextEdit, QMessageBox, QStackedWidget
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QProgressBar, QCheckBox
from PySide6.QtWidgets import QScrollArea, QLabel, QSizePolicy

#Backend libs
import threading, time, sys, os
from llm.model_loader import LocalLLM
from transformers import TextIteratorStreamer, logging

#Own libs (backend)
import Modules
import Functions

#Settings from config
from dir_config import BOT_NAME, BOT_DEV_NAME
from dir_config import MODEL_DIR, DOC_LIB_DIR, VECTOR_LIB_DIR

import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DEF_CONFIG_PATH = os.path.join(BASE_DIR, "config_default.json")




class IngestWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(int, int)
    error = Signal(str)

    def __init__(self, vector_lib, data_dir):
        super().__init__()
        self.vector_lib = vector_lib
        
        self.vector_lib.load_model_fp32(load_cross_encoder = False)
        
        self.data_dir = data_dir

    def run(self):
        # Accept multiple extensions supported by the universal reader
        supported_exts = (".pdf", ".docx", ".txt", ".xml", ".epub", ".html", ".htm")
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(supported_exts)]
        
        total = len(files)
        ingested = 0
        skipped = 0

        for idx, filename in enumerate(files, 1):
            if self.vector_lib.file_already_ingested(filename):  # renamed from pdf_already_ingested
                skipped += 1
            else:
                file_path = os.path.join(self.data_dir, filename)
                try:
                    self.vector_lib.add_file(file_path)  # universal ingestion method, renamed from add_pdf
                    ingested += 1
                except Exception as e:
                    self.error.emit(f"Error ingesting {filename}: {e}")
            self.progress.emit(idx, total)

        self.finished.emit(ingested, skipped)
        
        
class SettingsWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Settings")
        self.resize(1200, 700)

        # Keep reference to main window for future use
        self.main_window = main_window

        # Toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Back", self, triggered=self.close))

        # Central widget (empty for now)
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(BOT_DEV_NAME)
        self.resize(1200, 700)
        self._apply_theme()
        
        #Loading config data
        with open(CONFIG_PATH, "r") as f:
            self.CONFIG = json.load(f)
    
        self.llm = LocalLLM()
        self.model = None
        self.tokenizer = None
        self.chat_history = []
    
        self.Vector_lib = Modules.UniversalVectorStore(
            DOC_LIB_DIR, VECTOR_LIB_DIR,
            self.CONFIG['rag_params']['chunk_size'],
            self.CONFIG['rag_params']['overlap_ratio'],
            self.CONFIG['rag_params']["RAG_batch_size"]
        )
        #self.Vector_lib.load_model_fp32() #optional
        self.Vector_lib.load_model_fp16()
        
        
        if self.CONFIG["other"]["launch_tor_on_start"]:
            
            self.launch_TOR_server()
            

            
            
        

            
            

        # --- Toolbar ---
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Main chatbot", self, triggered=self.show_main))
        toolbar.addAction(QAction("Settings", self, triggered=self.show_settings))
        toolbar.addAction(QAction("Quit", self, triggered=self.close))
    
        # --- Central stacked widget ---
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
    
        # --- MAIN PANEL (splitter layout with full left/right content) ---
        self.main_panel = QWidget()
        main_layout = QVBoxLayout(self.main_panel)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
    
        # --- LEFT PANEL ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
    
        # Model Info Group
        model_group = QGroupBox("Model Info")
        info_layout = QFormLayout()
        info_layout.setContentsMargins(10, 20, 10, 10)
        self.model_dropdown = QComboBox()
        for d in os.listdir(MODEL_DIR):
            if os.path.isdir(os.path.join(MODEL_DIR, d)):
                self.model_dropdown.addItem(d)
                
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        info_layout.addRow("Select Model:", self.model_dropdown)
        info_layout.addRow(load_btn)
        
        clear_btn = QPushButton("Clear chat")
        clear_btn.clicked.connect(self.clear_chat)
        info_layout.addRow(clear_btn)
    
        self.status_label = QTextEdit("No model loaded")
        self.status_label.setReadOnly(True)
        self.status_label.setMaximumHeight(60)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout.addRow("Status:", self.status_label)
        model_group.setLayout(info_layout)
        left_layout.addWidget(model_group)
    

        # RAG Status Group
        rag_status_group = QGroupBox("RAG Status")
        rag_status_layout = QVBoxLayout()
        rag_status_layout.setContentsMargins(10, 20, 10, 10)
        rag_status_layout.setSpacing(8)
    
        self.RAG_add_btn = QPushButton("Update RAG lib")
        self.RAG_add_btn.clicked.connect(self.on_update_rag_lib)
        rag_status_layout.addWidget(self.RAG_add_btn)
    
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setVisible(False)
        rag_status_layout.addWidget(self.progress_bar)
    
        self.rag_status_box = QTextEdit()
        self.rag_status_box.setReadOnly(True)
        self.rag_status_box.setMaximumHeight(80)
        self.rag_status_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.rag_status_box.setPlainText("RAG system not initialized.")
        rag_status_layout.addWidget(self.rag_status_box)
    
        # RAG options
        self.RAG_local = QCheckBox("Local RAG")
        self.RAG_local.setChecked(True)
        self.RAG_online = QCheckBox("Online RAG")
        self.RAG_online.setChecked(True)
        self.RAG_online_TOR_search = QCheckBox("Online RAG - TOR search")
        self.RAG_online_TOR_search.setChecked(False)
        self.RAG_online_dark_TOR_search = QCheckBox("Online RAG - Dark TOR search")
        self.RAG_online_dark_TOR_search.setChecked(False)
    
        self.RAG_local.stateChanged.connect(self.update_rag_dependencies)
        self.RAG_online.stateChanged.connect(self.update_rag_dependencies)
        self.RAG_online_TOR_search.stateChanged.connect(self.update_rag_dependencies)
    
        rag_status_layout.addWidget(self.RAG_local)
        rag_status_layout.addWidget(self.RAG_online)
        rag_status_layout.addWidget(self.RAG_online_TOR_search)
        rag_status_layout.addWidget(self.RAG_online_dark_TOR_search)
    
        rag_status_group.setLayout(rag_status_layout)
        left_layout.addWidget(rag_status_group)
        left_layout.addStretch()
    
        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)
    
    
        #TOR / search engine group
        search_status_group = QGroupBox("Search group")
        search_status_layout = QVBoxLayout()
        search_status_layout.setContentsMargins(10, 20, 10, 10)
        search_status_layout.setSpacing(8)
        
        
        #Tor server switch
        self.TOR_server_btn = QCheckBox("Tor server")
        
        if self.CONFIG["other"]["launch_tor_on_start"]:
            self.TOR_server_btn.setChecked(True)
        else:
            self.TOR_server_btn.setChecked(False)
            
        self.TOR_server_btn.stateChanged.connect(self.manage_TOR_server)
        
        search_status_layout.addWidget(self.TOR_server_btn)
        
        search_status_group.setLayout(search_status_layout)
        left_layout.addWidget(search_status_group)
        #left_layout.addStretch()



        # --- RIGHT PANEL ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
    
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        right_layout.addWidget(self.log_area)
    
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("background-color: #1e222a;")
    
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch(1)  # keep messages top-aligned
        self.chat_scroll.setWidget(self.chat_widget)
    
        right_layout.addWidget(self.chat_scroll)
    
        input_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setMinimumHeight(40)
        self.input_line.returnPressed.connect(self.on_send)
        input_layout.addWidget(self.input_line)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.on_send)
        input_layout.addWidget(send_btn)
        right_layout.addLayout(input_layout)
    
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
    
        # Add main panel to stack
        self.stack.addWidget(self.main_panel)  # index 0
        
        
        
        ###################################################
        # --- SETTINGS PANEL ---
        ###################################################
        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout(self.settings_panel)
        
        # Splitter for 3 columns
        settings_splitter = QSplitter(Qt.Horizontal)
        settings_layout.addWidget(settings_splitter)
        
        # --- LEFT COLUMN ---
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Generation Params Group
        params_group = QGroupBox("Generation parameters")
        params_layout = QFormLayout()
        params_layout.setContentsMargins(10, 20, 10, 10)
        params_layout.setSpacing(8)
        
        
        #########################
        # Max tokens as a dropdown
        self.max_tokens_input = QComboBox()
        self.max_tokens_input.setEditable(False)  # user cannot type
        self.max_tokens_input.setMaximumHeight(25)
        
        # Add predefined options
        token_values = [64, 128, 256, 512, 786, 1024, 1536, 2048, 3072, 4096]
        for val in token_values:
            self.max_tokens_input.addItem(str(val))
        
        # Set default value
        self.max_tokens_input.setCurrentText(str(self.CONFIG['generation_params'].get("max_new_tokens", self.CONFIG['generation_params']['max_new_tokens'])))
        # Add to layout
        params_layout.addRow("Max gen. tokens:", self.max_tokens_input)
        #########################
        
        # Temperature
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.CONFIG['generation_params']['temperature']*100))
        self.temp_label = QLineEdit(str(self.CONFIG['generation_params']['temperature']))
        self.temp_label.setReadOnly(True)
        self.temp_label.setMaximumHeight(25)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/100:.2f}"))
        params_layout.addRow("Temperature:", self.temp_slider)
        params_layout.addRow("", self.temp_label)
        
        # Top P
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(int(self.CONFIG['generation_params']['top_p']*100))
        self.top_p_label = QLineEdit(str(self.CONFIG['generation_params']['top_p']))
        self.top_p_label.setReadOnly(True)
        self.top_p_label.setMaximumHeight(25)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_label.setText(f"{v/100:.2f}"))
        params_layout.addRow("Top P:", self.top_p_slider)
        params_layout.addRow("", self.top_p_label)
        
        # Rep penalty
        self.rep_penalty_slider = QSlider(Qt.Horizontal)
        self.rep_penalty_slider.setRange(100, 200)
        self.rep_penalty_slider.setValue(int(self.CONFIG['generation_params']['repetition_penalty']*100))
        self.rep_penalty_label = QLineEdit(str(self.CONFIG['generation_params']['repetition_penalty']))
        self.rep_penalty_label.setReadOnly(True)
        self.rep_penalty_label.setMaximumHeight(25)
        self.rep_penalty_slider.valueChanged.connect(lambda v: self.rep_penalty_label.setText(f"{v/100:.2f}"))
        params_layout.addRow("Repetition penalty:", self.rep_penalty_slider)
        params_layout.addRow("", self.rep_penalty_label)

        # No repeat ngram
        self.no_rep_ngram_slider = QSlider(Qt.Horizontal)
        self.no_rep_ngram_slider.setRange(0, 10)
        self.no_rep_ngram_slider.setValue(int(self.CONFIG['generation_params']['no_repeat_ngram_size']))
        self.no_rep_ngram_label = QLineEdit(str(self.CONFIG['generation_params']['no_repeat_ngram_size']))
        self.no_rep_ngram_label.setReadOnly(True)
        self.no_rep_ngram_label.setMaximumHeight(25)
        self.no_rep_ngram_slider.valueChanged.connect(lambda v: self.no_rep_ngram_label.setText(f"{v}"))
        params_layout.addRow("No rep. ngram size:", self.no_rep_ngram_slider)
        params_layout.addRow("", self.no_rep_ngram_label)
        

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Stretch pushes everything to the top
        left_layout.addStretch()
        
        settings_splitter.addWidget(left_col)
        settings_splitter.setStretchFactor(0, 1)  # equal width for all columns


        # --- middle COLUMN ---
        middle_col = QWidget()
        middle_layout = QVBoxLayout(middle_col)
        middle_layout.setContentsMargins(5, 5, 5, 5)
        middle_layout.setSpacing(10)
        
        # Generation Params Group
        rag_group = QGroupBox("RAG parameters")
        rag_layout = QFormLayout()
        rag_layout.setContentsMargins(10, 20, 10, 10)
        rag_layout.setSpacing(8)
        
        
        #rag max gen tokens
        ####################
        self.rag_max_tokens = QComboBox()
        self.rag_max_tokens.setEditable(False)  # user cannot type
        self.rag_max_tokens.setMaximumHeight(25)
        
        # Add predefined options
        rag_max_token_values = [64, 128, 256, 384, 512]
        for val in rag_max_token_values:
            self.rag_max_tokens.addItem(str(val))
        
        # Set default value
        self.rag_max_tokens.setCurrentText(str(self.CONFIG['rag_params'].get("RAG_max_new_tokens", self.CONFIG['rag_params']['RAG_max_new_tokens'])))
        # Add to layout
        rag_layout.addRow("RAG max gen. tokens:", self.rag_max_tokens)
        ####################
        
        # rag Temperature
        self.rag_temp_slider = QSlider(Qt.Horizontal)
        self.rag_temp_slider.setRange(0, 100)
        self.rag_temp_slider.setValue(int(self.CONFIG['rag_params']['RAG_temperature']*100))
        self.rag_temp_label = QLineEdit(str(self.CONFIG['rag_params']['RAG_temperature']))
        self.rag_temp_label.setReadOnly(True)
        self.rag_temp_label.setMaximumHeight(25)
        self.rag_temp_slider.valueChanged.connect(lambda v: self.rag_temp_label.setText(f"{v/100:.2f}"))
        rag_layout.addRow("RAG Temperature:", self.rag_temp_slider)
        rag_layout.addRow("", self.rag_temp_label)
        
        # rag Top P
        self.rag_top_p_slider = QSlider(Qt.Horizontal)
        self.rag_top_p_slider.setRange(0, 100)
        self.rag_top_p_slider.setValue(int(self.CONFIG['rag_params']['RAG_top_p']*100))
        self.rag_top_p_label = QLineEdit(str(self.CONFIG['rag_params']['RAG_top_p']))
        self.rag_top_p_label.setReadOnly(True)
        self.rag_top_p_label.setMaximumHeight(25)
        self.rag_top_p_slider.valueChanged.connect(lambda v: self.rag_top_p_label.setText(f"{v/100:.2f}"))
        rag_layout.addRow("RAG Top P:", self.rag_top_p_slider)
        rag_layout.addRow("", self.rag_top_p_label)
        
        #rag top n
        ####################
        self.top_n = QComboBox()
        self.top_n.setEditable(False)  # user cannot type
        self.top_n.setMaximumHeight(25)
        
        # Add predefined options
        n_values = [1,2,3,4,6,8,10,12]
        for val in n_values:
            self.top_n.addItem(str(val))
        
        # Set default value
        self.top_n.setCurrentText(str(self.CONFIG['rag_params'].get("top_n", self.CONFIG['rag_params']['top_n'])))
        # Add to layout
        rag_layout.addRow("Top n:", self.top_n)
        ####################
        
        
        ############
        # min_relevance
        self.min_relevance_slider = QSlider(Qt.Horizontal)
        self.min_relevance_slider.setRange(0, 100)
        self.min_relevance_slider.setValue(int(self.CONFIG['rag_params']['min_relevance']*100))
        self.min_relevance_label = QLineEdit(str(self.CONFIG['rag_params']['min_relevance']))
        self.min_relevance_label.setReadOnly(True)
        self.min_relevance_label.setMaximumHeight(25)
        self.min_relevance_slider.valueChanged.connect(lambda v: self.min_relevance_label.setText(f"{v/100:.2f}"))
        rag_layout.addRow("Min relevance:", self.min_relevance_slider)
        rag_layout.addRow("", self.min_relevance_label)
        
        # absolute cosine min
        self.absolute_cos_min_slider = QSlider(Qt.Horizontal)
        self.absolute_cos_min_slider.setRange(0, 100)
        self.absolute_cos_min_slider.setValue(int(self.CONFIG['rag_params']['absolute_cosine_min']*100))
        self.absolute_cos_min_label = QLineEdit(str(self.CONFIG['rag_params']['absolute_cosine_min']))
        self.absolute_cos_min_label.setReadOnly(True)
        self.absolute_cos_min_label.setMaximumHeight(25)
        self.absolute_cos_min_slider.valueChanged.connect(lambda v: self.absolute_cos_min_label.setText(f"{v/100:.2f}"))
        rag_layout.addRow("absolute cosine min:", self.absolute_cos_min_slider)
        rag_layout.addRow("", self.absolute_cos_min_label)
        
        
        # chunk size for embedder
        ####################
        self.chunk_size = QComboBox()
        self.chunk_size.setEditable(False)  # user cannot type
        self.chunk_size.setMaximumHeight(25)
        
        # Add predefined options
        chunk_values = [128, 256, 512, 786, 1024, 1536, 2048]
        for val in chunk_values:
            self.chunk_size.addItem(str(val))
        
        # Set default value
        self.chunk_size.setCurrentText(str(self.CONFIG['rag_params'].get("chunk_size", self.CONFIG['rag_params']['chunk_size'])))
        # Add to layout
        rag_layout.addRow("Chunk_size (embedding):", self.chunk_size)
        ####################
        
        
        # overlap ratio
        self.overlap_ratio_slider = QSlider(Qt.Horizontal)
        self.overlap_ratio_slider.setRange(0, 100)
        self.overlap_ratio_slider.setValue(int(self.CONFIG['rag_params']['overlap_ratio']*100))
        self.overlap_ratio_label = QLineEdit(str(self.CONFIG['rag_params']['overlap_ratio']))
        self.overlap_ratio_label.setReadOnly(True)
        self.overlap_ratio_label.setMaximumHeight(25)
        self.overlap_ratio_slider.valueChanged.connect(lambda v: self.overlap_ratio_label.setText(f"{v/100:.2f}"))
        rag_layout.addRow("Overlap ratio:", self.overlap_ratio_slider)
        rag_layout.addRow("", self.overlap_ratio_label)
        
        
        #rag batch size
        ####################
        self.rag_batch_size = QComboBox()
        self.rag_batch_size.setEditable(False)  # user cannot type
        self.rag_batch_size.setMaximumHeight(25)
        
        # Add predefined options
        batch_values = [2,4,8,12,16,24,32,48,64,96,128,256]
        for val in batch_values:
            self.rag_batch_size.addItem(str(val))
        
        # Set default value
        self.rag_batch_size.setCurrentText(str(self.CONFIG['rag_params'].get("RAG_batch_size", self.CONFIG['rag_params']['RAG_batch_size'])))
        # Add to layout
        rag_layout.addRow("RAG batch size:", self.rag_batch_size)
        ####################
        

        
        
        #############
        
        rag_group.setLayout(rag_layout)
        middle_layout.addWidget(rag_group)
        
        # Stretch pushes everything to the top
        middle_layout.addStretch()
        
        settings_splitter.addWidget(middle_col)
        settings_splitter.setStretchFactor(1, 1)  # equal width for all columns

        
        
        
        # --- right COLUMN ---
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        #Model loading params
        ######################################
        model_load_group = QGroupBox("Model loading params")
        model_load_layout = QFormLayout()
        model_load_layout.setContentsMargins(10, 20, 10, 10)
        model_load_layout.setSpacing(8)
        
        
        # Define full names, internal codes, and tooltips
        model_quant_display = [
            ("FP16 (half precision)", 16, "Full model quality, uses ~2x less memory than FP32."),
            ("8-bit integer", 8, "Saves memory, minimal accuracy drop (~1-2%)."),
            ("4-bit NF4 double quant", 4, "Maximum VRAM saving, tiny accuracy drop (~0-3%). Production-safe.")
        ]
        
        # Add options to combo box
        self.model_quant_options = QComboBox()
        self.model_quant_options.setEditable(False)
        self.model_quant_options.setMaximumHeight(25)
        
        # Populate combo box with tooltip
        for name, code, tooltip in model_quant_display:
            self.model_quant_options.addItem(name, code)          # store numeric code
            index = self.model_quant_options.count() - 1
            self.model_quant_options.setItemData(index, tooltip, Qt.ToolTipRole)
        
        # Set default value
        q = self.CONFIG['model_load_params'].get("quantization")
        quant_name = next((name for name, code, _ in model_quant_display if code == q), None)
        self.model_quant_options.setCurrentText(quant_name)
        
        # Add to layout
        model_load_layout.addRow("Model quant:", self.model_quant_options)
        

        
        model_load_group.setLayout(model_load_layout)
        right_layout.addWidget(model_load_group)
        ###########################################
        # Generation Params Group
        other_group = QGroupBox("Other parameters")
        other_layout = QFormLayout()
        other_layout.setContentsMargins(10, 20, 10, 10)
        other_layout.setSpacing(8)
        
        ############
        # Add True/False switch
        self.TOR_on_start = QCheckBox("Launch TOR on start")
        self.TOR_on_start.setChecked(bool(self.CONFIG['other'].get("launch_tor_on_start", True)))  # default = False

        # Function that runs when toggled
        def TOR_on_start_switch(state):
            # Update config dict (or object) directly
            self.CONFIG["other"]["launch_tor_on_start"] = bool(state)
        
        self.TOR_on_start.toggled.connect(TOR_on_start_switch)
        
        other_layout.addRow("", self.TOR_on_start)
        
        #############
        
        other_group.setLayout(other_layout)
        right_layout.addWidget(other_group)
        
        # Stretch pushes everything to the top
        right_layout.addStretch()
        
        settings_splitter.addWidget(right_col)
        settings_splitter.setStretchFactor(2, 1)  # equal width for all columns
        
        
        
        
        
        # Add settings panel to stack
        self.stack.addWidget(self.settings_panel)  # index 1
        ####################################################
        
        
        
        # --- Bottom buttons layout ---
        buttons_layout = QHBoxLayout()
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setMaximumWidth(150)
        
        def save_and_notify():
            self.save_settings()  # your existing function that writes config
            QMessageBox.information(
                self,
                "Settings Saved",
                "Some options may require restarting the application to take effect."
            )
        
        save_btn.clicked.connect(save_and_notify)
        buttons_layout.addWidget(save_btn)
        
        # Load defaults button
        load_defaults_btn = QPushButton("Load Defaults")
        load_defaults_btn.setMaximumWidth(150)
        
        def confirm_load_defaults():
            reply = QMessageBox.question(
                self,
                "Confirm",
                "Are you sure you want to return to default settings?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.load_default_settings()  # your function to reset values
        
        load_defaults_btn.clicked.connect(confirm_load_defaults)
        buttons_layout.addWidget(load_defaults_btn)
        
        # Add stretch so buttons stay on left side
        buttons_layout.addStretch()
        
        # Add buttons layout to main settings layout (below columns)
        settings_layout.addLayout(buttons_layout)

        
        # Start with main panel
        self.stack.setCurrentIndex(0)
    
        logging.set_verbosity_error()
        self.log("Application started. No model loaded.")
        self.update_rag_status()

     
    def save_settings(self):
        gen_params =   {
                        "temperature" : self.temp_slider.value() / 100,
                    	"top_p": self.top_p_slider.value() / 100,
                        "max_new_tokens": int(self.max_tokens_input.currentText()),
                        "repetition_penalty": self.rep_penalty_slider.value() / 100,
                        "no_repeat_ngram_size": self.no_rep_ngram_slider.value(),
                        "use_cache": True
                        }
        
        rag_params =   {
                        "RAG_max_new_tokens": int(self.rag_max_tokens.currentText()),
                        "RAG_temperature": self.rag_temp_slider.value() / 100,
                        "RAG_top_p": self.rag_top_p_slider.value() / 100,
                        "top_n": int(self.top_n.currentText()),
                        "min_relevance": self.min_relevance_slider.value() / 100,
                        "absolute_cosine_min": self.absolute_cos_min_slider.value() / 100,
                        "chunk_size": int(self.chunk_size.currentText()),
                        "overlap_ratio": self.overlap_ratio_slider.value() / 100,
                        "RAG_batch_size": int(self.rag_batch_size.currentText())
                         } 
        
        
        model_quant_display = [
            ("FP16 (half precision)", 16),
            ("8-bit integer", 8),
            ("4-bit NF4 double quant", 4)
        ]
        quant_name = str(self.model_quant_options.currentText())
        q = next((code for name, code in model_quant_display if name == quant_name), None)
        
        model_load_params = {
                            "quantization": int(q),
                            }        
        
        other =         {
                        "launch_tor_on_start": bool(self.CONFIG['other'].get("launch_tor_on_start", True))
                        }

        self.CONFIG = {
                  "generation_params": gen_params,
                  "rag_params": rag_params,
                  "model_load_params": model_load_params,
                  "other": other
                  }

        #Update Vector_lib params
        self.Vector_lib.chunk_size = self.CONFIG["rag_params"]["chunk_size"]
        self.Vector_lib.overlap_ratio = self.CONFIG["rag_params"]["overlap_ratio"]
        self.Vector_lib.batch_size = self.CONFIG["rag_params"]["RAG_batch_size"]


        with open(CONFIG_PATH, "w") as f:
            json.dump(self.CONFIG, f, indent=4)
        
        
    def load_default_settings(self):
        print("Loading default settings")
        with open(DEF_CONFIG_PATH, "r") as f:
            self.CONFIG = json.load(f)
            
        #Update Vector_lib params
        self.Vector_lib.chunk_size = self.CONFIG["rag_params"]["chunk_size"]
        self.Vector_lib.overlap_ratio = self.CONFIG["rag_params"]["overlap_ratio"]
        self.Vector_lib.batch_size = self.CONFIG["rag_params"]["RAG_batch_size"]
        
        #Overwriting settings with default
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.CONFIG, f, indent=4)
            
        #Reset the display
        #Gen params setting after default
        self.max_tokens_input.setCurrentText(str(self.CONFIG['generation_params'].get("max_new_tokens", self.CONFIG['generation_params']['max_new_tokens'])))
        self.temp_slider.setValue(int(self.CONFIG['generation_params']['temperature']*100))
        self.top_p_slider.setValue(int(self.CONFIG['generation_params']['top_p']*100))
        self.rep_penalty_slider.setValue(int(self.CONFIG['generation_params']['repetition_penalty']*100))
        self.no_rep_ngram_slider.setValue(int(self.CONFIG['generation_params']['no_repeat_ngram_size']))


        #Rag params setting after default
        self.rag_max_tokens.setCurrentText(str(self.CONFIG['rag_params'].get("RAG_max_new_tokens", self.CONFIG['rag_params']['RAG_max_new_tokens'])))
        self.rag_temp_slider.setValue(int(self.CONFIG['rag_params']['RAG_temperature']*100))
        self.rag_top_p_slider.setValue(int(self.CONFIG['rag_params']['RAG_top_p']*100))
        self.top_n.setCurrentText(str(self.CONFIG['rag_params'].get("top_n", self.CONFIG['rag_params']['top_n'])))
        self.min_relevance_slider.setValue(int(self.CONFIG['rag_params']['min_relevance']*100))
        self.absolute_cos_min_slider.setValue(int(self.CONFIG['rag_params']['absolute_cosine_min']*100))
        self.chunk_size.setCurrentText(str(self.CONFIG['rag_params'].get("chunk_size", self.CONFIG['rag_params']['chunk_size'])))
        self.overlap_ratio_slider.setValue(int(self.CONFIG['rag_params']['overlap_ratio']*100))
        self.rag_batch_size.setCurrentText(str(self.CONFIG['rag_params'].get("RAG_batch_size", self.CONFIG['rag_params']['RAG_batch_size'])))

        #Model load params
        # Set default value
        model_quant_display = [
            ("FP16 (half precision)", 16),
            ("8-bit integer", 8),
            ("4-bit NF4 double quant", 4)
        ]
        q = self.CONFIG['model_load_params'].get("quantization")
        quant_name = next((name for name, code in model_quant_display if code == q), None)
        
        self.model_quant_options.setCurrentText(quant_name)

        #Other
        self.TOR_on_start.setChecked(bool(self.CONFIG['other'].get("launch_tor_on_start", True)))
            


    def show_settings(self):
        self.stack.setCurrentIndex(1)
    
    def show_main(self):
        self.stack.setCurrentIndex(0)


    def manage_TOR_server(self):
        #Turn server if button turned on, checked
        if self.TOR_server_btn.isChecked():
            
            self.launch_TOR_server()
        #Turn off if checked off
        else:
            self.close_TOR_server()



    def launch_TOR_server(self):
        print("[TOR] Launching Tor process...")
        
        self.TOR_server = Functions.launch_tor_with_config(
                        config={'SocksPort': '9050'},
                        tor_cmd="C:\\Programy\\Tor\\tor\\tor.exe"
                    )
        print("[TOR] Tor running on 127.0.0.1:9050")
        
        
    def close_TOR_server(self):

        print("[TOR] Shutting down Tor server...")
        self.TOR_server.kill()
        print("[TOR] Tor stopped.")



    def update_rag_dependencies(self):
        # Online depends on Local
        self.RAG_online.setEnabled(self.RAG_local.isChecked())
        if not self.RAG_local.isChecked():
            self.RAG_online.setChecked(False)

        # Online1 depends on Online
        self.RAG_online_TOR_search.setEnabled(self.RAG_online.isChecked())
        if not self.RAG_online.isChecked():
            self.RAG_online_TOR_search.setChecked(False)

        # Online2 depends on Online1
        self.RAG_online_dark_TOR_search.setEnabled(self.RAG_online_TOR_search.isChecked())
        if not self.RAG_online_TOR_search.isChecked():
            self.RAG_online_dark_TOR_search.setChecked(False)




    def on_update_rag_lib(self):
        if not hasattr(self, "Vector_lib") or self.Vector_lib is None:
            self.rag_status_box.setPlainText("RAG system not initialized.")
            return

        self.RAG_add_btn.setEnabled(False)
        self.rag_status_box.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.log_area.clear()

        # Start the ingestion worker thread
        self.ingest_thread = IngestWorker(self.Vector_lib, DOC_LIB_DIR)
        self.ingest_thread.progress.connect(self.on_ingest_progress)
        self.ingest_thread.finished.connect(self.on_ingest_finished)
        self.ingest_thread.error.connect(self.on_ingest_error)
        self.ingest_thread.start()

    def on_ingest_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.log_area.setPlainText(f"Ingesting PDFs: {current}/{total}")

    def on_ingest_finished(self, ingested, skipped):
        self.rag_status_box.setPlainText(f"Ingestion complete. New PDFs added: {ingested}, Skipped: {skipped}")
        self.progress_bar.setVisible(False)
        self.update_rag_status()
        self.RAG_add_btn.setEnabled(True)
        
        #Reload the standard for reading fp16 model
        self.Vector_lib.load_model_fp32()
        #self.Vector_lib.load_model_fp16() # test with 32, should be 16

    def on_ingest_error(self, message):
        # Append error message to rag_status_box without interrupting progress
        current_text = self.rag_status_box.toPlainText()
        self.rag_status_box.setPlainText(current_text + "\n" + message)

    def update_rag_status(self):
        if not hasattr(self, "Vector_lib") or self.Vector_lib is None:
            self.rag_status_box.setPlainText("RAG system not initialized.")
            self.RAG_add_btn.setEnabled(True)
            return

        status = self.Vector_lib.get_ingestion_status()
        total_pdfs = len(status)
        ingested_count = sum(1 for v in status.values() if v)
        missing_count = total_pdfs - ingested_count

        if total_pdfs == 0:
            self.rag_status_box.setPlainText("No PDFs found in library folder.")
        else:
            self.rag_status_box.setPlainText(
                f"Total PDFs: {total_pdfs}\n"
                f"Ingested: {ingested_count}\n"
                f"Missing: {missing_count}"
            )

        self.RAG_add_btn.setEnabled(missing_count > 0)



    def add_message(self, text, from_user=False):
        # Create the message label
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    
        # Set dynamic width based on window ratio
        ratio = 0.5 if from_user else 0.8
        target_width = int(self.chat_scroll.viewport().width() * ratio)
    
        bubble.setMinimumWidth(target_width)
        bubble.setMaximumWidth(target_width)
    
        # Style based on sender
        if from_user:
            bubble.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    padding: 10px 14px;
                    background-color: #3a7bd5;
                    color: white;
                    font-size: 14px;
                }
            """)
        else:
            bubble.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    padding: 10px 14px;
                    background-color: #3e3e3e;
                    color: white;
                    font-size: 14px;
                }
            """)
    
        # Layout wrapper
        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(10, 5, 10, 5)
    
        container = QWidget()
        container.setLayout(wrapper)
    
        # Align user right, bot left
        if from_user:
            wrapper.addStretch()
            wrapper.addWidget(bubble)
        else:
            wrapper.addWidget(bubble)
            wrapper.addStretch()
    
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, container)
    
        # Ensure scrollbar follows
        QApplication.processEvents()
        self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        )



    def _apply_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #282c34; color: #abb2bf; }
            QGroupBox { background-color: #21252b; border: 1px solid #3e4451; margin-top: 10px; }
            QGroupBox::title { color: #61afef; }
            QTextEdit { background-color: #1e222a; color: #abb2bf; }
            QLineEdit { background-color: #3c3f4c; color: #ffffff; }
            QPushButton, QComboBox { background-color: #61afef; color: #282c34; border-radius: 4px; padding: 4px; }
            QToolBar { background-color: #21252b; }
    
            /* Nice checkbox style */
            QCheckBox {
                spacing: 10px;                    /* space between box and label */
                font-size: 10pt;
                color: #abb2bf;
            }
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
                border: 3px solid #3e4451;
                border-radius: 3px;
                background-color: #1e222a;
            }
            QCheckBox::indicator:checked {
                background-color: #61afef;
                border: 3px solid #3e4451;
            }
        """)

        #self.setFont(QFont("Segoe UI", 10))

    def log(self, msg):
        self.log_area.append(msg)

    def load_model(self):
        name = self.model_dropdown.currentText()
        path = os.path.join(MODEL_DIR, name)
        self.status_label.setPlainText("Loading…")
        QApplication.processEvents()
        try:
            #del self.model, self.tokenizer
            #torch.cuda.empty_cache()
            #gc.collect()
            if self.CONFIG['model_load_params']['quantization'] == 16:
                self.log(f"Loading model [ FP16 ]: {name}…")
                self.llm.load_fp16(path)
                
            elif self.CONFIG['model_load_params']['quantization'] == 8:
                self.log(f"Loading model [ INT8 ]: {name}…")
                self.llm.load_int8(path)
                    
            elif self.CONFIG['model_load_params']['quantization'] == 4:
                self.log(f"Loading model [ INT4 ]: {name}…")
                self.llm.load_int4(path)
                
            else:
                self.model = None
                
            self.tokenizer, self.model, self.chat_history = self.llm.get()
            self.status_label.setPlainText("Model ready")
            self.log(f"Model {name} loaded.")
        except Exception as e:
            self.status_label.setPlainText("Error loading model")
            self.log(f"Error: {e}")

    def clear_chat(self):
        # Clear layout widgets except for the final stretch item
        for i in reversed(range(self.chat_layout.count() - 1)):
            item = self.chat_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        if self.chat_history:
            self.chat_history = [self.chat_history[0]]
        self.log("Chat cleared.")

    def on_send(self):
        user_text = self.input_line.text().strip()
        if not user_text or self.model is None:
            return
        self.log(f"User: {user_text if len(user_text) <= 30 else user_text[:30] + '...'}")
        self.add_message(user_text, from_user=True)
        self.chat_history.append({"role": "user", "content": user_text})
        self.input_line.clear()
        self._generate_response()


    def generate_online_rag_queries(self, k=5):
        # Get the latest user query
        user_messages = [m['content'] for m in self.chat_history if m['role'] == 'user']
        user_query = user_messages[-1]
    
        # Instructional prompt for RAG generation
        rag_request = [
            {
                "role": "system",
                "content": (
                    f"You are a generator of exactly {k} search queries from the user's request. Follow these simple rules.\n\n"
                    f"1) OUTPUT: produce exactly {k} lines, one concise search query per line. No numbering, bullets, commentary, or extra text.\n\n"
                    f"2) STYLE: queries are short keyword-style phrases (not full sentences), suitable for web search.\n\n"
                    f"3) DIVERSITY: cover different angles (how-to, comparison, overview, feasibility, tutorials, buying/upgrade, safety/ethics, history, etc.). Make each line distinct.\n\n"
                    f"4) NO ANSWERING: do not answer the user's question yourself — only produce queries.\n\n"
                    f"5) GREETINGS: if the input is pure small talk or a greeting (very short text of up to 3 words consisting only of common greetings like \"hi\", \"hello\", \"thanks\", \"how are you\"), output {k} lines of the exact token: __NO_SEARCH__ and nothing else.\n\n"
                    f"6) SPECULATIVE / SILLY INPUTS: if the user asks about implausible or highly speculative things, reframe them into practical, realistic, web-searchable queries (feasibility/current-state, realistic alternatives, or actionable next steps).\n\n"
                    f"7) VAGUE INPUTS: if the user is vague, include queries that clarify intent: overview, how-to, comparison, and at least one actionable next-step.\n\n"
                    f"8) FORMAT: output should be formatted in a way, so each query has its own line and is starting by '1. ', '2. ' etc. There should be no headline, just raw {k} queries.\n\n"
                    f"Format rule reminder: EXACTLY {k} LINES, one query per line."
                )
            },
            {"role": "user", "content": user_query},
        ]
    
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            rag_request,
            tokenize=False,
            add_generation_prompt=False
        )
    
        # Tokenize and move to GPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        eos = self.tokenizer.eos_token_id
    
        # Generation parameters
        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.CONFIG['rag_params']['RAG_max_new_tokens'],
            "temperature": self.CONFIG['rag_params']['RAG_temperature'],
            "top_p": self.CONFIG['rag_params']['RAG_top_p'],
            "do_sample": True,
            "eos_token_id": eos,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
    
        # Run generation synchronously
        output_ids = self.model.generate(**gen_kwargs)
    
        # Decode output skipping prompt tokens
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
    
        # Post-process: split into queries
        queries = [q.strip() for q in generated_text.split("\n") if q.strip()]
    
        # Ensure exactly k results
        return queries[:k]


    def _generate_response(self):
        
        if self.RAG_online_dark_TOR_search.isChecked():
            print("Using Dark TOR RAG")
        
        elif self.RAG_online_TOR_search.isChecked():
            print("Using TOR RAG")
            online_queries = self.generate_online_rag_queries(k = 6)
            prompt, RAG_present = Functions.online_rag_chat_prompt(tokenizer = self.tokenizer,
                                                                   queries = online_queries,
                                                                   messages = self.chat_history,
                                                                   vector_lib = self.Vector_lib,
                                                                   top_n = self.CONFIG['rag_params']['top_n'],
                                                                   min_relevance = self.CONFIG['rag_params']['min_relevance'],
                                                                   absolute_cosine_min = self.CONFIG['rag_params']['absolute_cosine_min'],
                                                                   add_generation_prompt=True,
                                                                   TOR_search = True,
                                                                   TOR_server_on = self.TOR_server_btn.isChecked()
                                                                   )
        
        elif self.RAG_online.isChecked():
            print("Using online RAG")
            online_queries = self.generate_online_rag_queries(k = 6)
            
            prompt, RAG_present = Functions.online_rag_chat_prompt(tokenizer = self.tokenizer,
                                                                   queries = online_queries,
                                                                   messages = self.chat_history,
                                                                   vector_lib = self.Vector_lib,
                                                                   top_n = self.CONFIG['rag_params']['top_n'],
                                                                   min_relevance = self.CONFIG['rag_params']['min_relevance'],
                                                                   absolute_cosine_min = self.CONFIG['rag_params']['absolute_cosine_min'],
                                                                   add_generation_prompt=True,
                                                                   TOR_search = False,
                                                                   TOR_server_on = self.TOR_server_btn.isChecked()
                                                                   )
            
        
            
            
        elif self.RAG_local.isChecked():
            print("Using local RAG")
            prompt, RAG_present = Functions.local_rag_chat_prompt(self.tokenizer, self.chat_history, self.Vector_lib, self.CONFIG['rag_params']['top_n'], self.CONFIG['rag_params']['min_relevance'], self.CONFIG['rag_params']['absolute_cosine_min'])


                
        else:
            print("No RAG response")
            RAG_present = (False, "None")
            #Trying to use built in chat template builder, if not use the emergency one from functions
            try:
                prompt = self.tokenizer.apply_chat_template(
                                                self.chat_history,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
            except:
                print("Could not use native prompt builder - Initializing emergency one from built in functions...")
                prompt, RAG_present = Functions.emergency_chat_prompt(self.chat_history)
                print("Processing prompt done!")
        
            
        
        
        
        ##############################################
        self.log(f"RAG context present: {RAG_present[0]}, max similarity: {RAG_present[1]}")
        
        print(f"Total tokens: {len(self.tokenizer(prompt)['input_ids'])}")
        print("\n\nPrompt: ", prompt,"\n\n") # For debug ant testing, visible just in console
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        eos = self.tokenizer.eos_token_id
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **self.CONFIG['generation_params'],   # defaults first
            **inputs,
            "max_new_tokens": self.CONFIG['generation_params']['max_new_tokens'],
            "temperature": self.CONFIG['generation_params']['temperature'],
            "top_p": self.CONFIG['generation_params']['top_p'],
            "no_repeat_ngram_size": self.CONFIG['generation_params']['no_repeat_ngram_size'],
            "repetition_penalty": self.CONFIG['generation_params']['repetition_penalty'],
            "streamer": streamer,
            "do_sample": True,
            "eos_token_id": eos,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True).start()
        self.add_message("...", from_user=False)

        def stream_and_store():
            full_reply = ""
            start = time.time()
            for token in streamer:
                full_reply += token
                # reference to last QLabel
                msg_widget = self.chat_layout.itemAt(self.chat_layout.count() - 2).widget()
                msg_label = msg_widget.layout().itemAt(0 if msg_widget.layout().count() == 2 else 1).widget()
                msg_label.setText(full_reply + token)
                
            elapsed = time.time() - start
            rate = len(full_reply.split()) / elapsed if elapsed > 0 else 0
            self.log(f"{BOT_NAME}: {full_reply[:30] + '...' if len(full_reply) > 30 else full_reply}")
            self.log(f"Response done in {elapsed:.2f}s ({rate:.1f} tok/s)")
            self.chat_history.append({"role": "assistant", "content": full_reply})

        threading.Thread(target=stream_and_store, daemon=True).start()

    def closeEvent(self, event):
        if QMessageBox.question(self, "Confirm Exit", "Close session and exit?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
