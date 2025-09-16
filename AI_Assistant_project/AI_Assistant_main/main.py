# AI_Assistant_main/main.py

#App frontend lib
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QComboBox, QToolBar,
    QGroupBox, QFormLayout, QSlider, QSplitter, QTextEdit, QMessageBox
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
from config import GENERATION_PARAMS, RAG_PARAMS, BOT_NAME, BOT_DEV_NAME
from config import MODEL_DIR, PDF_LIB_DIR, VECTOR_LIB_DIR




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
        
        


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(BOT_DEV_NAME)
        self.resize(1200, 700)
        self._apply_theme()

        self.llm = LocalLLM()
        self.model = None
        self.tokenizer = None
        self.chat_history = []

        self.Vector_lib = Modules.UniversalVectorStore(PDF_LIB_DIR, VECTOR_LIB_DIR, RAG_PARAMS['chunk_size'], RAG_PARAMS['overlap_ratio'] )
        self.Vector_lib.load_model_fp32()
        #self.Vector_lib.load_model_fp16() # test with 32, should be 16
        
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Clear Chat", self, triggered=self.clear_chat))
        toolbar.addAction(QAction("Quit", self, triggered=self.close))

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # LEFT PANEL
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

        self.status_label = QTextEdit("No model loaded")
        self.status_label.setReadOnly(True)
        self.status_label.setMaximumHeight(60)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout.addRow("Status:", self.status_label)
        model_group.setLayout(info_layout)
        left_layout.addWidget(model_group)

        # Generation Params Group
        params_group = QGroupBox("Generation Params")
        params_layout = QFormLayout()
        params_layout.setContentsMargins(10, 20, 10, 10)

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(10)
        self.temp_label = QTextEdit("0.10")
        self.temp_label.setReadOnly(True)
        self.temp_label.setMaximumHeight(30)
        self.temp_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setPlainText(f"{v/100:.2f}"))
        params_layout.addRow("Temperature:", self.temp_slider)
        params_layout.addRow("", self.temp_label)

        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(0, 100)
        self.top_p_slider.setValue(85)
        self.top_p_label = QTextEdit("0.85")
        self.top_p_label.setReadOnly(True)
        self.top_p_label.setMaximumHeight(30)
        self.top_p_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_label.setPlainText(f"{v/100:.2f}"))
        params_layout.addRow("Top P:", self.top_p_slider)
        params_layout.addRow("", self.top_p_label)
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

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
        
        #Local Rag button option
        self.RAG_local = QCheckBox("Local RAG")
        self.RAG_local.setChecked(True)  
        #self.RAG_local.stateChanged.connect(self.Prepared function)
        rag_status_layout.addWidget(self.RAG_local)
        
        #Online Rag button option
        self.RAG_online = QCheckBox("Online RAG")
        self.RAG_online.setChecked(True)  
        #self.RAG_local.stateChanged.connect(self.Prepared function)
        rag_status_layout.addWidget(self.RAG_online)
        
        #TOR search Rag button option
        self.RAG_online_TOR_search = QCheckBox("Online RAG - TOR search")
        self.RAG_online_TOR_search.setChecked(False)  
        #self.RAG_local.stateChanged.connect(self.Prepared function)
        rag_status_layout.addWidget(self.RAG_online_TOR_search)
        
        #Dark_TOR search Rag button option
        self.RAG_online_dark_TOR_search = QCheckBox("Online RAG - Dark TOR search")
        self.RAG_online_dark_TOR_search.setChecked(False)  
        #self.RAG_local.stateChanged.connect(self.Prepared function)
        rag_status_layout.addWidget(self.RAG_online_dark_TOR_search)
        
        #Allow online only when local, TOR only if online etc.
        self.RAG_local.stateChanged.connect(self.update_rag_dependencies)
        self.RAG_online.stateChanged.connect(self.update_rag_dependencies)
        self.RAG_online_TOR_search.stateChanged.connect(self.update_rag_dependencies)
        

        rag_status_group.setLayout(rag_status_layout)
        left_layout.addWidget(rag_status_group)
        left_layout.addStretch()

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

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
        splitter.setStretchFactor(1, 3)  # right bigger

        logging.set_verbosity_error()
        self.log("Application started. No model loaded.")
        self.update_rag_status()


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
        self.ingest_thread = IngestWorker(self.Vector_lib, PDF_LIB_DIR)
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
        self.log(f"Loading model: {name}…")
        self.status_label.setPlainText("Loading…")
        QApplication.processEvents()
        try:
            #del self.model, self.tokenizer
            #torch.cuda.empty_cache()
            #gc.collect()
            
            self.llm.load(path)
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
            "max_new_tokens": 256,  # enough for k queries
            "temperature": 0.7,
            "top_p": 0.95,
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
        
        elif self.RAG_online.isChecked():
            print("Using online RAG")
            online_queries = self.generate_online_rag_queries(k = 5)
            print("\n")
            for q in online_queries:
                print(q)
            
            
        elif self.RAG_local.isChecked():
            print("Using local RAG")
            prompt, RAG_present = Functions.local_rag_chat_prompt(self.tokenizer, self.chat_history, self.Vector_lib, RAG_PARAMS['top_n'], RAG_PARAMS['min_relevance'], RAG_PARAMS['absolute_cosine_min'])


                
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
            **inputs,
            "streamer": streamer,
            "temperature": self.temp_slider.value() / 100,
            "top_p": self.top_p_slider.value() / 100,
            "do_sample": True,
            "eos_token_id": eos,
            "pad_token_id": self.tokenizer.pad_token_id,
            **GENERATION_PARAMS
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
