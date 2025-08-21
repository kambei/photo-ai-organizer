import torch
import timm
import os
import shutil
from PIL import Image, ImageTk
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

# --- CONFIGURAZIONE ---
# Modifica questi percorsi in base alle tue cartelle
SOURCE_FOLDER = './foto_da_analizzare'  # La cartella con tutte le immagini
EXAMPLE_IMAGE = './foto_esempio.jpg'  # L'immagine con il soggetto da cercare
DESTINATION_FOLDER = './foto_trovate'  # Dove verranno copiate le immagini simili

# Puoi sperimentare con la soglia. Più alto è il valore, più le immagini devono essere simili.
# Un buon punto di partenza è tra 0.85 e 0.95
SIMILARITY_THRESHOLD = 0.9

# Nome del modello pre-allenato da usare. 'resnet50' è un buon equilibrio tra velocità e precisione.
MODEL_NAME = 'resnet50'


# --------------------

def setup_environment():
    """Crea le cartelle di esempio se non esistono."""
    print("Configurazione dell'ambiente in corso...")
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER)
        print(f"Cartella '{SOURCE_FOLDER}' creata. Per favore, aggiungi le tue immagini qui.")

    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
        print(f"Cartella '{DESTINATION_FOLDER}' creata. I risultati verranno salvati qui.")

    if not os.path.exists(EXAMPLE_IMAGE):
        # Crea un'immagine di placeholder se non esiste
        try:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(EXAMPLE_IMAGE, 'jpeg')
            print(f"File '{EXAMPLE_IMAGE}' non trovato. Creato un file di esempio. Sostituiscilo con la tua immagine.")
        except Exception as e:
            print(f"Errore nella creazione dell'immagine di esempio: {e}")


def get_model():
    """Carica il modello di visione artificiale pre-allenato."""
    print(f"Caricamento del modello '{MODEL_NAME}' in corso... Questo potrebbe richiedere del tempo la prima volta.")
    # Utilizziamo un modello pre-allenato su ImageNet
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    # Imposta il modello in modalità di valutazione (non di addestramento)
    model.eval()
    print("Modello caricato con successo.")
    return model


def get_image_transforms():
    """Definisce le trasformazioni da applicare alle immagini prima di darle al modello."""
    # Queste trasformazioni sono standard per i modelli pre-allenati su ImageNet
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(image_path, model, transform):
    """
    Estrae un vettore di caratteristiche (features) da una singola immagine.
    Questo vettore è una rappresentazione numerica del contenuto dell'immagine.
    """
    try:
        # Apre l'immagine e si assicura che sia in formato RGB
        image = Image.open(image_path).convert('RGB')
        # Applica le trasformazioni
        image_tensor = transform(image).unsqueeze(0)  # Aggiunge una dimensione per il "batch"

        # Disabilita il calcolo del gradiente per risparmiare memoria e velocizzare
        with torch.no_grad():
            # Passa l'immagine attraverso il modello per ottenere le features
            features = model(image_tensor)

        # Restituisce le features come un array numpy
        return features.numpy()
    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine {image_path}: {e}")
        return None


def find_and_copy_similar_images(model, transform):
    """
    Versione CLI: ricerca e copia utilizzando le variabili globali.
    """
    if not os.path.exists(EXAMPLE_IMAGE):
        print(f"ERRORE: L'immagine di esempio '{EXAMPLE_IMAGE}' non è stata trovata. Impossibile procedere.")
        return

    print("\n--- Inizio dell'analisi ---")
    print(f"Analisi dell'immagine di esempio: {EXAMPLE_IMAGE}")
    example_features = extract_features(EXAMPLE_IMAGE, model, transform)
    if example_features is None:
        print("Impossibile analizzare l'immagine di esempio. Interruzione del processo.")
        return

    found_count = 0
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    print(f"Scansione della cartella di origine: {SOURCE_FOLDER}")
    image_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(supported_formats)]
    total_images = len(image_files)
    print(f"Trovate {total_images} immagini da analizzare.")

    for i, filename in enumerate(image_files):
        current_image_path = os.path.join(SOURCE_FOLDER, filename)
        print(f"[{i + 1}/{total_images}] Analisi di: {filename}...", end=' ')
        current_features = extract_features(current_image_path, model, transform)
        if current_features is None:
            print(" -> Saltata (errore).")
            continue
        similarity = cosine_similarity(example_features, current_features)[0][0]
        if similarity >= SIMILARITY_THRESHOLD:
            print(f" -> Trovata! (Somiglianza: {similarity:.2f})")
            destination_path = os.path.join(DESTINATION_FOLDER, filename)
            shutil.copy2(current_image_path, destination_path)
            found_count += 1
        else:
            print(f" -> Non simile (Somiglianza: {similarity:.2f})")

    print("\n--- Analisi completata ---")
    print(f"Copia di {found_count} immagini simili nella cartella '{DESTINATION_FOLDER}'.")


def analyze_images(model, transform, source_folder, example_image, destination_folder, threshold=0.9, on_update=None, on_complete=None):
    """
    Esegue l'analisi emettendo aggiornamenti per la GUI tramite callback.
    on_update(evento_dict) viene chiamato per ogni immagine con chiavi:
      - index, total, filename, image_path, similarity, is_match, copied_to (o None)
    on_complete(riepilogo_dict) con chiavi:
      - found_count, total_images, destination_folder
    """
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
        if not os.path.exists(example_image):
            msg = f"Immagine di esempio non trovata: {example_image}"
            if on_complete:
                on_complete({"error": msg, "found_count": 0, "total_images": 0, "destination_folder": destination_folder})
            return

        example_features = extract_features(example_image, model, transform)
        if example_features is None:
            msg = "Impossibile analizzare l'immagine di esempio."
            if on_complete:
                on_complete({"error": msg, "found_count": 0, "total_images": 0, "destination_folder": destination_folder})
            return

        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(supported_formats)]
        total_images = len(image_files)
        found_count = 0

        for idx, filename in enumerate(image_files, start=1):
            current_image_path = os.path.join(source_folder, filename)
            current_features = extract_features(current_image_path, model, transform)
            if current_features is None:
                if on_update:
                    on_update({
                        "index": idx, "total": total_images, "filename": filename, "image_path": current_image_path,
                        "similarity": None, "is_match": False, "copied_to": None, "error": "Errore features"
                    })
                continue
            similarity = float(cosine_similarity(example_features, current_features)[0][0])
            is_match = similarity >= float(threshold)
            copied_to = None
            if is_match:
                copied_to = os.path.join(destination_folder, filename)
                try:
                    shutil.copy2(current_image_path, copied_to)
                except Exception:
                    copied_to = None
                found_count += 1
            if on_update:
                on_update({
                    "index": idx, "total": total_images, "filename": filename, "image_path": current_image_path,
                    "similarity": similarity, "is_match": is_match, "copied_to": copied_to
                })

        if on_complete:
            on_complete({"found_count": found_count, "total_images": total_images, "destination_folder": destination_folder})
    except Exception as e:
        if on_complete:
            on_complete({"error": str(e), "found_count": 0, "total_images": 0, "destination_folder": destination_folder})


class PhotoAIOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo AI Organizer")
        self.model = None
        self.transform = None
        self.analysis_thread = None
        self.photo_img = None  # riferimento per Tkinter

        # Variabili Tkinter
        self.var_source = tk.StringVar(value=SOURCE_FOLDER)
        self.var_example = tk.StringVar(value=EXAMPLE_IMAGE)
        self.var_dest = tk.StringVar(value=DESTINATION_FOLDER)
        self.var_threshold = tk.DoubleVar(value=SIMILARITY_THRESHOLD)

        self._build_ui()

    def _build_ui(self):
        frm = tk.Frame(self.root, padx=10, pady=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Selettori path
        tk.Label(frm, text="Cartella Sorgente:").grid(row=0, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.var_source, width=50).grid(row=0, column=1, padx=5, pady=2, sticky="we")
        tk.Button(frm, text="Scegli...", command=self.choose_source).grid(row=0, column=2, padx=5)

        tk.Label(frm, text="Immagine Esempio:").grid(row=1, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.var_example, width=50).grid(row=1, column=1, padx=5, pady=2, sticky="we")
        tk.Button(frm, text="Scegli...", command=self.choose_example).grid(row=1, column=2, padx=5)

        tk.Label(frm, text="Cartella Destinazione:").grid(row=2, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.var_dest, width=50).grid(row=2, column=1, padx=5, pady=2, sticky="we")
        tk.Button(frm, text="Scegli...", command=self.choose_dest).grid(row=2, column=2, padx=5)

        tk.Label(frm, text="Soglia Similarità (0-1):").grid(row=3, column=0, sticky="e")
        tk.Entry(frm, textvariable=self.var_threshold, width=10).grid(row=3, column=1, sticky="w")

        self.btn_start = tk.Button(frm, text="Avvia Analisi", command=self.start_analysis)
        self.btn_start.grid(row=4, column=0, columnspan=3, pady=8, sticky="we")

        # Area immagine e stato
        self.image_panel = tk.Label(frm, text="Anteprima immagine analizzata", relief=tk.SUNKEN, width=60, height=20)
        self.image_panel.grid(row=5, column=0, columnspan=3, pady=8, sticky="we")

        self.status_label = tk.Label(frm, text="Pronto", anchor="w")
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="we")

        # Etichetta di "match" verde
        self.match_label = tk.Label(frm, text="", bg=self.root.cget("bg"), fg="white", font=("Arial", 12, "bold"))
        self.match_label.grid(row=7, column=0, columnspan=3, pady=4, sticky="we")

        frm.grid_columnconfigure(1, weight=1)

    def choose_source(self):
        path = filedialog.askdirectory(title="Seleziona cartella sorgente")
        if path:
            self.var_source.set(path)

    def choose_example(self):
        path = filedialog.askopenfilename(title="Seleziona immagine esempio", filetypes=[("Immagini", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff")])
        if path:
            self.var_example.set(path)

    def choose_dest(self):
        path = filedialog.askdirectory(title="Seleziona cartella destinazione")
        if path:
            self.var_dest.set(path)

    def start_analysis(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        source = self.var_source.get().strip()
        example = self.var_example.get().strip()
        dest = self.var_dest.get().strip()
        try:
            threshold = float(self.var_threshold.get())
        except Exception:
            threshold = SIMILARITY_THRESHOLD
            self.var_threshold.set(threshold)

        if not source or not os.path.isdir(source):
            messagebox.showerror("Errore", "Seleziona una cartella sorgente valida.")
            return
        if not example or not os.path.isfile(example):
            messagebox.showerror("Errore", "Seleziona un'immagine di esempio valida.")
            return
        if not dest:
            messagebox.showerror("Errore", "Seleziona una cartella di destinazione valida.")
            return

        self.btn_start.config(state=tk.DISABLED, text="Analisi in corso...")
        self.status_label.config(text="Caricamento modello...")
        self.match_label.config(text="", bg=self.root.cget("bg"))

        def worker():
            try:
                if self.model is None or self.transform is None:
                    self.model = get_model()
                    self.transform = get_image_transforms()
                self._set_status("Inizio analisi...")
                analyze_images(self.model, self.transform, source, example, dest, threshold,
                               on_update=self._on_update_async, on_complete=self._on_complete_async)
            except Exception as e:
                self._set_status(f"Errore: {e}")
                self._enable_start()

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    def _on_update_async(self, info):
        self.root.after(0, lambda: self._apply_update(info))

    def _on_complete_async(self, summary):
        self.root.after(0, lambda: self._apply_complete(summary))

    def _apply_update(self, info):
        idx = info.get("index", 0)
        total = info.get("total", 0)
        filename = info.get("filename", "")
        similarity = info.get("similarity")
        is_match = info.get("is_match", False)
        img_path = info.get("image_path")

        # Aggiorna stato
        sim_txt = "?" if similarity is None else f"{similarity:.3f}"
        self.status_label.config(text=f"[{idx}/{total}] {filename} - Similarità: {sim_txt}")

        # Mostra immagine ridimensionata
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((600, 600))
            self.photo_img = ImageTk.PhotoImage(img)
            self.image_panel.config(image=self.photo_img, text="")
        except Exception:
            self.image_panel.config(text="Impossibile caricare l'immagine", image="")

        # Indicatore verde in caso di match
        if is_match:
            self.match_label.config(text="MATCH! Immagine simile trovata", bg="#2ecc71", fg="white")
        else:
            # Svuota l'indicatore quando non c'è match
            self.match_label.config(text="", bg=self.root.cget("bg"))

    def _apply_complete(self, summary):
        if summary.get("error"):
            messagebox.showerror("Errore", summary["error"])
            self._set_status("Errore durante l'analisi")
        else:
            found = summary.get("found_count", 0)
            total = summary.get("total_images", 0)
            dest = summary.get("destination_folder", "")
            self._set_status(f"Analisi completata: {found}/{total} immagini copiate in '{dest}'")
        self._enable_start()

    def _set_status(self, text):
        self.status_label.config(text=text)

    def _enable_start(self):
        self.btn_start.config(state=tk.NORMAL, text="Avvia Analisi")


if __name__ == '__main__':
    # Avvia GUI come richiesto
    try:
        # Prepara le cartelle di default se l'utente sceglie di usarle
        setup_environment()
    except Exception:
        pass

    root = tk.Tk()
    app = PhotoAIOrganizerGUI(root)
    root.mainloop()
