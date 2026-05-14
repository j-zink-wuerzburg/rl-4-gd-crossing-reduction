import os
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import multiprocessing as mp

# Configuration
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'graphs', 'live-contest'))
DEFAULT_EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'exports'))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Training', 'runs', 'ppo_test_cluster_local', 'int_grid_model.zip'))
DEFAULT_OPT_TYPE = 'Local'
MAX_STEPS = 2000
# Default parallelism: leave one core free, cap to reasonable upper bound
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 8))


# Messages sent by workers: (kind, file_path, payload)
# kind in {"progress", "finished", "finished_planar", "error"}


def _worker_entry(file_path, export_dir, model_path, opt_type, max_steps, out_queue):
    """Spawn-safe worker entry that imports heavy libs inside the child process."""
    try:
        try:
            from .worker import run_graph  # when imported as package
        except Exception:
            from src.ui.worker import run_graph  # fallback if run from repo root
        run_graph(file_path, export_dir, model_path, opt_type, max_steps, out_queue)
    except Exception as e:
        try:
            out_queue.put(("error", file_path, str(e)))
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self, input_dir=DEFAULT_INPUT_DIR):
        super().__init__()
        self.title("Graph Runner")
        self.geometry("1000x600")
        self.minsize(900, 500)
        self.input_dir = input_dir

        # Multiprocessing context and message queue
        self.mp_ctx = mp.get_context("spawn")
        self.mp_queue: mp.Queue = self.mp_ctx.Queue()

        # State
        self.running_items = {}  # file_path -> (listbox index, current label)
        # Store pending as tuples (file_path, max_steps)
        self.pending_files = deque()  # items waiting to start
        self.running_procs = {}  # file_path -> Process
        # Track per-file target max steps for progress display
        self.file_steps_target = {}  # file_path -> max_steps

        # UI bits
        self.max_workers_var = tk.IntVar(value=DEFAULT_MAX_WORKERS)
        self.max_steps_var = tk.IntVar(value=MAX_STEPS)

        self._build_ui()
        self._load_left_list()
        self._pump_updates()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Layout: three columns (left, center, right)
        self.grid_columnconfigure(0, weight=1, uniform="columns")
        self.grid_columnconfigure(1, weight=1, uniform="columns")
        self.grid_columnconfigure(2, weight=1, uniform="columns")
        self.grid_rowconfigure(2, weight=1)

        # Header with input folder and concurrency control
        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, text=f"Input folder: {self.input_dir}", anchor="w").grid(row=0, column=0, sticky="w")
        controls = ttk.Frame(header)
        controls.grid(row=0, column=1, sticky="e")
        ttk.Label(controls, text="Max parallel:").grid(row=0, column=0, padx=(0, 4))
        spin = ttk.Spinbox(controls, from_=1, to=max(1, os.cpu_count() or 1), width=5, textvariable=self.max_workers_var)
        spin.grid(row=0, column=1)
        # Add max steps control for per-run customization
        ttk.Label(controls, text="Max steps:").grid(row=0, column=2, padx=(12, 4))
        steps_spin = ttk.Spinbox(controls, from_=1, to=1_000_000_000, increment=100, width=9, textvariable=self.max_steps_var)
        steps_spin.grid(row=0, column=3)

        ttk.Label(self, text="Available .json graphs").grid(row=1, column=0, sticky="nw", padx=8)
        ttk.Label(self, text="Running").grid(row=1, column=1, sticky="nw", padx=8)
        ttk.Label(self, text="Exported").grid(row=1, column=2, sticky="nw", padx=8)

        # Lists and scrollbars
        self.left_list = tk.Listbox(self, selectmode=tk.EXTENDED)
        self.center_list = tk.Listbox(self)
        self.right_list = tk.Listbox(self)

        self.left_scroll = ttk.Scrollbar(self, orient="vertical", command=self.left_list.yview)
        self.left_list.configure(yscrollcommand=self.left_scroll.set)

        self.center_scroll = ttk.Scrollbar(self, orient="vertical", command=self.center_list.yview)
        self.center_list.configure(yscrollcommand=self.center_scroll.set)

        self.right_scroll = ttk.Scrollbar(self, orient="vertical", command=self.right_list.yview)
        self.right_list.configure(yscrollcommand=self.right_scroll.set)

        # Placement
        self.left_list.grid(row=2, column=0, sticky="nsew", padx=(8, 0), pady=8)
        self.left_scroll.grid(row=2, column=0, sticky="nse", padx=(0, 8), pady=8)
        self.center_list.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=8)
        self.center_scroll.grid(row=2, column=1, sticky="nse", padx=(0, 8), pady=8)
        self.right_list.grid(row=2, column=2, sticky="nsew", padx=(8, 8), pady=8)
        self.right_scroll.grid(row=2, column=2, sticky="nse", padx=(0, 8), pady=8)

        # Bind double-click on left list
        self.left_list.bind('<Double-Button-1>', self._on_left_double_click)

    def _load_left_list(self):
        self.left_list.delete(0, tk.END)
        try:
            files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.json')]
            files.sort()
            for name in files:
                self.left_list.insert(tk.END, name)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list input folder: {e}")

    def _on_left_double_click(self, event):
        # Queue selected items; scheduler will start up to max parallel workers
        selection = self.left_list.curselection()
        if not selection:
            return
        # Read and sanitize max steps at queue time to allow per-run customization
        try:
            sel_steps = int(self.max_steps_var.get())
            if sel_steps < 1:
                sel_steps = 1
        except tk.TclError:
            sel_steps = MAX_STEPS
        for idx in selection:
            name = self.left_list.get(idx)
            file_path = os.path.join(self.input_dir, name)
            if file_path in self.running_items or file_path in self.pending_files or file_path in self.running_procs:
                continue  # already queued or running
            # Add to center list with 0/target progress
            label = f"{name} - steps: 0/{sel_steps}"
            center_idx = self.center_list.size()
            self.center_list.insert(tk.END, label)
            self.running_items[file_path] = (center_idx, label)
            self.file_steps_target[file_path] = sel_steps
            # Enqueue with specific steps
            self.pending_files.append((file_path, sel_steps))
        # Try to start as many as allowed
        self._maybe_start_more()

    def _maybe_start_more(self):
        # Start more workers if there are slots and pending files
        try:
            max_workers = max(1, int(self.max_workers_var.get()))
        except tk.TclError:
            max_workers = DEFAULT_MAX_WORKERS
        while len(self.running_procs) < max_workers and self.pending_files:
            item = self.pending_files.popleft()
            if isinstance(item, tuple):
                file_path, steps_for_run = item
            else:
                # backward safeguard
                file_path = item
                steps_for_run = self.file_steps_target.get(file_path, MAX_STEPS)
            try:
                p = self.mp_ctx.Process(
                    target=_worker_entry,
                    args=(file_path, DEFAULT_EXPORT_DIR, DEFAULT_MODEL_PATH, DEFAULT_OPT_TYPE, steps_for_run, self.mp_queue),
                    daemon=True,
                )
                p.start()
                self.running_procs[file_path] = p
            except Exception as e:
                self._mark_error(file_path, str(e))

    def _pump_updates(self, *args):
        # Drain worker messages
        try:
            while True:
                msg = self.mp_queue.get_nowait()
                kind, file_path, payload = msg
                if kind == "progress":
                    steps = payload
                    self._update_progress(file_path, steps)
                elif kind == "finished":
                    out_path = payload
                    self._mark_finished(file_path, out_path)
                    self._cleanup_proc(file_path)
                elif kind == "finished_planar":
                    out_path = payload
                    self._mark_finished_planar(file_path, out_path)
                    self._cleanup_proc(file_path)
                elif kind == "error":
                    err = payload
                    self._mark_error(file_path, err)
                    self._cleanup_proc(file_path)
        except queue.Empty:
            pass
        except Exception:
            pass

        # Reap any processes that exited without sending a final message
        to_remove = []
        for fp, proc in list(self.running_procs.items()):
            if not proc.is_alive() and proc.exitcode is not None:
                if fp in self.running_items:  # still shown as running
                    if proc.exitcode == 0:
                        name = os.path.basename(fp)
                        self._mark_error(fp, f"No completion message for {name}; assuming finished")
                    else:
                        self._mark_error(fp, f"Process exited with code {proc.exitcode}")
                to_remove.append(fp)
        for fp in to_remove:
            self._cleanup_proc(fp)

        # Try to start more if capacity is available
        self._maybe_start_more()

        # schedule next pump
        self.after(100, self._pump_updates, None)

    def _cleanup_proc(self, file_path):
        proc = self.running_procs.pop(file_path, None)
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass

    def _update_progress(self, file_path, steps):
        if file_path not in self.running_items:
            return
        idx, _ = self.running_items[file_path]
        name = os.path.basename(file_path)
        target = self.file_steps_target.get(file_path)
        if target is not None:
            label = f"{name} - steps: {steps}/{target}"
        else:
            label = f"{name} - steps: {steps}"
        try:
            self.center_list.delete(idx)
            self.center_list.insert(idx, label)
            self.running_items[file_path] = (idx, label)
        except tk.TclError:
            self._rebuild_center_mapping(file_path, label)

    def _mark_finished(self, file_path, out_path):
        # Remove from center
        self._remove_from_center(file_path)
        # Add to right
        name = os.path.basename(out_path)
        self.right_list.insert(tk.END, name)

    def _mark_finished_planar(self, file_path, out_path):
        # Remove from center
        self._remove_from_center(file_path)
        # Add to right with planar tag
        name = os.path.basename(out_path)
        self.right_list.insert(tk.END, f"{name} [planar]")

    def _mark_error(self, file_path, err):
        self._remove_from_center(file_path)
        name = os.path.basename(file_path)
        self.right_list.insert(tk.END, f"FAILED: {name} ({err})")

    def _remove_from_center(self, file_path):
        if file_path not in self.running_items:
            # Ensure we also drop the target cache if lingering
            self.file_steps_target.pop(file_path, None)
            return
        idx, _ = self.running_items.pop(file_path)
        try:
            self.center_list.delete(idx)
        except tk.TclError:
            pass
        # Drop target cache
        self.file_steps_target.pop(file_path, None)
        self._rebuild_all_center_indices()

    def _rebuild_center_mapping(self, file_path, label):
        # fallback: rebuild mapping by names
        self._rebuild_all_center_indices()
        if file_path in self.running_items:
            idx, _ = self.running_items[file_path]
            try:
                self.center_list.delete(idx)
                self.center_list.insert(idx, label)
                self.running_items[file_path] = (idx, label)
            except tk.TclError:
                pass

    def _rebuild_all_center_indices(self):
        # brute-force rebuild mapping based on labels order
        new_map = {}
        for i in range(self.center_list.size()):
            label = self.center_list.get(i)
            base = label.split(' - steps', 1)[0]
            for fp, (_, _old_label) in list(self.running_items.items()):
                if os.path.basename(fp) == base:
                    new_map[fp] = (i, label)
                    break
        self.running_items = new_map

    def _on_close(self):
        # Try to cleanly stop child processes
        for _, proc in list(self.running_procs.items()):
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        self.running_procs.clear()
        try:
            self.mp_queue.close()
        except Exception:
            pass
        self.destroy()


def main():
    # Ensure export dir exists
    os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)
    # Create and run the app
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
