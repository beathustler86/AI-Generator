import sys
import traceback
from pathlib import Path
import tkinter as tk

# Ensure src directory (this file sits in <root>/src) is on sys.path
CURRENT = Path(__file__).resolve().parent
if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))

try:
    from gui.main_window import MainWindow
except Exception as e:
    print("Failed to import MainWindow:", e)
    traceback.print_exc()
    sys.exit(1)

def main():
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()