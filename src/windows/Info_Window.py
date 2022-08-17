from constants import messages as msg

import webbrowser
import tkinter as tk


class Info_Window(tk.Toplevel):
    def __init__(self, parent, data):
        super().__init__(parent)
        self.geometry("400x250")
        self.title(msg.FOUND_TILLE)

        f1 = tk.Frame(self)
        f1.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        f2 = tk.Frame(self)
        f2.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        f3 = tk.Frame(self)
        f3.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        tk.Label(f1, text=msg.COIN_NAME_LABEL).pack(side=tk.LEFT)
        tk.Label(f1, text=data.get("NAME")).pack(side=tk.RIGHT)
        tk.Label(f2, text=msg.COIN_DESC_LABEL).pack(side=tk.LEFT)
        tk.Label(f2, text=data.get("DESCRIPTION")).pack(side=tk.RIGHT)
        tk.Label(f3, text=msg.COIN_URL_LABEL).pack(side=tk.LEFT)
        link = tk.Label(f3, text=data.get("URL"), fg="blue", cursor="hand2")
        link.pack(side=tk.RIGHT)
        link.bind("<Button-1>", lambda e: webbrowser.open_new_tab(data.get("URL")))
        tk.Button(self, text="OK", command=self.destroy).pack(
            expand=1, padx=15, pady=15
        )
