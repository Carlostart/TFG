from constants import messages as msg
import subprocess
import tkinter as tk


class Set_Info_Window(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("400x250")
        self.title(msg.FOUND_TILLE)

        coin_class = tk.StringVar()
        name = tk.StringVar()
        desc = tk.StringVar()
        url = tk.StringVar()

        f0 = tk.Frame(self)
        f0.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        f1 = tk.Frame(self)
        f1.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        f2 = tk.Frame(self)
        f2.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        f3 = tk.Frame(self)
        f3.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        tk.Label(f0, text=msg.COIN_CLASS_LABEL).pack(side=tk.LEFT)
        tk.Entry(f0, textvariable=coin_class).pack(side=tk.RIGHT)
        tk.Label(f1, text=msg.COIN_NAME_LABEL).pack(side=tk.LEFT)
        tk.Entry(f1, textvariable=name).pack(side=tk.RIGHT)
        tk.Label(f2, text=msg.COIN_DESC_LABEL).pack(side=tk.LEFT)
        tk.Entry(f2, textvariable=desc).pack(side=tk.RIGHT)
        tk.Label(f3, text=msg.COIN_URL_LABEL).pack(side=tk.LEFT)
        tk.Entry(f3, textvariable=url).pack(side=tk.RIGHT)

        def set_coin_info():
            command = ["python", "-u", "src/console.py", "set-info"]
            command += [coin_class.get(), name.get(), desc.get(), url.get()]
            subprocess.Popen(command)
            self.destroy()

        tk.Button(self, text="OK", command=set_coin_info).pack(
            expand=1, padx=15, pady=15
        )
