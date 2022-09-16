from constants import messages as msg
import subprocess
import tkinter as tk
from tkinter import ttk


class Set_Info_Window(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title(msg.COIN_INFO_LABEL)

        labelframe = ttk.LabelFrame(self, text=msg.COIN_INFO_LABEL)
        labelframe.grid(column=0, row=0, padx=5, pady=10)

        label1 = ttk.Label(labelframe, text=msg.COIN_CLASS_LABEL)
        label1.grid(column=0, row=0, padx=4, pady=4)
        coin_class = tk.StringVar()
        entry_class = ttk.Entry(labelframe, textvariable=coin_class)
        entry_class.grid(column=1, row=0, padx=4, pady=4)

        label2 = ttk.Label(labelframe, text=msg.COIN_NAME_LABEL)
        label2.grid(column=0, row=1, padx=4, pady=4)
        name = tk.StringVar()
        entry_name = ttk.Entry(labelframe, textvariable=name)
        entry_name.grid(column=1, row=1, padx=4, pady=4)

        label3 = ttk.Label(labelframe, text=msg.COIN_YEAR_LABEL)
        label3.grid(column=0, row=2, padx=4, pady=4)
        year = tk.StringVar()
        entry_year = ttk.Entry(labelframe, textvariable=year)
        entry_year.grid(column=1, row=2, padx=4, pady=4)

        label3 = ttk.Label(labelframe, text=msg.COIN_DESC_LABEL)
        label3.grid(column=0, row=3, padx=4, pady=4)
        entry_desc = tk.Text(labelframe, height=5, width=15)
        entry_desc.grid(column=1, row=3, padx=4, pady=4)

        label3 = ttk.Label(labelframe, text=msg.COIN_URL_LABEL)
        label3.grid(column=0, row=4, padx=4, pady=4)
        url = tk.StringVar()
        entry_url = ttk.Entry(labelframe, textvariable=url)
        entry_url.grid(column=1, row=4, padx=4, pady=4)

        def set_coin_info():
            command = ["python", "-u", "src/console.py", "set-info"]
            command += [
                coin_class.get(),
                name.get(),
                year.get(),
                entry_desc.get(1.0, "end-1c"),
                url.get(),
            ]
            subprocess.Popen(command)
            self.destroy()

        button = ttk.Button(
            labelframe, text="Asignar Información", command=set_coin_info
        )
        button.grid(column=1, row=5, padx=4, pady=4)
