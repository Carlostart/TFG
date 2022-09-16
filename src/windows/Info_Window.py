from cgitb import text
from constants import messages as msg

import webbrowser
import tkinter as tk
from tkinter import ttk


class Info_Window(tk.Toplevel):
    def __init__(self, parent, data):
        super().__init__(parent)
        self.title(msg.FOUND_TILLE)

        # f1 = tk.Frame(self)
        # f1.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        # f2 = tk.Frame(self)
        # f2.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        # f3 = tk.Frame(self)
        # f3.pack(fill=tk.BOTH, expand=1, padx=15, pady=15)
        # tk.Label(f1, text=msg.COIN_NAME_LABEL).pack(side=tk.LEFT)
        # tk.Label(f1, text=data.get("NAME")).pack(side=tk.RIGHT)
        # tk.Label(f2, text=msg.COIN_DESC_LABEL).pack(side=tk.LEFT)
        # tk.Label(f2, text=data.get("DESCRIPTION")).pack(side=tk.RIGHT)
        # tk.Label(f3, text=msg.COIN_URL_LABEL).pack(side=tk.LEFT)
        # link = tk.Label(f3, text=data.get("URL"), fg="blue", cursor="hand2")
        # link.pack(side=tk.RIGHT)
        # link.bind("<Button-1>", lambda e: webbrowser.open_new_tab(data.get("URL")))
        # tk.Button(self, text="OK", command=self.destroy).pack(
        #     expand=1, padx=15, pady=15
        # )

        labelframe = ttk.LabelFrame(self, text=msg.FOUND_TILLE)
        labelframe.grid(column=0, row=0, padx=5, pady=10)

        label1 = ttk.Label(labelframe, text=msg.COIN_NAME_LABEL)
        label1.grid(column=0, row=1, padx=4, pady=4)
        name_tb = tk.Text(labelframe, height=1, width=50)
        name_tb.insert("end", data.get("NAME"))
        name_tb.grid(column=1, row=1, padx=4, pady=4)

        label2 = ttk.Label(labelframe, text=msg.COIN_YEAR_LABEL)
        label2.grid(column=0, row=2, padx=4, pady=4)
        year_tb = tk.Text(labelframe, height=1, width=50)
        year_tb.insert("end", data.get("YEAR"))
        year_tb.grid(column=1, row=2, padx=4, pady=4)

        label3 = ttk.Label(labelframe, text=msg.COIN_DESC_LABEL)
        label3.grid(column=0, row=3, padx=4, pady=4)
        desc_tb = tk.Text(labelframe, height=5, width=50)
        desc_tb.insert("end", data.get("DESCRIPTION"))
        desc_tb.grid(column=1, row=3, padx=4, pady=4)
        sb = tk.Scrollbar(labelframe)
        sb.grid()
        desc_tb.config(yscrollcommand=sb.set)

        label4 = ttk.Label(labelframe, text=msg.COIN_URL_LABEL)
        label4.grid(column=0, row=4, padx=4, pady=4)
        url_tb = tk.Text(labelframe, height=1, width=50)
        url_tb.insert("end", data.get("URL"))
        url_tb.grid(column=1, row=4, padx=4, pady=4)

        button = ttk.Button(labelframe, text=msg.CLOSE_BTN, command=self.destroy)
        button.grid(column=1, row=5, padx=4, pady=4)
