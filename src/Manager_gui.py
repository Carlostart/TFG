from constants import messages as msg, colors
from screens.Home import Home

import os, subprocess
from PIL import Image, ImageTk
import tkinter as tk
from im_processing import cv2_to_imageTK
from tkinter.filedialog import askopenfilenames, askdirectory


class Window(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(msg.TITTLE)
        self.geometry("610x610")

        self.init_widgets()

        self.paths = []

    def init_widgets(self):

        self.paths_preview = tk.StringVar(self)
        self.paths_preview.set(msg.NO_SELECTED)
        self.browse_files_text = tk.StringVar(self)
        self.browse_files_text.set(msg.BROWSE_FILE)
        self.browse_dir_text = tk.StringVar(self)
        self.browse_dir_text.set(msg.BROWSE_DIR)

        tk.Label(self, text=msg.HEADER, **colors.STYLE).pack(side=tk.TOP, pady=8)

        browse_frame = tk.Frame(self)
        browse_frame.pack(side=tk.TOP, fill=tk.BOTH)

        tk.Label(browse_frame, textvariable=self.paths_preview).pack(
            side=tk.LEFT, padx=8
        )

        tk.Button(
            browse_frame,
            textvariable=self.browse_files_text,
            command=self.browse_filenames,
        ).pack(side=tk.RIGHT, padx=5)
        tk.Button(
            browse_frame,
            textvariable=self.browse_dir_text,
            command=self.browse_folder,
        ).pack(side=tk.RIGHT, padx=5)

        displayed_imgs_frame = tk.Frame(
            self, width=600, height=400, background=colors.BACKGROUND
        )
        tk.Button(
            self,
            text=msg.SHOW_IMGS,
            command=lambda: self.update_displayed_imgs(displayed_imgs_frame),
        ).pack(fill=tk.BOTH, padx=5)
        displayed_imgs_frame.pack(side=tk.TOP, fill=tk.BOTH)

        action_buttons_frame = tk.Frame(self)
        action_buttons_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        tk.Button(
            action_buttons_frame, text=msg.FIND_BTN, command=self.find_command
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        tk.Button(
            action_buttons_frame, text=msg.ADD_BTN, command=self.add_command
        ).pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def browse_filenames(self):
        self.browse_files_text.set(msg.BROWSING)
        self.paths = list(askopenfilenames(parent=self, title=msg.ASK_FILE))
        self.browse_files_text.set(msg.BROWSE_FILE)
        self.paths_preview.set(self.get_path_preview(self.paths))

    def browse_folder(self):
        self.browse_dir_text.set(msg.BROWSING)
        dir = askdirectory(parent=self, title=msg.ASK_DIR)
        self.paths = [os.path.join(dir, file) for file in os.listdir(dir)]
        self.browse_dir_text.set(msg.BROWSE_DIR)
        self.paths_preview.set(self.get_path_preview(self.paths))

    def get_path_preview(self, paths):
        preview = ""
        for path in paths:
            if len(preview) < 50:
                preview += path.split("/")[-1] + ", "
            else:
                preview = preview[:50] + " ..."
        return preview

    def update_displayed_imgs(self, frame):
        for wd in frame.winfo_children():
            print(wd)
            wd.destroy()
        for im_path, i in zip(self.paths, range(6)):
            img_label = tk.Label(
                frame, width=200, height=200, background=colors.BACKGROUND
            )
            img = Image.open(im_path)

            if img.width > img.height:
                f = 200 / img.width
                img = img.resize((200, int(img.height * f)))
            else:
                f = 200 / img.height
                img = img.resize((int(img.width * f), 200))

            img_label.img = ImageTk.PhotoImage(img)
            img_label["image"] = img_label.img

            img_label.grid(column=i % 3, row=i % 2)
            img = ImageTk.PhotoImage(img)

    def find_command(self):
        command = ["python", "-u", "src/console.py", "find"]
        for pth in self.paths:
            command += [pth]

        subprocess.Popen(command)

    def add_command(self):
        command = ["python", "-u", "src/console.py", "add"]
        for pth in self.paths:
            command += [pth]

        subprocess.Popen(command)
