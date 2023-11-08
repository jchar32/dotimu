import os
import wx
from typing import List, Tuple

def get_path(wildcard: str = ""):
    """Simply gui to get file path(s). If no file is selected, returns "".

    Args:
        wildcard (str, optional): can be a filter to only display specific file types. Examples: "*.csv", "*.json", "*.py". Defaults to "".

    Returns:
        direc (str | List[str]): directory of files selected
        file (str | List[str]): file name(s)
    """
    current_dir = os.getcwd()
    app = wx.App(redirect=False)

    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.STAY_ON_TOP | wx.FD_MULTIPLE | wx.FD_NO_FOLLOW
    dialog = wx.FileDialog(None, "Select file(s) you wihs to load", defaultDir=current_dir, wildcard=wildcard, style=style)

    direc = ""
    file = ""

    if dialog.ShowModal() == wx.ID_CANCEL:
        direc = ""
        file = ""

    if dialog.ShowModal() == wx.ID_OK:
        paths = dialog.GetPaths()

        if len(paths) == 1:
            direc = os.path.dirname(paths)
            file = os.path.basename(paths)
        elif len(paths) > 1:
            direc = []
            file = []
            for i, p in enumerate(paths):
                direc.append(os.path.dirname(p))
                file.append(os.path.basename(p))
        else:
            print("No file selected")
            direc = ""
            file = ""

    dialog.Destroy()
    app.Destroy()
    return direc, file

if __name__ == "__main__":
    d, p = get_path()
    for i in d:
        print(i)
    for j in p:
        print(j)
