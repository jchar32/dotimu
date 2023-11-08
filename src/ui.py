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
    dialog = wx.FileDialog(None, "Select file(s) you want to load", defaultDir=current_dir, wildcard=wildcard, style=style)

    direc = ""
    file = ""

    with wx.FileDialog(None, "Select file(s) you want to load", defaultDir=current_dir, wildcard=wildcard, style=style) as dialog:

        if dialog.ShowModal() == wx.ID_CANCEL:
            direc = ""
            file = ""
            wx.LogError("No file selected")
            return direc, file

        paths = dialog.GetPaths()
        direc = os.path.dirname(paths[0])
        if len(paths) == 1:
            file = os.path.basename(paths)
        elif len(paths) > 1:
            file = []
            for p in paths:
                file.append(os.path.basename(p))
        else:
            wx.LogError("No file selected")

    app.Destroy()
    return direc, file

if __name__ == "__main__":
    d, p = get_path()
    print(f"\n Directory Selected: {d} --> \n")
    for j in p:
        print(j)
