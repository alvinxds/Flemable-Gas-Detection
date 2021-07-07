# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:51:27 2019

@author: anton
"""

import wx

# Define the tab content as classes:
class TabOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        
        
class TabTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        t = wx.StaticText(self, -1, "This is the second tab", (20,20))


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Gas Detection Algorithmic Methods",
                          size=(720, 480))

        # Create a panel and notebook (tabs holder)
        p = wx.Panel(self)
        nb = wx.Notebook(p)

        # Create the tab windows
        tab1 = TabOne(nb)
        tab2 = TabTwo(nb)
        

        # Add the windows to tabs and name them.
        nb.AddPage(tab1, "Method 1")
        nb.AddPage(tab2, "Method 2")
        

        # Set noteboook in a sizer to create the layout
        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        p.SetSizer(sizer)

del app

if __name__ == "__main__":
    app = wx.App()
    MainFrame().Show()
    app.MainLoop()

