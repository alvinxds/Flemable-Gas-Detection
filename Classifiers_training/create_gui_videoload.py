# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:59:11 2019

@author: anton
"""

#import wx python
import wx, wx.media

class RandomPanel(wx.Panel):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, parent, color):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour(color)

def button_pressed(e):
    frm.player.Load(video_path)

def button_pause(e):
    frm.player.Pause() 

def button_stop(e):
    frm.player.Stop()  

def button_continue(e):
    frm.player.Play()  

def video_is_loaded(e,):
    frm.player.Play() 
 
#sets the path to the mp3 file
video_path = 'C:/Users/anton/Documents/Emp/Thesis/Python_code/gas_detect/foregrd_det_backr_sub/vid_backgr2.mp4'
 
#creates a wxPython App with the 'print' output sent to the terminal, rather
# than the wx log
app = wx.App(redirect=False)
 
#create  Frame without any parent
frm = wx.Frame(None, id=wx.ID_ANY, title="Gas Detection Algorithmic Methods",
                          size=(680, 480))

topSplitter = wx.SplitterWindow(frm)
panelOne = RandomPanel(topSplitter,"blue")
panelTwo = RandomPanel(topSplitter,"green")

topSplitter.SplitHorizontally(panelOne,panelTwo)

#creates the MediaCtrl, invisible to us, with the frame as a parent, and
# the WMP10 backend set
frm.player = wx.media.MediaCtrl(parent=frm, szBackend=wx.media.MEDIABACKEND_WMP10)
 
#Binds the EVT_MEDIA_LOADED to the function song_is_loaded written above,
# so that when the song gets loaded to memory, it actually gets played.
frm.Bind(wx.media.EVT_MEDIA_LOADED, video_is_loaded)
 
#a wx Sizer so that the two widjets gets places
main_sizer = wx.BoxSizer()
 
#the play button, to start the load
btn = wx.Button(frm, label='Play')
btn2 = wx.Button(frm, label='Pause')
btn3 = wx.Button(frm, label='Continue')
btn4 = wx.Button(frm, label='Stop')

#binding the load mp3 function above to the clicking of this button
btn.Bind(wx.EVT_BUTTON, button_pressed)
btn2.Bind(wx.EVT_BUTTON, button_pause)
btn3.Bind(wx.EVT_BUTTON, button_continue)
btn4.Bind(wx.EVT_BUTTON, button_stop)
 
#add the player and the button the sizer, to be placed on the screen
main_sizer.Add(frm.player)
main_sizer.Add(btn)
main_sizer.Add(btn2)
main_sizer.Add(btn3)
main_sizer.Add(btn4)
 
#declare the main_sizer as the top sizer for the entire frame
frm.SetSizer(main_sizer)
#show the frame to the user
frm.Show()
 
#start the mainloop, required for any wxPython app. It kicks off the logic.
print 'running'
app.MainLoop()

del app
