#!/usr/bin/env python

import sys
from PyQt4 import QtGui, QtCore

class myWidget(QtGui.QWidget):
    def __init__ (self, xi=-1, yi=-1, x=250, y=250):
        QtGui.QWidget.__init__(self)
        if (xi is -1) or (yi is -1):
            self.center(x, y)
        else:
            self.setGeometry(xi, yi, x, y)

    def center(self, x, y):
       screen = QtGui.QDesktopWidget()
       ssize = screen.screenGeometry()
       X = ssize.width()
       Y = ssize.height()
       self.setGeometry((X - x)/2, (Y - y)/2, x, y)

class myButton(QtGui.QPushButton):
    def __init__ (self, widget, name, xi=10, yi=10, x=60, y=30):
        QtGui.QPushButton.__init__(self, name, widget)
        self.setGeometry(xi, yi, x, y)

app = QtGui.QApplication(sys.argv)

# create widget
widget = myWidget()
widget.setWindowTitle("Test Widget")
widget.setToolTip("This is a <b>test</b> <i>widget</i>")
quit = myButton(widget,'quit')
widget.connect(quit, QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()'))

widget.show()
sys.exit(app.exec_())
