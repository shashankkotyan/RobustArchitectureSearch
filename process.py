#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email:  shashankkotyan@gmail.com
"""

import traceback, multiprocessing as mp


class Process(mp.Process):


    def __init__(self, *args, **kwargs):

        mp.Process.__init__(self, *args, **kwargs)

        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    
    def run(self):
        
        try:
            
            mp.Process.run(self)
            self._cconn.send(None)

        except Exception as e:

            tb = traceback.format_exc()
            self._cconn.send((e, tb))


    @property
    def exception(self):

        if self._pconn.poll(): self._exception = self._pconn.recv()
        
        return self._exception
