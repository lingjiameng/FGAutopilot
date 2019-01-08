# -*- coding:utf-8 -*-
# flightgear 通信脚本 发送命令
from telnetlib import Telnet
import sys
import socket
import re
import time

# __all__ = ["FlightGear"]

CRLF = '\r\n'


class FGTelnet(Telnet):
    def __init__(self, host, port):
        Telnet.__init__(self, host, port)
        self.prompt = []
        self.prompt.append(re.compile(b'/[^>]*> '))
        self.timeout = 5
        #Telnet.set_debuglevel(self,2)

    def help(self):
        return

    def ls(self, dir=None):
        """
        Returns a list of properties.
        """
        if dir == None:
            self._putcmd('ls')
        else:
            self._putcmd('ls %s' % dir)
        return self._getresp()

    def dump(self):
        """Dump current state as XML."""
        self._putcmd('dump')
        return self._getresp()

    def cd(self, dir):
        """Change directory."""
        self._putcmd('cd ' + dir)
        self._getresp()
        return

    def pwd(self):
        """Display current path."""
        self._putcmd('pwd')
        return self._getresp()

    def get(self, var):
        """Retrieve the value of a parameter."""
        self._putcmd('get %s' % var)
        return self._getresp()

    def set(self, var, value):
        """Set variable to a new value"""
        self._putcmd('set %s %s' % (var, value))
        self._getresp()  # Discard response

    def quit(self):
        """Terminate connection"""
        self._putcmd('quit')
        self.close()
        return

    # Internal: send one command to FlightGear
    def _putcmd(self, cmd):
        cmd = cmd + CRLF
        Telnet.write(self, cmd.encode("ascii"))  # TODO: The error may caused.
        return

    # Internal: get a response from FlightGear
    def _getresp(self):
        (i, match, resp) = Telnet.expect(self, self.prompt, self.timeout)
        # Everything preceding it is the response.
        # print("TTTTTTT")
        # print(str(resp).split('\\r'))
        # print("TTTTTTT")
        # return str(resp).split('\n')[:-1]
        return '\n'.join(str(resp).split('\\r'))
        # return split(resp, '\n')[:-1]


class FG_CMD:
    """FlightGear interface class.

    An instance of this class represents a connection to a FlightGear telnet
    server.

    Properties are accessed using a dictionary style interface:
    For example:

    # Connect to flightgear telnet server.
    fg = FlightGear('myhost', 5500)
    # parking brake on
    fg['/controls/gear/brake-parking'] = 1
    # Get current heading
    heading = fg['/orientation/heading-deg']
    
    Other non-property related methods
    """

    def __init__(self, address):
        try:
            self.telnet = FGTelnet(address[0], address[1])
        except socket.error:
            self.telnet = None
            raise socket.error

    def __del__(self):
        # Ensure telnet connection is closed cleanly.
        self.quit()

    def __getitem__(self, key):
        """Get a FlightGear property value.
        Where possible the value is converted to the equivalent Python type.
        """
        s = self.telnet.get(key)[0]
        match = re.compile('[^=]*=\s*\'([^\']*)\'\s*([^\r]*)\r').match(s)
        if not match:
            return None
        value, type = match.groups()
        #value = match.group(1)
        #type = match.group(2)
        if value == '':
            return None

        if type == '(double)':
            return float(value)
        elif type == '(int)':
            return int(value)
        elif type == '(bool)':
            if value == 'true':
                return 1
            else:
                return 0
        else:
            return value

    # set 命令
    def __setitem__(self, key, value):
        """Set a FlightGear property value."""
        self.telnet.set(key, value)

    # 断开连接 没有测试
    def quit(self):
        """Close the telnet connection to FlightGear."""
        if self.telnet:
            self.telnet.quit()
            self.telnet = None

    #move to next view
    def view_next(self):
        self.telnet.set("/command/view/next", "true")

    #move to next view
    def view_prev(self):
        self.telnet.set("/command/view/prev", "true")

    # 执行任意命令 通用接口
    def _put_cmd_common_use(self, command):
        self.telnet._putcmd(command)
        return self.telnet._getresp()

    # reset命令
    def reset(self):
        self.telnet._putcmd("run reset")
        print("reset完成", end=' ')
        self.print_local_time()
        return None

    # 打印当前时间
    def print_local_time(self):
        local_time = time.localtime()
        print(local_time.tm_hour, ":", local_time.tm_min, ":", local_time.tm_sec)

    # reposition命令
    def reposition(self):
        self.telnet._putcmd("run reposition")
        print("reposition完成", end=' ')
        self.print_local_time()
        return None

    # auto start
    def auto_start(self):
        self.__setitem__("/sim/model/autostart", 1)
        self.__setitem__("/controls/gear/brake-parking", 0)
        print("auto start完成", end=' ')
        self.print_local_time()
        return None