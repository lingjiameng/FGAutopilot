import sys
sys.path.append("..")
from fgmodule.fgcmd import FG_CMD
import time



def replayandstop(fg):
    re=fg._put_cmd_common_use("run replay")
    re+=fg._put_cmd_common_use("set /sim/freeze/replay-state 3")
    re+=fg._put_cmd_common_use("set /sim/replay/disable true")
    return re

def main():
    fg = FG_CMD(('localhost', 5555))
    fg.start()
    while True:
        cmd = input("input:")
        if cmd=="replayandstop":
            re = replayandstop(fg)
        else:
            re = fg._put_cmd_common_use(cmd)
        print(re)
        # fg.reposition()
        # fg.auto_start()
    return 0

    # fg.quit()

if __name__ == '__main__':
    main()
'''
成功
nasal
setprop('/sim/crashed', "true");##EOF##


失败
nasal
ac_state.save()
##EOF##


成功
nasal
var filename = getprop("/sim/fg-home") ~ "/Export/state.xml";
io.write_properties( path: filename, prop: "/position" );
##EOF##


nasal
var filename = getprop("/sim/fg-home") ~ "/Export/state.xml";
io.write_properties( path: filename, prop: "/" );
##EOF##


成功，但是只有一小部分飞机信息
nasal
aircraft.data.save();
##EOF##

成功但是效果不对
nasal
var filename = getprop("/sim/fg-home") ~ "/Export/state.xml";
io.read_properties(filename, "/sim");
##EOF##

成功 但是只有位置会重置
nasal
var filename = getprop("/sim/fg-home") ~ "/Export/state.xml";
io.read_properties(filename, "/position");
##EOF##

成功
nasal
var filename = getprop("/sim/fg-home") ~ "/Export/state.xml";
io.read_properties(filename, "/");
##EOF##

'''


'''
**********not work********
run replay
set /sim/replay/disable-my-controls 0
setprop("/sim/replay/disable-my-controls",1)
'''

'''
********
run replay
set /sim/freeze/replay-state 3
set /sim/replay/disable true
'''
