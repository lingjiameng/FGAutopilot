from fgcmd import FG_CMD
import time

def main():
    fg = FG_CMD('localhost', 5501)
    while True:
        cmd = input("input:")
        re = fg._put_cmd_common_use(cmd)
        print(re)
    return 0

    # fg.quit()

if __name__ == '__main__':
    main()
