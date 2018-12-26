# FG Autopilot

**important:** create folder`data/flylog` in the same directory before you run the code.

## code

- `client.py` 主程序
    - 默认100s后关闭，如需更改，只需更改文件尾`time.sleep(90)`为`time.sleep(999999999)`,增加主程序休眠时间
- `server.py` 模仿flight gear 收发数据，用于代码调试

## data

- we will save all the fly log in folder `data/flylog`. their named by the time the log created.