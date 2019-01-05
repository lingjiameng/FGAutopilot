import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("data/flylog/log2019-1-5_13-50-28.csv")
plt.subplot(211)
plt.title("hi-heading")
plt.plot(df.index, df.loc[:, ["hi-heading"]], color="b")


plt.subplot(212)
plt.title("rudder")
plt.plot(df.index, df.loc[:, ["rudder"]],color ="r")

plt.show()
