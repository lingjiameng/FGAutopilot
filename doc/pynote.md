# python learning note for this project



## [geographiclib](https://geographiclib.sourceforge.io/html/python/index.html)

1. [dictionary](https://geographiclib.sourceforge.io/html/python/interface.html#geodesic-dictionary)

2. Inverse()

    ```python
    from geographiclib.geodesic import Geodesic
    Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)
    {'lat1': -41.32,
     'a12': 179.6197069334283,
     's12': 19959679.26735382,
     'lat2': 40.96,
     'azi2': 18.825195123248392,
     'azi1': 161.06766998615882, #第二个点相对于第一个点的方位角
     'lon1': 174.81,
     'lon2': -5.5}
    ```



## pandas 

 1. `Dataframe`添加一行

    ```python
    df.loc[i] = {'a':1,'b':2}
    df.loc[i] = [1,2]
    ```

 2. 归一化

    ```
    df - df.min() / (df.max() - df.min())
    ```

 3. 合并

    ```python
    pd.concat(objs, axis=0)
    ```

 4. 删除包含`NaN`的列或行

    ```python
    #删除表中全部为NaN的行
    df.dropna(axis=0,how='all') 
    #删除表中含有任何NaN的行
    df.dropna(axis=0,how='any') #drop all rows that have any NaN values
    #改变为axis=1,即可删除列
    ```

 5. 替换`NaN`

    ```python
    # 替换NaN为value
    df.fillna(value)
    ```

    
