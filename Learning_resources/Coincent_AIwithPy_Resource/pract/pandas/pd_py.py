Python 3.10.6 (main, Nov 14 2022, 16:10:14) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
df=pd.DataFrame([1,3,5,12,6,8],[10,11,12,20,50,8])
df
     0
10   1
11   3
12   5
20  12
50   6
8    8
df=pd.DataFrame({'A':[1,3,5,12,6,8],'B':[10,11,12,20,50,8]},index=[0,1,2,3,4,5])
df
    A   B
0   1  10
1   3  11
2   5  12
3  12  20
4   6  50
5   8   8
df=pd.DataFrame({'a':[1,2,8,4],'b':[5,6,9,8],'c':[11,12,30,14]},index=[0,1,2,3,4,5])
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#5>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 663, in __init__
    mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 493, in dict_to_mgr
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 123, in arrays_to_mgr
    arrays = _homogenize(arrays, index, dtype)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 620, in _homogenize
    com.require_length_match(val, index)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/common.py", line 561, in require_length_match
    raise ValueError(
ValueError: Length of values (4) does not match length of index (6)
df=pd.DataFrame({'a':[1,2,8,4],'b':[5,6,9,8],'c':[11,12,30,14]},index=[0,1,2,3])
df
   a  b   c
0  1  5  11
1  2  6  12
2  8  9  30
3  4  8  14
df=pd.DataFrame({'X':[78,85,96,80,86],'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]},index=[0,1,2,3,4])
df
    X   Y   Z
0  78  84  86
1  85  94  97
2  96  89  96
3  80  83  72
4  86  86  83
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19], 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1], 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']} labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
SyntaxError: invalid syntax
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19], 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1], 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#11>", line 1, in <module>
NameError: name 'np' is not defined
import numpy as np
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19], 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1], 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
exam_data
{'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'], 'score': [12.5, 9, 16.5, nan, 9, 20, 14.5, nan, 8, 19], 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1], 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data)
df
        name  score  attempts qualify
0  Anastasia   12.5         1     yes
1       Dima    9.0         3      no
2  Katherine   16.5         2     yes
3      James    NaN         3      no
4      Emily    9.0         2      no
5    Michael   20.0         3     yes
6    Matthew   14.5         1     yes
7      Laura    NaN         1      no
8      Kevin    8.0         2      no
9      Jonas   19.0         1     yes
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df=pd.DataFrame(exam_data,index=labels)
df
        name  score  attempts qualify
a  Anastasia   12.5         1     yes
b       Dima    9.0         3      no
c  Katherine   16.5         2     yes
d      James    NaN         3      no
e      Emily    9.0         2      no
f    Michael   20.0         3     yes
g    Matthew   14.5         1     yes
h      Laura    NaN         1      no
i      Kevin    8.0         2      no
j      Jonas   19.0         1     yes
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df=pd.DataFrame(exam_data,labels)
df
        name  score  attempts qualify
a  Anastasia   12.5         1     yes
b       Dima    9.0         3      no
c  Katherine   16.5         2     yes
d      James    NaN         3      no
e      Emily    9.0         2      no
f    Michael   20.0         3     yes
g    Matthew   14.5         1     yes
h      Laura    NaN         1      no
i      Kevin    8.0         2      no
j      Jonas   19.0         1     yes
df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10 entries, a to j
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   name      10 non-null     object 
 1   score     8 non-null      float64
 2   attempts  10 non-null     int64  
 3   qualify   10 non-null     object 
dtypes: float64(1), int64(1), object(2)
memory usage: 400.0+ bytes
df.info
<bound method DataFrame.info of         name  score  attempts qualify
a  Anastasia   12.5         1     yes
b       Dima    9.0         3      no
c  Katherine   16.5         2     yes
d      James    NaN         3      no
e      Emily    9.0         2      no
f    Michael   20.0         3     yes
g    Matthew   14.5         1     yes
h      Laura    NaN         1      no
i      Kevin    8.0         2      no
j      Jonas   19.0         1     yes>
df.describe
<bound method NDFrame.describe of         name  score  attempts qualify
a  Anastasia   12.5         1     yes
b       Dima    9.0         3      no
c  Katherine   16.5         2     yes
d      James    NaN         3      no
e      Emily    9.0         2      no
f    Michael   20.0         3     yes
g    Matthew   14.5         1     yes
h      Laura    NaN         1      no
i      Kevin    8.0         2      no
j      Jonas   19.0         1     yes>
df.head(3)
        name  score  attempts qualify
a  Anastasia   12.5         1     yes
b       Dima    9.0         3      no
c  Katherine   16.5         2     yes
df=pd.DataFrame(exam_data,index=labels)
df.iloc(:,[2,0,3,1])
SyntaxError: invalid syntax
df.iloc[:,[2,0,3,1]]
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df=df.iloc[:,[2,0,3,1]]
df
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df.describe
<bound method NDFrame.describe of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>


df.head(3)
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
df.iloc[:,[1,3]]
        name  score
a  Anastasia   12.5
b       Dima    9.0
c  Katherine   16.5
d      James    NaN
e      Emily    9.0
f    Michael   20.0
g    Matthew   14.5
h      Laura    NaN
i      Kevin    8.0
j      Jonas   19.0
df.loc[:,['name','score']]
        name  score
a  Anastasia   12.5
b       Dima    9.0
c  Katherine   16.5
d      James    NaN
e      Emily    9.0
f    Michael   20.0
g    Matthew   14.5
h      Laura    NaN
i      Kevin    8.0
j      Jonas   19.0
df.loc[['b','d','f','g']:,['score','qualify']]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 144, in pandas._libs.index.IndexEngine.get_loc
TypeError: '['b', 'd', 'f', 'g']' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#38>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 1067, in __getitem__
    return self._getitem_tuple(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 1256, in _getitem_tuple
    return self._getitem_tuple_same_dim(tup)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 924, in _getitem_tuple_same_dim
    retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 1290, in _getitem_axis
    return self._get_slice_axis(key, axis=axis)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 1324, in _get_slice_axis
    indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6602, in slice_indexer
    start_slice, end_slice = self.slice_locs(start, end, step=step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6810, in slice_locs
    start_slice = self.get_slice_bound(start, "left")
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6723, in get_slice_bound
    slc = self.get_loc(label)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3810, in get_loc
    self._check_indexing_error(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 5968, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: ['b', 'd', 'f', 'g']
df.loc[['b','d','f','g'],['score','qualify']]
   score qualify
b    9.0      no
d    NaN      no
f   20.0     yes
g   14.5     yes
df.iloc[[1,4,6,7],[2,3]]
  qualify  score
b      no    9.0
e      no    9.0
g     yes   14.5
h      no    NaN
df.iloc[[1,3,5,6],[2,3]]
  qualify  score
b      no    9.0
d      no    NaN
f     yes   20.0
g     yes   14.5
df.iloc[[1,3,5,6],[3,2]]
   score qualify
b    9.0      no
d    NaN      no
f   20.0     yes
g   14.5     yes
df[df['attempts']>2]
   attempts     name qualify  score
b         3     Dima      no    9.0
d         3    James      no    NaN
f         3  Michael     yes   20.0
df[df[0]>2]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 165, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 5745, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 5753, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#44>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3805, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    raise KeyError(key) from err
KeyError: 0
df[df[1]>2]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 165, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 5745, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 5753, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 1

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#45>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3805, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    raise KeyError(key) from err
KeyError: 1
df[df['attempts']>2]
   attempts     name qualify  score
b         3     Dima      no    9.0
d         3    James      no    NaN
f         3  Michael     yes   20.0
df
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0

##8
df.count()
attempts    10
name        10
qualify     10
score        8
dtype: int64
df.count(rows)
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#51>", line 1, in <module>
NameError: name 'rows' is not defined
df.count(df)
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#52>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 10688, in count
    axis = self._get_axis_number(axis)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 554, in _get_axis_number
    return cls._AXIS_TO_AXIS_NUMBER[axis]
TypeError: unhashable type: 'DataFrame'
df.count
<bound method DataFrame.count of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>
df.count('axis'=1)
SyntaxError: expression cannot contain assignment, perhaps you meant "=="?
df.count(axis=1)
a    4
b    4
c    4
d    3
e    4
f    4
g    4
h    3
i    4
j    4
dtype: int64
df.count(axis=0)
attempts    10
name        10
qualify     10
score        8
dtype: int64
df.count(df.count())
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#57>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 10688, in count
    axis = self._get_axis_number(axis)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 554, in _get_axis_number
    return cls._AXIS_TO_AXIS_NUMBER[axis]
TypeError: unhashable type: 'Series'
df.count(df.count)
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 554, in _get_axis_number
    return cls._AXIS_TO_AXIS_NUMBER[axis]
KeyError: <bound method DataFrame.count of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#58>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 10688, in count
    axis = self._get_axis_number(axis)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 556, in _get_axis_number
    raise ValueError(f"No axis named {axis} for object type {cls.__name__}")
ValueError: No axis named <bound method DataFrame.count of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0> for object type DataFrame
df.count(axis=1)
a    4
b    4
c    4
d    3
e    4
f    4
g    4
h    3
i    4
j    4
dtype: int64
df.describe
<bound method NDFrame.describe of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>
df.describe()
        attempts      score
count  10.000000   8.000000
mean    1.900000  13.562500
std     0.875595   4.693746
min     1.000000   8.000000
25%     1.000000   9.000000
50%     2.000000  13.500000
75%     2.750000  17.125000
max     3.000000  20.000000
df.info
<bound method DataFrame.info of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>
df.info()
<class 'pandas.core.frame.DataFrame'>
Index: 10 entries, a to j
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   attempts  10 non-null     int64  
 1   name      10 non-null     object 
 2   qualify   10 non-null     object 
 3   score     8 non-null      float64
dtypes: float64(1), int64(1), object(2)
memory usage: 700.0+ bytes
df.info('index')
<class 'pandas.core.frame.DataFrame'>
Index: 10 entries, a to j
Data columns (total 4 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   attempts  10 non-null     int64  
 1   name      10 non-null     object 
 2   qualify   10 non-null     object 
 3   score     8 non-null      float64
dtypes: float64(1), int64(1), object(2)
memory usage: 700.0+ bytes
df.columns
Index(['attempts', 'name', 'qualify', 'score'], dtype='object')
df.count(df.columns)
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#66>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 10688, in count
    axis = self._get_axis_number(axis)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 554, in _get_axis_number
    return cls._AXIS_TO_AXIS_NUMBER[axis]
TypeError: unhashable type: 'Index'
df.columns()
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#67>", line 1, in <module>
TypeError: 'Index' object is not callable
df.columns
Index(['attempts', 'name', 'qualify', 'score'], dtype='object')
df.rows
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#69>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 5902, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'rows'
df.sample
<bound method NDFrame.sample of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>
len(df)
10
len(df.columns)
4
len(axis[0)
    
SyntaxError: closing parenthesis ')' does not match opening parenthesis '['
len(axis[0])
    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#74>", line 1, in <module>
NameError: name 'axis' is not defined
##9
    
df[df['score']==NaN]
    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#76>", line 1, in <module>
NameError: name 'NaN' is not defined
df[df['score']=='NaN']
    
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []
df[df['score']==None]
    
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []
df[df['score']=None]
    
SyntaxError: cannot assign to subscript here. Maybe you meant '==' instead of '='?
df[df['score']==""]
    
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []

df.isnull()
    
   attempts   name  qualify  score
a     False  False    False  False
b     False  False    False  False
c     False  False    False  False
d     False  False    False   True
e     False  False    False  False
f     False  False    False  False
g     False  False    False  False
h     False  False    False   True
i     False  False    False  False
j     False  False    False  False
df.isnull
    
<bound method DataFrame.isnull of    attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0>
df[df.isnull()]
    
   attempts name qualify  score
a       NaN  NaN     NaN    NaN
b       NaN  NaN     NaN    NaN
c       NaN  NaN     NaN    NaN
d       NaN  NaN     NaN    NaN
e       NaN  NaN     NaN    NaN
f       NaN  NaN     NaN    NaN
g       NaN  NaN     NaN    NaN
h       NaN  NaN     NaN    NaN
i       NaN  NaN     NaN    NaN
j       NaN  NaN     NaN    NaN
df
    
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df[df.isnull]
    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#86>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3759, in __getitem__
    key = com.apply_if_callable(key, self)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/common.py", line 364, in apply_if_callable
    return maybe_callable(obj, **kwargs)
TypeError: DataFrame.isnull() takes 1 positional argument but 2 were given
df[df.isna]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#87>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3759, in __getitem__
    key = com.apply_if_callable(key, self)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/common.py", line 364, in apply_if_callable
    return maybe_callable(obj, **kwargs)
TypeError: DataFrame.isna() takes 1 positional argument but 2 were given
df[df.isna()]
   attempts name qualify  score
a       NaN  NaN     NaN    NaN
b       NaN  NaN     NaN    NaN
c       NaN  NaN     NaN    NaN
d       NaN  NaN     NaN    NaN
e       NaN  NaN     NaN    NaN
f       NaN  NaN     NaN    NaN
g       NaN  NaN     NaN    NaN
h       NaN  NaN     NaN    NaN
i       NaN  NaN     NaN    NaN
j       NaN  NaN     NaN    NaN
df.isna()
   attempts   name  qualify  score
a     False  False    False  False
b     False  False    False  False
c     False  False    False  False
d     False  False    False   True
e     False  False    False  False
f     False  False    False  False
g     False  False    False  False
h     False  False    False   True
i     False  False    False  False
j     False  False    False  False
df[df['score'].isna()]
   attempts   name qualify  score
d         3  James      no    NaN
h         1  Laura      no    NaN
df[df['score'].isnull()]
   attempts   name qualify  score
d         3  James      no    NaN
h         1  Laura      no    NaN
##10
df[df['score']>15 * df['score']<20]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#93>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 1527, in __nonzero__
    raise ValueError(
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
df[df[('score']>15) * (df['score']<20)]
SyntaxError: closing parenthesis ']' does not match opening parenthesis '('
df[df[(['score']>15) * (df['score']<20)]
   ]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#96>", line 1, in <module>
TypeError: '>' not supported between instances of 'list' and 'int'
df[df[(['score']>15) * (df['score']<20)]

   ;
   
SyntaxError: '[' was never closed
df[df[(['score']>15) * (df['score']<20)]]
   
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#100>", line 1, in <module>
TypeError: '>' not supported between instances of 'list' and 'int'
df[df[(['score']>15)]
   ]
   
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#102>", line 1, in <module>
TypeError: '>' not supported between instances of 'list' and 'int'
df[df['score']>15]
   
   attempts       name qualify  score
c         2  Katherine     yes   16.5
f         3    Michael     yes   20.0
j         1      Jonas     yes   19.0
df[df['score']>15 * df['score']<20]
   
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#104>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 1527, in __nonzero__
    raise ValueError(
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
df[df['score']<20]
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
e         2      Emily      no    9.0
g         1    Matthew     yes   14.5
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df[15<df['score']<20]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#106>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 1527, in __nonzero__
    raise ValueError(
ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
df[df['score']<20]
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
e         2      Emily      no    9.0
g         1    Matthew     yes   14.5
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df[df['score'].between(15,20)]
   attempts       name qualify  score
c         2  Katherine     yes   16.5
f         3    Michael     yes   20.0
j         1      Jonas     yes   19.0
df[df['score']>15 & df['score']<20]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 305, in na_logical_op
    result = op(x, y)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/roperator.py", line 54, in rand_
    return operator.and_(right, left)
TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 319, in na_logical_op
    result = libops.scalar_binop(x, y, op)
  File "pandas/_libs/ops.pyx", line 180, in pandas._libs.ops.scalar_binop
ValueError: Buffer dtype mismatch, expected 'Python object' but got 'double'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#109>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/common.py", line 72, in new_method
    return method(self, other)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/arraylike.py", line 76, in __rand__
    return self._logical_method(other, roperator.rand_)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/series.py", line 6254, in _logical_method
    res_values = ops.logical_op(lvalues, rvalues, op)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 395, in logical_op
    res_values = na_logical_op(lvalues, rvalues, op)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 328, in na_logical_op
    raise TypeError(
TypeError: Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]
df2=df[df['score']>15 & df['score']<20]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 305, in na_logical_op
    result = op(x, y)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/roperator.py", line 54, in rand_
    return operator.and_(right, left)
TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 319, in na_logical_op
    result = libops.scalar_binop(x, y, op)
  File "pandas/_libs/ops.pyx", line 180, in pandas._libs.ops.scalar_binop
ValueError: Buffer dtype mismatch, expected 'Python object' but got 'double'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#110>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/common.py", line 72, in new_method
    return method(self, other)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/arraylike.py", line 76, in __rand__
    return self._logical_method(other, roperator.rand_)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/series.py", line 6254, in _logical_method
    res_values = ops.logical_op(lvalues, rvalues, op)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 395, in logical_op
    res_values = na_logical_op(lvalues, rvalues, op)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/ops/array_ops.py", line 328, in na_logical_op
    raise TypeError(
TypeError: Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]
df2=df[(df['score']>15) & (df['score']<20)]
df2
   attempts       name qualify  score
c         2  Katherine     yes   16.5
j         1      Jonas     yes   19.0
df2=df[(df['score']>15) * (df['score']<20)]
df2
   attempts       name qualify  score
c         2  Katherine     yes   16.5
j         1      Jonas     yes   19.0
df[(df['score']>15) * (df['attempts']<2)]
   attempts   name qualify  score
j         1  Jonas     yes   19.0
df[df['score']]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#116>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3811, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6113, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6173, in _raise_if_missing
    raise KeyError(f"None of [{key}] are in the [{axis_name}]")
KeyError: "None of [Float64Index([12.5, 9.0, 16.5, nan, 9.0, 20.0, 14.5, nan, 8.0, 19.0], dtype='float64')] are in the [columns]"
df[df['score']==20]
   attempts     name qualify  score
f         3  Michael     yes   20.0
df[0:0]
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []
df[2:1]
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []
df[1:2]
   attempts  name qualify  score
b         3  Dima      no    9.0
df[0:2]
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
df[0:]
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
d         3      James      no    NaN
e         2      Emily      no    9.0
f         3    Michael     yes   20.0
g         1    Matthew     yes   14.5
h         1      Laura      no    NaN
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
df[:0]
Empty DataFrame
Columns: [attempts, name, qualify, score]
Index: []
df[1:2]
   attempts  name qualify  score
b         3  Dima      no    9.0
df[1:2:2]
   attempts  name qualify  score
b         3  Dima      no    9.0
df[1:2:3]
   attempts  name qualify  score
b         3  Dima      no    9.0
df[1:2,3]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 144, in pandas._libs.index.IndexEngine.get_loc
TypeError: '(slice(1, 2, None), 3)' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#127>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3805, in __getitem__
    indexer = self.columns.get_loc(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3810, in get_loc
    self._check_indexing_error(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 5968, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError: (slice(1, 2, None), 3)
df[1:2]
   attempts  name qualify  score
b         3  Dima      no    9.0
df[df[1:2]]
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#129>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3792, in __getitem__
    return self.where(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    return func(*args, **kwargs)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 331, in wrapper
    return func(*args, **kwargs)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 11920, in where
    return super().where(
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 211, in wrapper
    return func(*args, **kwargs)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/util/_decorators.py", line 331, in wrapper
    return func(*args, **kwargs)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 9968, in where
    return self._where(cond, other, inplace, axis, level)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 9657, in _where
    raise ValueError(msg.format(dtype=dt))
ValueError: Boolean array expected for the condition, not object
df[df[1:2]:2]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 144, in pandas._libs.index.IndexEngine.get_loc
TypeError: '   attempts  name qualify  score
b         3  Dima      no    9.0' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#130>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3777, in __getitem__
    indexer = convert_to_index_sliceable(self, key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 2494, in convert_to_index_sliceable
    return idx._convert_slice_indexer(key, kind="getitem", is_frame=True)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 4327, in _convert_slice_indexer
    indexer = self.slice_indexer(start, stop, step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6602, in slice_indexer
    start_slice, end_slice = self.slice_locs(start, end, step=step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6810, in slice_locs
    start_slice = self.get_slice_bound(start, "left")
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6723, in get_slice_bound
    slc = self.get_loc(label)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3810, in get_loc
    self._check_indexing_error(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 5968, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError:    attempts  name qualify  score
b         3  Dima      no    9.0
df[df[1:2]:2]
Traceback (most recent call last):
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3803, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 138, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 144, in pandas._libs.index.IndexEngine.get_loc
TypeError: '   attempts  name qualify  score
b         3  Dima      no    9.0' is an invalid key

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#131>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/frame.py", line 3777, in __getitem__
    indexer = convert_to_index_sliceable(self, key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexing.py", line 2494, in convert_to_index_sliceable
    return idx._convert_slice_indexer(key, kind="getitem", is_frame=True)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 4327, in _convert_slice_indexer
    indexer = self.slice_indexer(start, stop, step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6602, in slice_indexer
    start_slice, end_slice = self.slice_locs(start, end, step=step)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6810, in slice_locs
    start_slice = self.get_slice_bound(start, "left")
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6723, in get_slice_bound
    slc = self.get_loc(label)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3810, in get_loc
    self._check_indexing_error(key)
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 5968, in _check_indexing_error
    raise InvalidIndexError(key)
pandas.errors.InvalidIndexError:    attempts  name qualify  score
b         3  Dima      no    9.0
df.iloc()
<pandas.core.indexing._iLocIndexer object at 0x7f8fd4dc6b10>
df.iloc([1:2])
SyntaxError: invalid syntax
df.loc[['b']]
   attempts  name qualify  score
b         3  Dima      no    9.0
df.loc[['b'],['score']]
   score
b    9.0
df.loc[['b'],['score']]
   score
b    9.0
df.loc[['b'],['score']]=11.
df.loc[['b'],['score']]=11.5
df.loc[['b'],['score']]
   score
b   11.5
df['attempts'].sum
<bound method NDFrame._add_numeric_operations.<locals>.sum of a    1
b    3
c    2
d    3
e    2
f    3
g    1
h    1
i    2
j    1
Name: attempts, dtype: int64>
df['attempts'].sum()
19
##14
df['score'].me()
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#143>", line 1, in <module>
  File "/home/fansan/.local/lib/python3.10/site-packages/pandas/core/generic.py", line 5902, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'Series' object has no attribute 'me'. Did you mean: 'e'?
df['score'].mean()
13.875
df.describe()
        attempts      score
count  10.000000   8.000000
mean    1.900000  13.875000
std     0.875595   4.421942
min     1.000000   8.000000
25%     1.000000  10.875000
50%     2.000000  13.500000
75%     2.750000  17.125000
max     3.000000  20.000000
df.loc[['b'],['score']]=9
df['score'].mean()
13.5625

##15