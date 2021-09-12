import pandas as pd

df = pd.DataFrame({'std': [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10010, 10009, 10011],
                   'mean': [88958, 72527, 43311, 74057, 94692, 43311, 88070, 94409, 94409, 25828]})

print(df)

df['排序-1'] = df.sort_values(by=['std', 'mean'])['mean'].rank(method='first', ascending=False)

# dt = df.sort_values(by=['排序-1'])

print(df['排序-1'].values.astype(int) )
# dt = df.sort_values(by=['排序-1'])
#
# print(dt)
