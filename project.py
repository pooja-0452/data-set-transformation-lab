import pandas as pd
import numpy as np

df = pd.read_csv('day02_usage.csv')

chat = df['Chat'].to_numpy()
video = df['Video'].to_numpy()
study = df['Study'].to_numpy()
games = df['Games'].to_numpy()

print(chat.sum())
print(video.sum())
print(study.sum())
print(games.sum())

print(round(chat.mean(), 1))
print(round(video.mean(), 1))
print(round(study.mean(), 1))
print(round(games.mean(), 1))

diff = study - games

print(diff.argmax())
print(diff.argmin())

#apps = df.columns
apps=['Chat','Video','Study','Games']


data = df[apps].to_numpy()
print(data.shape)

#B — The direction dial
app_totals=data.sum(axis=0)
day_totals=data.sum(axis=1)

app_means=data.mean(axis=0)
day_means=data.mean(axis=1)

app_min = data.min(axis=0)
day_min = data.min(axis=1)

app_max = data.max(axis=0)
day_max = data.max(axis=1)

#C — Kill the nine-line loop
names = np.array(apps)
#names = ['Chat','Video','Study','Games']

'''x = data.argmax(axis=1)
x=[3,2,0,1,3,1,0,2,1,3]
names[[3,2,0,1,3,1,0,2,1,3]]
winningapp_perday = names[x]
#['Video', 'Video', ...]'''

winners = names[data.argmax(axis=1)]

unique, counts = np.unique(winners, return_counts = True)
d=dict(zip(unique,counts.tolist())).items()

print(*d,sep='\n')

#D — Broadcasting: a table divided by a column

per_day = data.sum(axis=1)
#shares = data / per_day
shares = data / per_day[:,None] * 100

print(np.round(shares.mean(axis=0),1))

#(30,)
#(30,1)
print(per_day.shape)
print(per_day[:,None].shape)

per_app_max = data.max(axis=0)
#per_app_max.shape -> (4,)
scale = data/per_app_max

#(30,4)/(4,)
#(30,4)/(30,)
#(30,4)/(30,1)

#E — Questions you could not ask before
weekend_mask = (df['Day'] % 6 == 0 ) | (df['Day'] % 7 == 0)

np.round(data[weekend_mask].mean(axis=0),1)

np.round(data[~weekend_mask].mean(axis=0),1)

print(per_day.argmax()+1,per_day.max())
