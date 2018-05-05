import numpy as np
from tkinter import *
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import importlib

#-----------------------------------------------------------------------------------------------------------------
'''Used to make a time stamp print out'''
#from recSys4t import *
#print(clock)

'''used to import turicreate file'''
#import turicreate as tc
#from recSys4tc import turiWork
#turiWork()
#-----------------------------------------------------------------------------------------------------------------

'''GUI creation of recommender system using Lightfm'''
# gu
# root is main page
root = Tk()
root.title('CS4319 Movie Recommender System')
root.geometry('920x640+0+0')

heading = Label(root, text="MOVIE RECOMMENDER SYSTEM",font=("arial",12,"bold"), fg="steelblue").pack()

label1 = Label(root, text="USER: ", font=("arial",10,"bold"),fg="black").place(x=10,y=100)
label2 = Label(root, text="USER: ", font=("arial",10,"bold"),fg="black").place(x=10,y=125)
label3 = Label(root, text="USER: ", font=("arial",10,"bold"),fg="black").place(x=10,y=150)

#input for users
u1 = IntVar()
u2 = IntVar()
u3 = IntVar()
#input for fav movie similar search
m1 = IntVar()

#u1 = ''
entry_box1 = Entry(root, textvariable=u1, width=8,bg="lightblue",exportselection=0).place(x=50, y=100)
entry_box2 = Entry(root, textvariable=u2, width=8,bg="lightblue",exportselection=0).place(x=50, y=125)
entry_box2 = Entry(root, textvariable=u3, width=8,bg="lightblue",exportselection=0).place(x=50, y=150)

#fetch data from Lightfm, use movies with user rating of 4.0 or greater
data = fetch_movielens(min_rating=4.0)

#create model
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=32, num_threads=4)

# outputs recommendation to the terminal
def userRec(user_id,fav_movies,top_items):
    #print out the results
    print('------------------------------------------')
    print('User %s' % user_id)
    print('Favorite Movies:')
        
    for x in fav_movies[:5]:
        print('%s' % x)
            
    print('\nRecommended Movies:')
        
    for y in top_items[:5]:
        print('%s' % y)

# outputs recommendation to the GUI
def guiRec(user_id,fav_movies,top_items):
    #print out the results

    newwin = Toplevel(root)
    text = Text(newwin)
    
    text.insert(INSERT, 'User %s\n' % user_id)
    text.insert(INSERT, 'Favorite Movies:\n')
    
    for x in fav_movies[:5]:
        text.insert(INSERT, '%s\n' % x)
        text.pack()
        
    text.insert(INSERT, '\nRecommended Movies:\n')
        
    for y in top_items[:5]:
        text.insert(INSERT,'%s\n' % y)
        text.pack()

# movie recommender sys
def sample_recommendation(model, data, user_ids):
    
    user1 = int(u1.get())
    user2 = int(u2.get())
    user3 = int(u3.get())
    
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    
    #generate recommendations for each user we input
    for user_id in user_ids:
        
        #movies they already like
        fav_movies = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies our model predicts they will like
        scores = model.predict(user_id,np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        #print out the results
        userRec(user_id,fav_movies,top_items)
        guiRec(user_id,fav_movies,top_items)

def Recommend_user():
    #print(type(u1))
    user1 = int(u1.get())
    user2 = int(u2.get())
    user3 = int(u3.get())
    #print(type(user))
    sample_recommendation(model, data, [user1,user2,user3])
    
#Collaborative algorithm using user ratings at 4.0 to find similar items that have been rated >=4.0 and similar
Collab_algo = Button(root, text="Recommend Movies",
 width=16,height=2,bg="gray",command=Recommend_user).place(x=8,y=175)

#spacer

#input for favorite movie
label4 = Label(root, text="what movie are you interested in?  ", font=("arial",10,"bold"),fg="black").place(x=10,y=225)
label5 = Label(root, text="Selection: ", font=("arial",10,"bold"),fg="black").place(x=10,y=250)
entry_box3 = Entry(root, textvariable=m1, width=8,bg="lightgreen",exportselection=0).place(x=50, y=275)

simText = Text(root)
like = int(m1.get())

#show first 5 movies in movielens list
for i in range(5):
    simText.insert(INSERT, '(' + str(i+1) +')' + data[('item_labels')][i] + '\n')
    simText.pack()

def item_similarity():
    like = int(m1.get())
    simText.insert(INSERT,'Great! You might also like these movies: \n ')
    simText.pack()
    
    simText.insert(INSERT, data[('item_labels')][cosine_similarity(model.item_embeddings)[like-1].argsort()][-5:][::-1])
    simText.pack()
    simText.insert(INSERT,'\n')
    
sim_item = Button(root, text="Search for similar movies",
 width=22,height=2,bg="silver",command=item_similarity).place(x=8,y=325)

root.mainloop()
