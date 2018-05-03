import turicreate as tc
import pandas as pd
import time

#Create Item similarity model
def itemSim():
    
    #import data
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

    #Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

    #title columns
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1')
    #ratings_base.shape, ratings_test.shape
    
    movie_title_id = [ratings_base, items]
    df_for_training = pd.concat(movie_title_id)
    
    movie_title_rate = [ratings_test, items]
    df_for_test = pd.concat(movie_title_rate)
    
    '''TuriWork'''
    #Train model using data
    train_data = tc.SFrame(ratings_base)
    test_data = tc.SFrame(ratings_test)
    
    #create item similarity model
    k = 5 #set k to 5
    item_sim_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
    item_sim_recomm = item_sim_model.recommend(users=[1,2],k=k)
    item_sim_recomm.print_rows(num_rows=25)
    
    #convert SFrame to pd dataframe
    pd_itemSim = item_sim_recomm.to_dataframe()

    #Print top movies for first user
    print('\n\nGreat! User 1 may enjoy watching: ')
    #print out the top 5 movies using k =5
    for i in range(0,k):
        print(items[items.movie_id == pd_itemSim['movie_id'][i]].movie_title)
        
    #Print top movies for second user
    print('\n\nGreat! User 2 may enjoy watching: ')
    #print out the top 5 movies using k =5
    for i in range(k,k*2):
        print(items[items.movie_id == pd_itemSim['movie_id'][i]].movie_title)        


#Item Popularity model
def itemPop():

    '''import data'''
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

    #Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

    #title columns
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1')
    #ratings_base.shape, ratings_test.shape
    
    movie_title_id = [ratings_base, items]
    df_for_training = pd.concat(movie_title_id)
    
    movie_title_rate = [ratings_test, items]
    df_for_test = pd.concat(movie_title_rate)

    '''TuriWork'''
    #Train model using data
    train_data = tc.SFrame(ratings_base)
    test_data = tc.SFrame(ratings_test)

    #Create popularity model
    k=5
    popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
    #k=5 specifies top 5 recommendations to be given
    popularity_recomm = popularity_model.recommend(users=[1,2],k=k)
    
    #convert SFrame to pd dataframe
    pd_popItem = popularity_recomm.to_dataframe()
    
    #print out the top 5 movies for user 1 using k =5
    print('\n\nGreat! User 1 may enjoy watching: ')
    for i in range(0,k):
        print(items[items.movie_id == pd_popItem['movie_id'][i]].movie_title)

    #print out the top 5 movies for user 2 using k =5
    print('\n\nGreat! User 2 may enjoy watching: ')
    for i in range(k,k*2):
        print(items[items.movie_id == pd_popItem['movie_id'][i]].movie_title)
    return popularity_model
 
 
#Ranking Factorization Model
def rankFactor():
    '''import data'''
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

    #Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

    #title columns
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1')
    #ratings_base.shape, ratings_test.shape
    
    movie_title_id = [ratings_base, items]
    df_for_training = pd.concat(movie_title_id)
    
    movie_title_rate = [ratings_test, items]
    df_for_test = pd.concat(movie_title_rate)

    '''TuriWork'''
    #Train model using data
    train_data = tc.SFrame(ratings_base)
    test_data = tc.SFrame(ratings_test)
    
    #Create rank factorization model
    k=5
    rankFactor_model = tc.ranking_factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
    #k=5 specifies top 5 recommendations to be given
    rankFactor_recomm = rankFactor_model.recommend(users=[1,2],k=k)
    
    #convert SFrame to pd dataframe
    pd_rankFactor = rankFactor_recomm.to_dataframe()
    
    #print out the top 5 movies for user 1 using k =5
    print('\n\nGreat! User 1 may enjoy watching: ')
    for i in range(0,k):
        print(items[items.movie_id == pd_rankFactor['movie_id'][i]].movie_title)

    #print out the top 5 movies for user 2 using k =5
    print('\n\nGreat! User 2 may enjoy watching: ')
    for i in range(k,k*2):
        print(items[items.movie_id == pd_rankFactor['movie_id'][i]].movie_title)
        

def modelCompare():
    
    '''import data'''
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

    #Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

    #title columns
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1')
    #ratings_base.shape, ratings_test.shape
    
    movie_title_id = [ratings_base, items]
    df_for_training = pd.concat(movie_title_id)
    
    movie_title_rate = [ratings_test, items]
    df_for_test = pd.concat(movie_title_rate)

    '''TuriWork'''
    #Train model using data
    train_data = tc.SFrame(ratings_base)
    test_data = tc.SFrame(ratings_test)
    
    k=5
    
    #Create rank factorization model
    rankFactor_model = tc.ranking_factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
    #k=5 specifies top 5 recommendations to be given
    rankFactor_recomm = rankFactor_model.recommend(users=[1,2],k=k)
    
    #Create popularity model
    popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
    #k=5 specifies top 5 recommendations to be given
    popularity_recomm = popularity_model.recommend(users=[1,2],k=k)
    
    #create item similarity model
    item_sim_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
    item_sim_recomm = item_sim_model.recommend(users=[1,2],k=k)
    
    tc.recommender.util.compare_models(test_data, [rankFactor_model, popularity_model,item_sim_model], model_names=["Rank Factorization Model", "Item Popularity Model", "Item Similarity Model"])

    
