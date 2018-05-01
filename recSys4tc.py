import turicreate as tc
import pandas as pd
'''TURICREATE Calculations'''
'''Compare model using Turicreate'''    

def turiWork():

    '''import data'''
    #Reading users file:
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

    #Reading ratings file:
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('u.data', sep='\t', names=r_cols, encoding='latin-1')

    #Reading items file:
    i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

    #title columns
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols, encoding='latin-1')
    ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols, encoding='latin-1')
    ratings_base.shape, ratings_test.shape

    '''TuriWork'''
    #Train model using data
    train_data = tc.SFrame(ratings_base)
    test_data = tc.SFrame(ratings_test)

    #Create popularity model
    popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
    #k=5 specifies top 5 recommendations to be given
    popularity_recomm = popularity_model.recommend(users=[1,2],k=5)
    popularity_recomm.print_rows(num_rows=25)

    #Create Item similarity model
    item_sim_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
    item_sim_recomm = item_sim_model.recommend(users=[1,2],k=5)
    item_sim_recomm.print_rows(num_rows=25)

    #Compare popularity to item similarity
    tc.recommender.util.compare_models(test_data, [popularity_model, item_sim_model], model_names=["Popularity Model", "Item Similarity Model"])
