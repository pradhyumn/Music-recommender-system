from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from scipy.sparse import coo_matrix
import pandas as pd
from time import perf_counter

df_train=pd.read_parquet('train_als.parquet')
df_test=pd.read_parquet('test_als.parquet')

df_test = df_test.sort_values(["user_id", "avg_rating"], ascending=[True, False])

df_train['user_id'] = df_train['user_id'].astype(int)
df_train['recording_id_index'] = df_train['recording_id_index'].astype(int)

df_test['user_id'] = df_test['user_id'].astype(int)
df_test['recording_id_index'] = df_test['recording_id_index'].astype(int)

interactions_train = coo_matrix((df_train['avg_rating'], 
                                 (df_train['user_id'], df_train['recording_id_index'])))

interactions_test = coo_matrix((df_test['avg_rating'], 
                                (df_test['user_id'], df_test['recording_id_index'])))

# Instantiate and train the model
model = LightFM(no_components=10,loss='warp')
print("Starting Training: ")
start_time = perf_counter()
model.fit(interactions_train, epochs=10, num_threads=8)
time_taken = perf_counter() - start_time
print("Time taken to train " + str(time_taken))
test_precision = precision_at_k(model, interactions_test, k=100).mean()