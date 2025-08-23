import torch
import numpy
import pandas
import xgboost
import seaborn
import matplotlib.pyplot
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pandas.read_parquet("processed_videos.parquet")

y = data["likes_views_ratio"]

title_embeddings_matrix = numpy.vstack(data["title_embeddings"].values)
thumbnail_embeddings_matrix = numpy.vstack(data["thumbnail_embeddings"].values)

x = numpy.hstack([
    data[["timestamp", "duration(sec)"]].values,
    title_embeddings_matrix,
    thumbnail_embeddings_matrix
])

x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

model = xgboost.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    device="cuda",
    early_stopping_rounds=30,
    n_jobs=-1
)

model.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=False
)

x_test_gpu = torch.tensor(x_test).cuda()
y_pred = model.predict(x_test_gpu)
r2 = r2_score(y_test, y_pred)

print(f"R-squared score: {r2:.4f}\n")

seaborn.scatterplot(x=y_test, y=y_pred)
matplotlib.pyplot.xlabel("Actual likes to views ratio")
matplotlib.pyplot.ylabel("Predicted likes to views ratio")
matplotlib.pyplot.title("Actual vs. Predicted likes to views ratio")
matplotlib.pyplot.show()
