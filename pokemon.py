## Pokemon Stats Analysis and ML Project

### Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

from scipy.stats import ttest_ind
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

plt.style.use('ggplot')

### Load Data

pokemon_df = pd.read_csv('data/pokemon.csv')
pokemon_df.head()

### Exploratory Data Analysis

# Check nulls and dtypes
pokemon_df.info()
pokemon_df.isnull().mean()

# Display summary statistics
pokemon_df.describe()

#### Data Cleaning

# Fill NA's for `Type 2`
pokemon_df['Type 2'] = pokemon_df['Type 2'].fillna('')

# Drop unecessary coluns
pokemon = pokemon_df.drop(['#','Total', 'Generation', 'Legendary'], axis=1)
pokemon.head(2)

#### Statistical Analysis

# Mean and Median of Attack and Defense
mean_attack = pokemon.Attack.mean()
mean_defense = pokemon.Defense.mean()
median_attack = pokemon.Attack.median()
median_defense = pokemon.Defense.median()

print(f"Mean Attack: {mean_attack:.2f}")
print(f"Mean Defense: {mean_defense:.2f}")
print(f"Median Attack: {median_attack:.2f}")
print(f"Median Defense: {median_defense:.2f}")


# Mean and Median of HP and Speed
mean_hp = pokemon.HP.mean()
mean_speed = pokemon.Speed.mean()
median_hp = pokemon.HP.median()
median_speed = pokemon.Speed.median()

print(f"Mean Attack: {mean_hp:.2f}")
print(f"Mean Defense: {mean_speed:.2f}")
print(f"Median Attack: {median_hp:.2f}")
print(f"Median Defense: {median_speed:.2f}")


# Standard Deviations
std_attack = pokemon.Attack.std()
std_defense = pokemon.Defense.std()
std_hp = pokemon.HP.std()
std_speed = pokemon.Speed.std()

print(f"Standard Deviation of Attack: {std_attack:.2f}")
print(f"Standard Deviation of Defense: {std_defense:.2f}")
print(f"Standard Deviation of HP: {std_hp:.2f}")
print(f"Standard Deviation of Speed: {std_speed:.2f}")


# Filter the data for the generations of interest
generation_1_attack = pokemon_df[pokemon_df["Generation"] == 1]["Attack"]
generation_2_attack = pokemon_df[pokemon_df["Generation"] == 2]["Attack"]
generation_3_attack = pokemon_df[pokemon_df["Generation"] == 3]["Attack"]

# Perform t-test for Attack stat between Generation 1 and Generation 2
t_stat, p_value = ttest_ind(generation_1_attack, generation_2_attack)

# Print the results
print("Gen 1 and Gen 2 T-statistic for Attack Attribute:", t_stat)
print("Gen 1 and Gen 2 p-value for Attack Attribute:", p_value)

# Perform t-test for Attack stat between Generation 1 and Generation 3
t_stat, p_value = ttest_ind(generation_1_attack, generation_3_attack)

# Print the results
print("Gen 1 and Gen 3 T-statistic for Attack Attribute:", t_stat)
print("Gen 1 and Gen 3 p-value for Attack Attribute:", p_value)

#### Data Visualizations

# Create pair plot to explore relatonship between variables
sns.pairplot(data=pokemon)

# Create boxplot so show the distributions of our features
plt.figure(figsize=(7, 5))
plt.title("Feature Distribution")
sns.boxplot(data=pokemon)

pokemon.groupby(['Type 1'])["Attack", "Defense", "Speed"].mean()

# Create boxplot so show the distributions of Type 1 Attack stats
plt.figure(figsize=(7, 5))
plt.title("Type 1 Attack Distribution")
sns.boxplot(data=pokemon, x="Attack", y="Type 1")


# Create boxplot so show the distributions of Type 2 Attack stats
plt.figure(figsize=(7, 5))
plt.title("Type 2 Attack Distribution")
sns.boxplot(data=pokemon, x="Attack", y="Type 2")


# Create boxplot so show the distributions of Type 1 Defense stats
plt.figure(figsize=(7, 5))
plt.title("Type 1 Defense Distribution")
sns.boxplot(data=pokemon, x="Defense", y="Type 1")


# Create boxplot so show the distributions of Type 2 Attack stats
plt.figure(figsize=(7, 5))
plt.title("Type 2 Defense Distribution")
sns.boxplot(data=pokemon, x="Defense", y="Type 2")


# Create scatter plot plot to show relationship between attack and defense
plt.figure(figsize=(7, 5))
plt.title("Relationship between Attack and Defense")
sns.regplot(data=pokemon, x="Attack", y="Defense", color="r")
plt.show()


# Using Pearson Correlation to find relation between our variables
corr = pokemon.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("darkgrid"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, square=True, cmap='Reds', annot=True)


# Group pokemon by type and and find the statistical averages
stats_avg = pokemon.groupby(['Type 1'])["HP", "Attack", "Defense", "Speed", "Sp. Atk", "Sp. Def"].mean().reset_index()
stats_avg_2 = pokemon.groupby(['Type 2'])["HP", "Attack", "Defense", "Speed", "Sp. Atk", "Sp. Def"].mean().reset_index()
stats_avg.head()


# Create barplots for Pokemon stats by type 1
plt.figure(figsize=(25, 20))

features = ['HP', 'Attack', 'Defense', 'Speed']
target = stats_avg['Type 1']

for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    x = stats_avg[col]
    y = target
    sns.barplot(data=stats_avg, x=x, y=y, 
        order=stats_avg.sort_values(col)['Type 1'],
        palette="Reds")
    plt.title(col)
    plt.xlabel(col)

# Create barplots for Pokemon stats by type 2
plt.figure(figsize=(25, 20))

features = ['HP', 'Attack', 'Defense', 'Speed']
target = stats_avg_2['Type 2']

for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    x = stats_avg_2[col]
    y = target
    sns.barplot(data=stats_avg_2, x=x, y=y, 
        order=stats_avg_2.sort_values(col)['Type 2'],
        palette="Reds")
    plt.title(col)
    plt.xlabel(col)


# Sort the Pokemon data by Attack stat in descending order
top_10_attack = pokemon_df.sort_values(by="Attack", ascending=False).head(10)

# Plot the top 10 attack pokemon

# Create a bar chart of the top 10 "Attack" Pokemon
fig = px.bar(top_10_attack, x="Attack", y="Name", text="Attack",
             title="Top 10 Attack Pokemon", labels={"Name": "Pokemon Name", "Attack": "Attack Stat"})
fig.update_layout(height=400, width=600)
fig.show()


# Filter the data for Generation 3
generation_3_data = pokemon_df[pokemon_df["Generation"] == 3]

# Sort by Attack stat in descending order
top_10_by_generation = generation_3_data.sort_values("Attack", ascending=False).head(10)

# Round Attack stat to 2 decimals
top_10_by_generation["Attack"] = top_10_by_generation["Attack"].round(2)

# Create a bar chart
fig = px.bar(top_10_by_generation, x="Name", y="Attack", text="Attack",
             title="Top 10 Attack Pokemon for Generation 3", labels={"Name": "Pokemon Name", "Attack": "Attack Stat"})
fig.update_layout(height=400, width=600)
fig.show()


# Filter the data for Pokemon with Legendary status set to True
top_10_legendary = pokemon_df[pokemon_df["Legendary"] == False].nlargest(10, "Attack")

# Round the Attack stat to 2 decimal places
top_10_legendary["Attack"] = top_10_legendary["Attack"].round(2)

# Create a bar chart of the top 10 Pokemon w/out Legendary status
fig = px.bar(top_10_legendary, x="Name", y="Attack", text="Attack",
             title="Top 10 Pokemon non-Legendary by Attack Stat",
             labels={"Name": "Pokemon Name", "Attack": "Attack Stat"})
fig.update_layout(width=600, height=500) 
fig.show()


# ### Predictive Analysis

# #### Data Preprocessing

pokemon_df = pd.read_csv('data/pokemon.csv')
pokemon_df.head()

# Define our target and feature variables
X = pokemon_df.iloc[:,2:10]
y = pokemon_df.Legendary


# Create pipeline to preprocess and transform the categorical data
categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="")),
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)


# Create pipeline to preprocess and transform the numerical data
numerical_pipeline = Pipeline(
    steps=[
    ("inpute", SimpleImputer(strategy="mean")),
    ("StandardScaler", StandardScaler())
    ]
)

# Define the numerical and categorical data
cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns



# Define transformer
transformer = ColumnTransformer(
    transformers=[
    ("numeric", numerical_pipeline, num_cols),
    ("categorical", categorical_pipeline, cat_cols)
    ]
)

# Apply Processing
X = transformer.fit_transform(X)
y = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42)


# Create dictionary of models
models = {
    "SVC": SVC(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier()
    
}



# Store model result
model_results = []
model_names = []

# Train the models
for name,model in models.items():
    history = model.fit(X_train, y_train)
    preds = history.predict(X_test)
    score = accuracy_score(y_test, preds)
    model_results.append(score)
    model_names.append(name)
    
    # Create dataframe and print results
    df_results = pd.DataFrame([model_names, model_results])
    df_results = df_results.transpose()
    df_results = df_results.rename(columns={0:'Model', 1:'Accuracy'}).sort_values(by='Accuracy', ascending=False)
    
df_results


#### Hyperpareter tuning: XGBClassifier using GridSearchCV


# Define the hyperparameter grid
param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}

# Instantiate XGB Classifier
xgb_model = XGBClassifier(objective="binary:logistic",
                          random_state=42)

# Instantiate Grid Search
grid_cv = GridSearchCV(xgb_model,
                       param_grid,
                       n_jobs=-1,
                       cv=3,
                       scoring="roc_auc")

history = grid_cv.fit(X_train, y_train)

print("-"*100)
print("Best parameters found: ", grid_cv.best_params_)
print("Best ROC AUC Score found: ",(grid_cv.best_score_))


# Train model using best parameters
xgb_model = XGBClassifier(**grid_cv.best_params_, random_state=42)

xgb_model.fit(X_train, y_train)
predicted = xgb_model.predict(X_test)
print(f'Accuracy Score = {accuracy_score(y_test, predicted)}')


#### Model Evaluation

# Plot ROC Curve
fpr, tpr, thresh = roc_curve(y_test, predicted)

# Calculate ROC curve
fpr, tpr, thresh = roc_curve(y_test, predicted)

# Create ROC trace
roc_trace = go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    line=dict(width=2, color='blue'),
    name='ROC curve'
)

# Create diagonal line
diag_trace = go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    line=dict(width=1, color='gray', dash='dash'),
    showlegend=False
)

# Create layout
layout = go.Layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate')
)

# Create figure
fig = go.Figure(data=[roc_trace, diag_trace], layout=layout)

# Show plot
fig.show()

roc_auc = roc_auc_score(y_test, predicted)
print(f"ROC AUC Score: {roc_auc:.3f}")



