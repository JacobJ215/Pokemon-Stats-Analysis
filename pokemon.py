#!/usr/bin/env python
# coding: utf-8

## Pokemon Stats Analysis and ML Project

### Import Libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

print(f"Standard Deviatoin of Attack: {std_attack:.2f}")
print(f"Standard Deviatoin of Defense: {std_defense:.2f}")
print(f"Standard Deviatoin of HP: {std_hp:.2f}")
print(f"Standard Deviatoin of Speed: {std_speed:.2f}")


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

