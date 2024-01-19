import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class Promotion_ML:
    def run(self):
        with open('players_test.json', 'r') as json_file:
            data = json.load(json_file)

        # -----------PLAYER DATA-----------
        # Convert JSON player data to a DataFrame
        players_data = pd.DataFrame(data['test_players'])

        # Transform 'born' & 'height'
        players_data['born'] = pd.to_datetime(players_data['born'])
        players_data[['height_feet', 'height_inches']] = players_data['height'].str.split("'", expand=True).astype(int)

        # One-hot encode 'position', 'league', and 'shoots'
        players_data = pd.get_dummies(players_data, columns=['position', 'league', 'shoots'],
                                      prefix=['position', 'league', 'shoots'])

        # Drop unnecessary columns
        players_data = players_data.drop(['id', 'name'], axis=1)

        # Convert all columns to numeric
        players_data = players_data.apply(pd.to_numeric, errors='coerce')

        # Impute missing values with the median
        players_data = players_data.fillna(players_data.median())

        # Replace infinite values with a large numeric value
        players_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        players_data.fillna(1e10, inplace=True)  # Use a large numeric value, adjust as needed

        # -----------PROMOTION DATA-----------
        # Convert JSON promotion data to a DataFrame
        players_promotions = pd.DataFrame(data['test_promotions'])

        # Convert 'year_promoted' to datetime
        players_promotions['year_promoted'] = pd.to_datetime(players_promotions['year_promoted'], format='%Y')

        # Extract the year for classification
        players_promotions['year_promoted'] = players_promotions['year_promoted'].dt.year

        # Drop unnecessary columns
        players_promotions = players_promotions.drop(['id'], axis=1)

        # Impute missing values in 'year_promoted' with the median
        players_promotions['year_promoted'] = players_promotions['year_promoted'].fillna(players_promotions['year_promoted'].median())

        # Replace infinite values in 'year_promoted' with a large numeric value
        players_promotions.replace([np.inf, -np.inf], np.nan, inplace=True)
        players_promotions['year_promoted'].fillna(1e10, inplace=True)  # Use a large numeric value, adjust as needed

        # -----------START TRAINING-----------
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(players_data, players_promotions['promoted'], test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create a RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the classifier
        classifier.fit(X_train, y_train.astype(int))  # Ensure y_train is cast to int

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test.astype(int), predictions)  # Ensure y_test is cast to int
        report = classification_report(y_test.astype(int), predictions, zero_division=1)


        print(f'Training Accuracy: {accuracy}')
        print('Training Classification Report:\n', report)

        # -----------MAKE PREDICTION FOR NEW PLAYER-----------
        # did not make it
        new_player = pd.DataFrame([{
            "id": 13,
            "name": "Ryan Ulmer",
            "born": "2007-10-06",
            "height": "5'9",
            "weight": 154,
            "shoots": "L",
            "position": "F",
            "league": "SAAHL U15",
            "GP": 31,
            "G": 25,
            "A": 29,
            "TP": 54,
            "PIM": 12
        }])

        # Transform 'born' & 'height'
        new_player['born'] = pd.to_datetime(new_player['born'])
        new_player[['height_feet', 'height_inches']] = new_player['height'].str.split("'", expand=True).astype(int)

        # One-hot encode 'position' and 'shoots' for the new player
        new_player['position_' + new_player['position']] = 1
        new_player['shoots_' + new_player['shoots']] = 1

        # Drop the original 'position' and 'shoots' columns
        new_player = new_player.drop(['position', 'shoots'], axis=1)

        # Align the new player columns with the training data columns
        new_player = new_player.reindex(columns=players_data.columns, fill_value=0)

        # Convert all columns to numeric
        new_player = new_player.apply(pd.to_numeric, errors='coerce')

        # Impute missing values with the median
        new_player = new_player.fillna(new_player.median())

        # Replace infinite values with a large numeric value
        new_player.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_player.fillna(1e10, inplace=True)  # Use a large numeric value, adjust as needed

        # Ensure the new player data has the same columns as the training data
        new_player = new_player.reindex(columns=players_data.columns, fill_value=0)

        # Standardize the new player's features
        new_player_data = scaler.transform(new_player)

        # Make a prediction for the new player
        new_player_prediction = classifier.predict(new_player_data)

        print(f'The prediction for the new player is: {new_player_prediction}')


if __name__ == '__main__':
    Promotion_ML().run()
