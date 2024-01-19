# Import necessary libraries
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# wetsern drafted U15

class Promotion_ML:
    def run(self):
        with open('players_test.json', 'r') as json_file:
            data = json.load(json_file)

        # Convert the JSON data to a DataFrame
        test_players = data['test_players']
        test_player_data = pd.DataFrame(test_players)

        test_promotions = []
        for player in test_players:
            test_promotions.append(player['promoted'])
        test_player_promotions = pd.Series(test_promotions)

        print(test_players, test_promotions)
        print(test_player_data, test_player_promotions)

        # # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # # Standardize the features
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        # # Create a RandomForestClassifier (you can choose another classifier based on your needs)
        # classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # # Train the classifier
        # classifier.fit(X_train, y_train)

        # # Make predictions on the test set
        # predictions = classifier.predict(X_test)

        # # Evaluate the model
        # accuracy = accuracy_score(y_test, predictions)
        # report = classification_report(y_test, predictions)

        # print(f'Accuracy: {accuracy}')
        # print('Classification Report:\n', report)

        # # Now, you can use the trained model to predict whether a new player will be promoted
        # new_player_data = pd.DataFrame([[new_feature1, new_feature2, new_feature3, ...]])
        # new_player_data = scaler.transform(new_player_data)  # Standardize the new player's features
        # prediction = classifier.predict(new_player_data)

        # print(f'The prediction for the new player is: {prediction}')


if __name__ == '__main__':
    Promotion_ML().run()