import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('MAZE.csv')

# Inspect the data to understand its structure
print(data.head())

# Assuming the second column is 'game_play' and columns 4-6 are the required values
# Adjust column indices if necessary based on the output from data.head()
x = data.iloc[:, 1]  # Second column as x-axis (game plays)
y1 = data.iloc[:, 3]  # Fourth column (PIAIRL)
y2 = data.iloc[:, 4]  # Fifth column (MFIRL)
y3 = data.iloc[:, 5]  # Sixth column (EXPERT)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='PIAIRL')
plt.plot(x, y2, label='MFIRL')
plt.plot(x, y3, label='EXPERT')
plt.xlabel('Game Plays')
plt.ylabel('Values')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
