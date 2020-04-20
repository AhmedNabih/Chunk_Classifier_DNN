# Chunk_Classifier_DNN
* just a sudoku chunk classifier by a DNN for the purpose of Learning to master DNN.
* every chunk is a 3x3 blocks and a sudoku board is a 9x9 blocks, so every row and column in the 9x9 blocks will correspond to a certain index.
* every thing is 1-based indexes
* i.e. row = 5 and column = 8 => index = 44.
* every chunk have 3x3 blocks maps to it's chunk number.
### Goal
- Is to map index number to chunk number using a DNN.

```
1  2  3     4  5  6     7  8  9
10 11 12    13 14 15    16 17 18
19 20 21    22 23 24    25 26 27
________    ________    ________
1           2           3

28 29 30    31 32 33    34 35 36
37 38 39    40 41 42    43 44 45
46 47 48    49 50 51    52 53 54
________    ________    ________
4           5           6

55 56 57    58 59 60    61 62 63
64 65 66    67 68 69    70 71 72
73 74 75    76 77 78    79 80 81
________    ________    ________
7           8           9
```
### DNN Architecture
- output layer have an activation sigmoid function any thing else is an relu activation function.
![DNN Architecture](https://github.com/AhmedNabih/Chunk_Classifier_DNN/blob/master/Images/DNN_Architecture2.PNG)
