# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
#

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('_titanic.csv', index_col="PassengerId")
df = pd.DataFrame(data, columns=['Pclass', 'Fare', 'Age', 'Sex'])

df[df['Sex'] == 'male'] = 1
df[df['Sex'] == 'female'] = 2

#df = df.notnull()

X = np.array(df)
Y = np.array(pd.DataFrame(data, columns=['Survived']))

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

importances = clf.feature_importances_
print(importances[0])
