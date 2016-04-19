# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).
#

import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('./_titanic.csv', index_col='PassengerId')
main_data_frame = pandas.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
main_data_frame = main_data_frame[["Pclass", "Fare", "Age", "Sex", "Survived"]].dropna()

filtered_df = main_data_frame[["Pclass", "Fare", "Age", "Sex"]].replace("female", 0).replace("male", 1)

clf = DecisionTreeClassifier(random_state=241)

Y = main_data_frame['Survived']
X = filtered_df
print(X.columns)


clf.fit(X, Y)

print(clf.feature_importances_)

# import numpy as np
# import pandas as pn
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#
#
#
# data = pn.read_csv('_titanic.csv', index_col='PassengerId')
#
#
# filtered_data = data[["Pclass", "Fare", "Age", "Sex", "Survived"]].dropna()
# goal_data = filtered_data["Survived"]
#
# clr = DecisionTreeClassifier(random_state=241)
#
#
# # In[13]:
#
# mutated_filtered_data = filtered_data[["Pclass", "Fare", "Age", "Sex"]].replace("female", 0).replace("male", 1)
#
#
# # In[14]:
#
# mutated_filtered_data.head()
#
#
# # In[15]:
#
# goal_data.head()
#
#
# # In[16]:
#
# clr = clr.fit(mutated_filtered_data, goal_data)
#
#
# # In[17]:
#
# importances = clr.feature_importances_
#
#
# # In[18]:
#
# importances
#
#
# # In[19]:
#
# clr.get_params()
#
#
# # In[20]:
#
# print(list(zip(mutated_filtered_data.columns, clr.feature_importances_)))
#
#
# # In[ ]:


