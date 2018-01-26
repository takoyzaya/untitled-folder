import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import csv

print("Получение обучающей выборки из news_train.txt...")
news_train = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

print("Создание массива новостей...")
count = 1
news_train_data = []
for x in news_train:
    count+=1
    news_train_data.append(x[2])

print("Создание массива тегов новостей...")
count = 1
news_train_data_target = []
for x in news_train:
    count+=1
    news_train_data_target.append(x[0])

'''
print("Cоздание словаря характерных признаков и перевод данных в векторы признаков...")
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(news_train_data)

print("Вычисление частоты и обратной частоты термина...")
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print("Обучение Наивного Байесовского классификатора...")
clf = MultinomialNB().fit(X_train_tfidf, news_train_data_target)

print("Предсказание результатов на новых данных...")
docs_new = ['В Ираке новые танки', 'Экономика России идет на спад']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
count = 0
for item in predicted:
    print('%r => %s' % (docs_new[count], item))
    count+=1

print("Cоздание конвеерной обработки...")
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                    ])

print("Обучение модели...")
_ = text_clf.fit(news_train_data, news_train_data_target)


print("Оценка производительности работы на обучающей выборке...")
'''
print("Получение обучающей выборки новостей из news_train.txt...")
news_training_test = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

print("Создание массива новостей...")
news_training_test_data = []
count = 1
for x in news_training_test:
    count+=1
    news_training_test_data.append(x[2])
    
print("Создание массива тегов новостей...")
count = 1
news_training_test_data_target = []
for x in news_train:
    count+=1
    news_training_test_data_target.append(x[0])
'''    
print("Начало тестирования...")
docs_test = news_training_test_data
predicted = text_clf.predict(docs_test)
print ('Точность тестирования: ', np.mean(predicted == news_training_test_data_target))
'''
docs_test = news_training_test_data
print("Обучение модели с помощью линейного метода опорных векторов...")
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='perceptron', penalty='l2',
                                            alpha=1e-3, max_iter=5, random_state=42, learning_rate='optimal')),
                    ])
_ = text_clf.fit(news_train_data, news_train_data_target)

print("Начало тестирования...")
predicted = text_clf.predict(docs_test)
print ('Точность тестирования: ', np.mean(predicted == news_training_test_data_target))


print("Получение тестовой выборки из news_test.txt...")
news_test_final = list(csv.reader(open('news_test.txt', 'rt', encoding="utf8"), delimiter='\t'))

print("Создание массива новостей...")
news_test_data_final = []
count = 1
for x in news_test_final:
    count+=1
    news_test_data_final.append(x[1])

docs_test = news_test_data_final
predicted = text_clf.predict(docs_test)

print("Запись в файл...")
f = open("final_output.txt", 'w')
for item in predicted:
  f.write("%s\n" % item)
