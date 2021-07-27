# AutoML. Модель с алгоритмом классификации
* в данном примере рассматриваем бинарную классификацию;
* модель учится различать "токсичные" и "нетоксичные" комментарии;
* модель обучается на готовой таблице, которую можно найти в папке [datasets](https://github.com/SpaciSoxrani/UDV-Summer-School-2021/tree/add-object-detection/tasks/Smirnova_Anastasiya_AutoML(regression_and_classification)/AutoML_BinaryClassification/AutoMLApp/datasets)
* хорошие комментарии имеют оценку - 0, плохие - 1;
* модель тестируется на второй таблице из той же папки. Высчитывыется ее точность предсказаний;
* с помощью AutoML выбирается лучшая метрика для данного случая с большой точностью;
* в методе TestSinglePrediction мы тестируем собственный комментарий. На консоле увидим прогноз модели.

[мой репозиторий](https://github.com/SpaciSoxrani/AutoMLApp_BinaryClassification)
