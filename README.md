# Описание решения
## Команда (соло) ceet (public: 1.4283389 | private: 1.26345809)
## Дополнительные данные 


| **Название датасета**  | **Описание**  | **Ссылка** |**Лицензия**|
|:------------- |:---------------:| -------------:| -----------:
| Города России | Справочник информации о городах России (Столбец за начало 2020, лика нет) |  https://github.com/hflabs/city   |  Creative Commons Attribution-ShareAlike 4.0 International License          
| Медиана дохода по субъектам Российской Федерации*   | Данные за 1 января 2020, нет лика          |         https://rosstat.gov.ru/search?q=доход   | " ...использовать открытые данные в некоммерческих и коммерческих целях..."

*Доходы по субъектам были немного обработаны вручную для удобства чтения и работы с ними, без изменения структуры данных
(Все данные в main/raif_hack/data)

## Неудачные эксперименты

1) **Обучение двух моделей**

    Были обучены две модели: для Москвы и Санкт-Петербурга, и для остальных регионов России.

    Цитата из методички Лейфера по оценке:
    >>Факторами, на основании которых мы разделили выборку, является численность населения. Москва и Санкт-Петербург были вынесены в отдельные категории, поскольку они отличаются по своим социально- экономическим характеристикам от других городов России, и соответственно имеют свои особенности ценообразования. 

    **Результат:** результат получился значительно ниже бейзлайна при довольно стабильной модели. Код в сабмите с соотв. названию.
2) **Разлиичные соотношения инфраструктуры и населения**
    Были сгенерированы фичи, например, как: 

    ```python
    dataframe['osm_crossing_points_in_0.01 per population'] = dataframe['osm_crossing_points_in_0.01'] / dataframe[f'reform_house_population_{radius}']
    ```

    Так же особого результата не получил, качество значительно ухудшилось. 





## Текущие фичи 
    Инженеринг более подробно приведен в файле: main/raif_hack/preprocessing.py

    1) Метро - была добавлена статистика по метро в городах России. (Кол-во станций, наличие станций)

    2) Расстояние от объекта до цента города(координаты центра города взяты из доп.данных). Результат улучшился, я считаю, что если шум равномерный(надеюсь хаха), то в целом шум и не особо влияет на расстояние(точнее на все расстояния влияет равномерно).

    3) Медианная зарплата по регионам(из доп данных. по доходам)

    4) Населения в каждом отдельном городе(одна из самых существенных фич по фич.импортансу) (Из доп данных.)

    5) Федеральный округ(нужен далее для статистик), тип и название города, уровень по ФИАС, координаты центра. (Из доп данных.)


    5) Различные бинарные признаки по типу: наличие метро, город-миллионник

    6) Статистики по датасету(наибольший прирост скора):
        - Медианное значение цены за кв. метр по типу здания и населения города(по уровням)
        - Медианное значение цены за кв. метр по типу здания и региону России(по уровням)
        Не забывайте,что считать можно только на трейне.
        (Было бы больше рук, можно было бы еще докинуть различных статистик, они дают большой прирост). 

    
    Остальные фичи можно подробнее посмотреть в файле main/raif_hack/preprocessing.py





## Идеидля проверки в будущем

| **Подход** | **Описание**  | **Ссылки** |
|:------------- |:---------------:| -------------:|
| Approximate Nearest Neighbors | Поиск похожих зданий в различных регионах |     https://github.com/spotify/annoy|
| Всевозможные статистики по трейну     | Одному было трудновато все успеть, поэтому хорошо бы досчитать остальные статистики        |         -  |
|:)| Сделать нормальную валидайию со сплитом по времени
| Кодировка улицы парой (город, улица) |  Улицы не зависят от города, что является определенным шумом в данных, например, S12711 - улица ленина
| Тюнинг модели| Хорошо бы подобрать хайпероптом параметры модели| 
| Проверить различные показатели от ЦБ РФ | 
| Расшифровать коды окато и октмо | Должно помочь получить больше информации о типе населенного пункта, его инфраструктуре



Запуск решения почти совпадает с запуском бейзлайна, так что трудностей возникнуть не должно)0)0)0

UPD: Обучение CatBoostRegressor(learning_rate=0.05, iterations=2500) на MANUAL_PRICE
