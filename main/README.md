# Описание

Этот скрипт повторяет по структуре бейзлайн от Райфа.

# Запуск

<li> запустить обучение

    poetry run python3 train.py --train_data <path_to_train_data> --model_path <path_to_pickle_ml_model>
</li>
    <li> запустить предикт

    poetry run python3 predict.py --model_path <path_to_pickled_model> --test_data <path_to_test_data> --output <path_to_output_csv_file>

## Описание директорий:
* models - модели к каждому комиту(с хорошим скором)
* outputs - файлы с предиктами
* notebooks - тетрадки с экспериментами, обзором данных(советую pandas profiling), фич. импортансом моделей(советую eli5)
* raif_hack - технические файлы, как в бейзлайне