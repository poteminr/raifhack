import argparse
import logging.config
import pandas as pd
from traceback import format_exc

from raif_hack.model import BenchmarkModel
from raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat
from raif_hack.features import prepare_categorical
from raif_hack import preprocessing
from raif_hack.utils import UNKNOWN_VALUE

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path_russia", "-mpr", type=str, dest="mpr", required=True, help="Куда сохранить обученную ML модель")
    parser.add_argument("--model_path_mspb", "-mpspb", type=str, dest="mpspb", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)
        train_df = preprocessing.preprocessing(train_df)

        train_df[['federal_district']] = train_df[['federal_district']].fillna(UNKNOWN_VALUE)

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model_russia = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)

        model_moscow_and_spb = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)


        X_offer_mspb = X_offer[(X_offer.city == "Москва") | (X_offer.city == "Санкт-Петербург")]
        y_offer_mspb = train_df[(train_df.price_type == PriceTypeEnum.OFFER_PRICE) & ((train_df.city == "Москва") | (train_df.city == "Санкт-Петербург"))][TARGET]
        X_manual_mspb = X_manual[(X_manual.city == "Москва") | (X_manual.city == "Санкт-Петербург")]
        y_manual_mspb = train_df[(train_df.price_type == PriceTypeEnum.MANUAL_PRICE) & ((train_df.city == "Москва") | (train_df.city == "Санкт-Петербург"))][TARGET]

        X_offer_russia = X_offer[(X_offer.city != "Москва") & (X_offer.city != "Санкт-Петербург")]
        y_offer_russia = train_df[(train_df.price_type == PriceTypeEnum.OFFER_PRICE) & ((train_df.city != "Москва") & (train_df.city != "Санкт-Петербург"))][TARGET]
        X_manual_russia = X_manual[(X_manual.city != "Москва") & (X_manual.city != "Санкт-Петербург")]
        y_manual_russia = train_df[(train_df.price_type == PriceTypeEnum.MANUAL_PRICE) & ((train_df.city != "Москва") & (train_df.city != "Санкт-Петербург"))][TARGET]      



        print(X_offer_mspb.shape)
        print(X_manual_mspb.shape)

        print(X_offer_russia.shape)
        print(X_manual_russia.shape)



        logger.info('Fit model')
        # model.fit(X_offer, y_offer, X_manual, y_manual)

        model_russia.fit(X_offer_russia, y_offer_russia, X_manual_russia, y_manual_russia, [f'{i}' for i in range(99)])
        model_moscow_and_spb.fit(X_offer_mspb, y_offer_mspb, X_manual_mspb, y_manual_mspb, [f'{i}' for i in range(92)])

        logger.info('Save model')
        # model.save(args['mp'])
        model_russia.save(args['mpr'])
        model_moscow_and_spb.save(args['mpspb'])

        # predictions_offer = model.predict(X_offer)
        predictions_offer_russia = model_russia.predict(X_offer_russia)
        predictions_offer_mspb = model_moscow_and_spb.predict(X_offer_mspb)


        # metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        # logger.info(f'Metrics stat for training data with offers prices: {metrics}')
        metrics_russia = metrics_stat(y_offer_russia.values, predictions_offer_russia/(1+model_russia.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        metrics_spbm = metrics_stat(y_offer_mspb.values, predictions_offer_mspb/(1+model_moscow_and_spb.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента



        # predictions_manual = model.predict(X_manual)
        predictions_manual_russia = model_russia.predict(X_manual_russia)
        predictions_manual_spbm = model_moscow_and_spb.predict(X_manual_mspb)



        # metrics = metrics_stat(y_manual.values, predictions_manual)
        # logger.info(f'Metrics stat for training data with manual prices: {metrics}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')