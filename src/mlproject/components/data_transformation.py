from dataclasses import dataclass
import os
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object
import sys
import numpy as np
import pandas as pd


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):

        try:
            
            cat_column = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_column = ["reading_score","writing_score"]


            num_pipeline = Pipeline(steps=[
                ("num_imputer",SimpleImputer(strategy="median")),
                ("num_scaler",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("cat_imputer",SimpleImputer(strategy="most_frequent")),
                ("cat_onehotencoder",OneHotEncoder()),
                ("cat_standar_scaler",StandardScaler(with_mean=False))

            ])

            logging.info("Categorical feature: {cat_column}")
            logging.info("Numerical Feature: {num_column}")

            preprocessor = ColumnTransformer([

                ("numerical",num_pipeline,num_column),
                ("categorical",cat_pipeline,cat_column)

            ])

            return preprocessor

        
        except Exception as e:
           raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test data")

            preprocessing_obj = self.get_transformer_obj()

            target_column = "math_score"
            numerical_column = ["reading_score","writing_score"]

            # train
            input_feature_train_df = train_df.drop([target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            # test
            input_feature_test_df = test_df.drop([target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying the preprocessing of the training and testing data")

            # train
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            logging.info("saved preprocessing info")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        
        except Exception as e:
            raise CustomException(e,sys)