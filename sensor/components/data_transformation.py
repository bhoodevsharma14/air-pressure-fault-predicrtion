from sensor.entity import artifact_entity,config_entity
from sensor.exception import sensorException
from sensor.logger import logging
from typing import Optional
import os,sys
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN


class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise sensorException(e, sys)
    
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant',fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('Imputer',simple_imputer),
                ('RobustScaler',robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise sensorException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Droping Target Column {TARGET_COLUMN} from train and test dataframe")
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f"Saving Target Column {TARGET_COLUMN} into train and test target dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"Label Encoding on target feature of train and test dataframe")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            logging.info(f"Imputing Data and Robust Scaling Data.")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            logging.info(f"Sampling Using SMOTETomek")
            smt = SMOTETomek(random_state=42)
            input_feature_train_arr,target_feature_train_arr = smt.fit_resample(input_feature_train_arr,target_feature_train_arr)

            input_feature_test_arr,target_feature_test_arr = smt.fit_resample(input_feature_test_arr,target_feature_test_arr)

            # Target encoder
            logging.info(f"Target Encoding")
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_arr]

            # Save numpy array
            logging.info(f"Saving Target Array")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,array=test_arr)
            
            logging.info(f"Saving Transformation objects.")
            utils.save_object(file_path=self.data_transformation_config.transformation_object_path, obj=transformation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transform_artifact = artifact_entity.DataTransformationArtifact(
                transformation_object_path = self.data_transformation_config.transformation_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )
            logging.info(f"Transform data saved at {data_transform_artifact.__dict__}")

            return data_transform_artifact
        except Exception as e:
            raise sensorException(e, sys)