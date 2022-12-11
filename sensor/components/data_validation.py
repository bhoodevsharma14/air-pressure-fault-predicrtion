import sys,os
from sensor.entity import artifact_entity,config_entity
from sensor.exception import sensorException
from sensor.logger import logging
from typing import Optional
from scipy.stats import ks_2samp
import pandas as pd
from sensor import utils
import numpy as np
from sensor.config import TARGET_COLUMN

class DataValidation:

    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise sensorException(e, sys)

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str,threshold=0.3)->Optional[pd.DataFrame]:
        """
        This Function will drop column which contains missing values more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column 
        ======================================================================================
        returns Pandas DataFrame if atlest a single column is available in after removing the columns having null value more than 30%
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # Selecting Column Names which has to be droped
            drop_column_names = null_report[null_report>threshold].index
            
            self.validation_error["droped columns"] = list(drop_column_names)
            logging.info(f"Droping columns from {report_key_name} which does not meet theshold {threshold}\n columns are \n {self.validation_error['droped columns']} ")
            df.drop(list(drop_column_names),axis=1,inplace=True)

            if len(df.columns)==0:
                return None
            
            return df

        except Exception as e:
            raise sensorException(e, sys)

    def is_required_column_exsist(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            logging.info(f"Checking If required columns exsists in {report_key_name}")
            base_columns = base_df.columns
            current_columns = current_df.columns
            
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                logging.info(f"{report_key_name} having missing column {missing_columns}")
                self.validation_error["Missing Columns"] = missing_columns
                return False
            return True

        except Exception as e:
            raise sensorException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            logging.info(f"Checking Data drift {report_key_name}")
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns
            
            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                # Null Hypothesis is that both column drawn from same column data drawn from same distribution
                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue > 0.05:
                    # accepting Null Hypothesis
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                else:
                    # rejecting null hypothesis
                    logging.info(f"There is Data Drift in {base_column}")
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
            
            self.validation_error[report_key_name] = drift_report


        except Exception as e:
            raise sensorException(e, sys)

    def initiate_data_validation(self)->None:
        try:
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace=True)

            # base_df has na as null
            base_df = self.drop_missing_values_columns(df=base_df,report_key_name="missing_values_within_base_dataset")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_train_dataset")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_within_test_dataset")
            
            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_column_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_column_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_column_float(df=test_df, exclude_columns=exclude_columns)

            train_df_columns_status = self.is_required_column_exsist(base_df=base_df,current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            test_df_columns_status = self.is_required_column_exsist(base_df=base_df,current_df=test_df,report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:
                self.data_drift(base_df=base_df,current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                self.data_drift(base_df=base_df,current_df=test_df,report_key_name="data_drift_within_test_dataset")
            
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)

            return data_validation_artifact


        except Exception as e:
            raise sensorException(e, sys)
