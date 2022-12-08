from sensor.entity import artifact_entity,config_entity
from sensor.exception import sensorException
from sensor.logger import logging
from sensor import utils
from typing import Optional
import os,sys
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainingConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise sensorException(e,sys)

    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise sensorException(e,sys)

    def initiate_model_trainer(self)->artifact_entity.ModelTrainingArtifact:
        try:
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            model = self.train_model(x=x_train, y=y_train)

            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)
            
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            if f1_test_score < self.model_trainer_config.expected_score:
                raise sensorException(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}, model_actual_score:{f1_test_score}",sys)

            diff = abs(f1_train_score-f1_test_score)

            if diff > self.model_trainer_config.overfitting_treshold:
                raise sensorException(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}",sys)
            
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact  = artifact_entity.ModelTrainingArtifact(model_path=self.model_trainer_config.model_path,f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            
            return model_trainer_artifact

        except Exception as e:
            raise sensorException(e,sys)