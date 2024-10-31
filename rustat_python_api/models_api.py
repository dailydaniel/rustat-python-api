import requests
import numpy as np
import pandas as pd


class DynamoLab:
    def __init__(self, host: str = "http://localhost:8000/"):
        self.host = host

    def get_active_models(self):
        url = self.host + "models"
        response = requests.get(url)
        return response.json()

    def run_model(
        self,
        model: str,
        data: pd.DataFrame,
        return_type: str = "json",
        inplace: bool = False,
        return_df: bool = False,
        inplace_column: str = None
    ):
        """
        model: str
            The name of the model to run
        return_type: str
            One of: "json", "list", "numpy"
        """

        url = self.host + "predict"
        csv_string = data.to_csv(index=False)

        body = {
            "model": model,
            "data": csv_string,
        }

        response = requests.post(url, json=body)

        if return_df or inplace:
            if inplace_column is None:
                raise ValueError("inplace_column must be specified if inplace is True")
            if inplace:
                data[inplace_column] = np.array(response.json()["prediction"])
            else:
                df = data.copy()
                df[inplace_column] = np.array(response.json()["prediction"])
                return df
        else:
            if return_type == "json":
                return response.json()
            elif return_type == "list":
                return response.json()["prediction"]
            elif return_type == "numpy":
                return np.array(response.json()["prediction"])
            else:
                raise ValueError("return_type must be one of: json, list, numpy")
