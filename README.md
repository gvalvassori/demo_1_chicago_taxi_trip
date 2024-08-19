
# GCP Machine Learning Specialization: Demo 1 - Chicago Taxi Trips

Caso de uso para demostración de buenas prácticas en Data Science, Machine Learning y MLOps para la especialización a nivel compañía. 

En esta demostración se utiliza el challenge de Kaggle de [Chicago Taxi Trips](https://www.kaggle.com/datasets/chicago/chicago-taxi-trips-bq) con el objetivo de predecir el costo total de un viaje en taxi en la ciudad de Chicago. Para tal propósito, se realiza la práctica end-to-end que contempla:

    1. Análisis Exploratorio de Datos.
    2. Desarrollo y selección de modelo y características.
    3. Desarrollo de pipeline de MLOps para disponibilizar el modelo en un endpoint de Vertex Análisis.
    4. Análisis de performance del modelo desplegado.




## Instalación

Para ejecutar localmente los códigos, es necesario tener acceso al proyecto de GCP. En caso de que usted lo tenga, luego de clonar el repositorio es suficiente con crear un entorno virtual de Python (3.10) y ejecutar el siguiente comando

```bash
  pip install -r requirements.txt
```

Posteriormente, podrá ejecutar todas las notebooks y reproducir la demo.

## Deployment

Para desplegar el modelo seleccionado a un endpoint de Vertex AI o bien reentrenar, alcanza con ejecutar la notebook de MLops. Luego dentro de Vertex AI Pipelines se podrá ver la canalización. Finalmente el endpoint estará disponible en Vertex AI Online Predictions.

El pipeline fue hecho con Tensorflow eXtended.


## Acknowledgements / Code origin certification
Todos los códigos fueron desarrollados por CoreBI S.A., la parte de MLOps construída en TFX se basó en la documentación oficial que fue adaptada y modificada para la presente demo:

 - [TFX User Guide](https://www.tensorflow.org/tfx/guide)
 - [Vertex AI Training and Serving with TFX and Vertex Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/gcp/vertex_pipelines_vertex_training)
 - [Simple TFX Pipeline Tutorial using Penguin dataset](https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple)

Todo lo utilizado es de fuentes Open Source.
