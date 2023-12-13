from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import requests


# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def execute_zeppelin_notebook_1(**kwargs):
    # Zeppelin API endpoint to execute a notebook
    zeppelin_api_url = 'http://kpro-zeppelin-1:8082/api/notebook/job/2JGPTDQSX'

    # Make a POST request to execute the Zeppelin notebook
    response = requests.post(zeppelin_api_url)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        print("Zeppelin notebook executed successfully 1.")
    else:
        print(f"Failed to execute Zeppelin notebook 1. Status code: {response.status_code}")
        print(response.text)



def execute_zeppelin_notebook_2(**kwargs):
    # Zeppelin API endpoint to execute a notebook
    zeppelin_api_url = 'http://kpro-zeppelin-1:8082/api/notebook/job/2JHEMCM7R'

    # Make a POST request to execute the Zeppelin notebook
    response = requests.post(zeppelin_api_url)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        print("Zeppelin notebook executed successfully 2.")
    else:
        print(f"Failed to execute Zeppelin notebook 2. Status code: {response.status_code}")
        print(response.text)

def execute_zeppelin_notebook_3(**kwargs):
    # Zeppelin API endpoint to execute a notebook
    zeppelin_api_url = 'http://kpro-zeppelin-1:8082/api/notebook/job/2JGMHZU61'

    # Make a POST request to execute the Zeppelin notebook
    response = requests.post(zeppelin_api_url)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        print("Zeppelin notebook executed successfully 3.")
    else:
        print(f"Failed to execute Zeppelin notebook 3. Status code: {response.status_code}")
        print(response.text)

# Create the main DAG
dag = DAG(
    'reputation_dashboard_t',
    default_args=default_args,
    description='Reputation Dashboard DAG',
    schedule_interval=None,
)
with dag:
    # Use the PythonOperator to execute the Zeppelin notebook
    AppliqueModel_notebook_task_2 = PythonOperator(
        task_id='AppliqueModel_notebook_1',
        python_callable=execute_zeppelin_notebook_1,
        provide_context=True,  # Pass the context to the Python function
       
    )

    KafkaYoutube_notebook_task_1 = PythonOperator(
        task_id='KafkaYoutube_notebook_2',
        python_callable=execute_zeppelin_notebook_2,
        provide_context=True,  # Pass the context to the Python function
       
    )

    DashBoard_notebook_task_3 = PythonOperator(
        task_id='DashBoard_notebook_3',
        python_callable=execute_zeppelin_notebook_3,
        provide_context=True,  # Pass the context to the Python function
       
    )

    # Set task dependencies
    KafkaYoutube_notebook_task_1 >> AppliqueModel_notebook_task_2 >> DashBoard_notebook_task_3