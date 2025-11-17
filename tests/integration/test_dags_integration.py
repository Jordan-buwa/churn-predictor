import pytest
from airflow.models import DagBag, TaskInstance
from airflow.utils.state import State
from datetime import datetime

# List all DAG IDs for test
DAG_IDS = [
    "from_postgresdb_to_azure_blob",  
    # "dag_id_2",
    # "dag_id_3",
]

# Load all DAGs from airflow/dags
dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)


@pytest.mark.parametrize("dag_id", DAG_IDS)
def test_dag_loaded(dag_id):
    """
    Ensure each DAG is correctly loaded and has tasks.
    """
    dag = dag_bag.get_dag(dag_id)
    assert dag is not None, f"DAG '{dag_id}' failed to load"
    assert len(dag.tasks) > 0, f"DAG '{dag_id}' has no tasks"


@pytest.mark.parametrize("dag_id", DAG_IDS)
def test_tasks_exist(dag_id):
    """
    Ensure all tasks exist in each DAG.
    """
    dag = dag_bag.get_dag(dag_id)
    for task in dag.tasks:
        t = dag.get_task(task.task_id)
        assert t is not None, f"Task '{task.task_id}' not found in DAG '{dag_id}'"


@pytest.mark.parametrize("dag_id", DAG_IDS)
def test_task_execution(dag_id):
    """
    Test basic execution of all tasks in each DAG.
    """
    dag = dag_bag.get_dag(dag_id)
    for task in dag.tasks:
        ti = TaskInstance(task=task, execution_date=datetime.now())
        ti.run(ignore_ti_state=True)
        assert ti.state == State.SUCCESS, f"Task '{task.task_id}' failed in DAG '{dag_id}'"
