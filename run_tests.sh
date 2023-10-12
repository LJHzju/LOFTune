mode='single'
workloads=('TPCDS_20' 'IMDB_10')

for workload in "${workloads[@]}"
do
  # Directory containing the SQL files
  new_tasks_file="./data/${workload}G_${mode}/new_tasks"
  array=(${workload//_/ })
  workload=${array[0]}
  data_size=${array[1]}

  # Get a list of all SQL files in the directory
  # shellcheck disable=SC2207
  new_tasks=($(cat "$new_tasks_file"))

  # Run the tasks using the main.py script
  for task in "${new_tasks[@]}"
  do
    python main.py --mode ${mode} --workload "${workload}" --data_size "${data_size}" --type recommend-config-exp3 --task_id "${task}" --epochs 1
  done
done