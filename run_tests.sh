mode='multi'
task_strs=('TPCDS_300#')

for task_str in "${task_strs[@]}"
do
  # Directory containing the SQL files

    # Split by '#'
  workload_ds=$(echo $task_str | cut -d'#' -f1)
  task_suffix=$(echo $task_str | cut -d'#' -f2)

  # Split the result by '_'
  workload=$(echo $workload_ds | cut -d'_' -f1)
  data_size=$(echo $workload_ds | cut -d'_' -f2)

  new_tasks_file="./data/${workload}_${data_size}G_${mode}/new_tasks${task_suffix}"

  # Get a list of all SQL files in the directory
  # shellcheck disable=SC2207
  new_tasks=($(cat "$new_tasks_file"))

  # Run the tasks using the main.py script
  for task in "${new_tasks[@]}"
  do
    python main.py --mode ${mode} \
                   --workload "${workload}" \
                   --data_size "${data_size}" \
                   --type recommend-config-exp3 \
                   --task_id "${task}" \
                   --model tbcnn \
                   --epochs 35 \
                   --task_suffix "${task_suffix}"
  done
done