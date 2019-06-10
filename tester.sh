ITERATIONS="10000"
MAX_CAP="500"
for gen_size in 20 40 80 160 320 640
do
  for num_points in 50 100 200 400 800
  do
    for mut_prob in 0.02 0.05 0.1 0.2
    do
      for cross_prob in 0.05 0.1 0.2 0.4
      do
        python genetic.py $gen_size $ITERATIONS $mut_prob $cross_prob $MAX_CAP $gen_size -c test_files/testfile_$num_points
      done
    done
  done
done
