python ./init_exp_logs.py;
python ./run_testGAN.py -f 'exp_logs_TG1.xlsx';
python ./run_testGAN.py -f 'exp_logs_TB1.xlsx';
python ./run_testGAN.py -f 'exp_logs_TD2.xlsx';
python ./find_best_setup.py;

python ./get_compare_df.py;