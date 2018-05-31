#python ./run.py;

python ./analysis/compare/testGAN/init_exp_logs.py;
python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TG1.xlsx';
python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TB1.xlsx';
python ./analysis/compare/testGAN/run_testGAN.py -f 'exp_logs_TD2.xlsx';
python ./analysis/compare/testGAN/find_best_setup.py;

python ./analysis/compare/testGAN/get_compare_df.py;
#python ./analysis/reduce_dim/reduce_dim.py;
