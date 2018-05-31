import pandas as pd

df = pd.read_csv('menu_model_addr.csv')


df = df.loc[~df.duplicated('G_path'),:]
df.iloc[-1]

df = pd.DataFrame()
exp_id = 0
while exp_id < 200:

    row = pd.DataFrame([exp_id, './logs/trained_models/GAN/GAN_G_exp{}.h5'.format(exp_id)\
                    ,'./logs/trained_models/GAN/GAN_D_exp{}.h5'.format(exp_id)\
                    ,'./logs/trained_models/GAN/GAN_T_exp{}.h5'.format(exp_id)\
                    ],index=['exp_id','G_path','D_path','T_path'])
    
    df = df.append(row.transpose())
    exp_id+=1

df.to_csv('menu_model_addr.csv',index=False)