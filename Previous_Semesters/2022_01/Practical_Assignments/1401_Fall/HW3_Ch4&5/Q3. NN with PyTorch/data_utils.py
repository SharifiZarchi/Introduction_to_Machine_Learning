import os
import subprocess
import pandas as pd


def download_data():
    if not os.path.exists('./efigi-1.6/png/'):
        subprocess.run([
            'wget',
            'https://www.astromatic.net/download/efigi/efigi_png_gri-1.6.tgz',
            '-O data.tgz'
        ])
        subprocess.run([
            'tar',
            '-xzvf data.tgz'
        ])
        os.remove('./ data.tgz')
        print('efigi_png downloaded!')

    if not os.path.exists('./efigi-1.6/EFIGI_attributes.txt'):
        subprocess.call([
            'wget',
            'https://www.astromatic.net/download/efigi/efigi_tables-1.6.2.tgz',
            '-O data.tgz'
        ])
        subprocess.call([
            'tar',
            '-xzvf data.tgz'
        ])
        os.remove('./ data.tgz')

        path = './efigi-1.6/'
        for f in os.listdir('./efigi-1.6/'):
            if f != 'EFIGI_attributes.txt' and f.endswith('.txt'):
                os.remove(f'{path}/{f}')
        
        print('efigi_tables downloaded!')
        

def prepare_data(path):
    with open(path, 'rb') as f:
        line = None
        for i in range(82):
            line = f.readline()
        headers = str(line).split()[1:]

    df = pd.read_csv(path, comment='#', header=None)
    s = df[0].str.split()
    df = pd.DataFrame(s.values.tolist(), index=s.index, columns=headers)
    df['T'] = df['T'].astype(int)
    
    class_names = ['Ellipticals', 'Lenticulars', 'Spirals', 'Irregulars']
    class_types = [[-4, -5, -6], [-3, -2, -1], list(range(0, 10)), [90, 10, 11]]

    data = pd.DataFrame()
    data['name'] = df['PGCname']
    for i, (c_name, c_type) in enumerate(zip(class_names, class_types)):
        mask = df['T'].isin(c_type)
        data.loc[mask, 'class_name'] = c_name
        data.loc[mask, 'class_label'] = i

    data['class_label'] = data['class_label'].astype(int)
    return data