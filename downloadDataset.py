import os
import json
import shutil
import pathlib
import subprocess

def main():

    kaggleConfigDir = f"{os.path.expandvars('$HOME')}/.kaggle"
    os.makedirs(kaggleConfigDir, exist_ok= True)
    apiToken = {
        'username':'hammad1015',
        'key'     :'80e683b1b5eae7c2ab10784934836974'
    }
    
    f = open(f'{kaggleConfigDir}/kaggle.json', 'w')
    f.write(json.dumps(apiToken))
    f.flush()
    f.close()
    
    subprocess.run(f'chmod 600 {kaggleConfigDir}/kaggle.json'.split())

    dataFolder = 'data'
    datasets = [
        'andrewmvd/car-plate-detection',
        # 'achrafkhazri/labeled-licence-plates-dataset',
    ]
    
    for dataset in datasets:
        subprocess.run(f'kaggle datasets download {dataset} --unzip -p {dataFolder}'.split())

    root = pathlib.Path(dataFolder)

    f = lambda e: (
        shutil.move(str(e),dataFolder)
        if e.is_file() else
        [f(e) for e in e.iterdir()]
    )
    f(root)

    [shutil.rmtree(str(e))
        for e in root.iterdir()
        if e.is_dir()
    ]


if __name__ == '__main__': main()