import pandas as pd
import subprocess
import json
import numpy as np
import os
import subprocess
import shlex
from typer.testing import CliRunner
import typer
import platform
import signal
import atexit
import psutil
import sys
from pathlib import Path
from sdcard.config import Config  
from sdcard.main import sdcard

def main(config_path: str = typer.Argument(None, help="Root path to MarImBA collection."),
         max_processes:int = typer.Option(4, help="Number of concurrent transfers"),
         format_type:str = typer.Option('exfat', help="Card format type"),
         clean:bool = typer.Option(False, help="move all the files to location"),
         debug:bool = typer.Option(False, help="Card format type")):
    config = Config(config_path)
    if debug:
        runner = CliRunner()
    processes = set()
    if platform.system() == "Linux":
        mountpoints = pd.DataFrame(json.loads(subprocess.getoutput('lsblk -J -o  NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT'))['blockdevices'])
        mountpoints = mountpoints[~mountpoints.children.isna()]
        mountpoints =pd.DataFrame(mountpoints.children.apply(lambda x: x[0]).to_list())[['name','mountpoint','fstype','size']]
        mountpoints = mountpoints.loc[(mountpoints.fstype==format_type) & (~mountpoints.mountpoint.isna())]           
        paths = pd.DataFrame(subprocess.getoutput('udevadm info -q path -n $(ls /dev/s*1)').splitlines(),columns=['Path'])
        paths[['host','dev']]=paths.Path.str.extract(r'(?P<host>host\d+).*block\/(?P<dev>([^\\]+$))')[['host','dev']]
        paths['name'] =paths.dev.str.split('/',expand=True)[1]
        mountpoints =mountpoints.merge(paths, on='name', how='inner')
        commands =[]
        for index, df in mountpoints.groupby('host'):
            args =["import"] #, "--config-path",config.get_path('CATALOG_DIR').as_posix()]    
            for card in df.mountpoint.to_list():
                args.append(card)
            if clean:
                args.append('--clean')
            commands.append(args)
        for args in commands:
            if debug:
                runner.invoke(sdcard,args)
            else:
                if len(processes) >= max_processes:
                    os.wait()
            venv_path = Path(sys.prefix)
            activate_script = venv_path / "bin" / "activate"
            #cmd = f'gnome-terminal -- bash -c "source {activate_script} && sdcard {" ".join(args)}"'
            
            cmd = f'gnome-terminal -- bash --login -c "export PATH={venv_path}/bin:$PATH && sdcard {" ".join(args)}"'
            processes.add(subprocess.Popen(shlex.split(cmd)))
        os.wait()            
    # else:
    #     commands =[]
    #     current_env = os.environ.get('VIRTUAL_ENV')
    #     marimbapath = os.path.join(current_env,'Scripts','marimba.cmd')  
    #     for i in psutil.disk_partitions():
    #         if i.fstype.lower()==format_type:
    #             p =psutil.disk_usage(i.mountpoint)
    #             if np.ceil(p.total/1000000000)<=512:
    #                 args =["start", "cmd","/k",marimbapath,"import",collection_path,instrument_id,i.mountpoint.replace('\\','/')] 
    #                 if clean:
    #                     args.append('--clean')
    #                 commands.append(args)
    #     processes = set()    
    #     for args in commands:
    #         processes.add(subprocess.Popen(args, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE))               
    #         if len(processes) >= max_processes:
    #             processes.difference_update([p for p in processes if p.poll() is not None])
    #     for p in processes:
    #         if p.poll() is None:
    #             p.wait()

if __name__ == "__main__":
    typer.run(main)