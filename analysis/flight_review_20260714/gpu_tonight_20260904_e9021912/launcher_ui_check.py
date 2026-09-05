import os,sys,json,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()));os.environ['DRONE_VISION_NO_RELAUNCH']='1'
import tkinter as tk
import app_Launcher_v2 as launcher
from m5_field_launch import mission_arguments,MOTION,GPU_RECONSTRUCTION
out=Path('/tmp/drone-tonight-gpu-20260904/launcher-ui.json');commands=[]
class Child:
 pid=999999999
 def poll(self):return None

def spawn(cmd,**kw):commands.append(cmd);return Child()
prefs=launcher.load_prefs(str(Path.cwd()));prefs.update(auto_start_stream=False,auto_launch_default_script=False)
with patch.object(launcher,'load_prefs',return_value=prefs),patch.object(launcher,'save_prefs'),patch.object(launcher.App,'_start_connection_monitor'),patch.object(launcher.App,'_start_ip_refresh'),patch.object(launcher,'terminate_process_tree'),patch.object(launcher.subprocess,'Popen',side_effect=spawn):
 root=tk.Tk();app=launcher.App(root,str(Path.cwd()));root.update()
 labels=[]
 for mission in [MOTION,*GPU_RECONSTRUCTION]:
  app.select_script(mission);app.launch_script();root.update();labels.extend(str(b.cget('text')) for b in app.script_buttons[mission]);app.kill_script()
 for cmd,mission in zip(commands,[MOTION,*GPU_RECONSTRUCTION]):assert cmd[2:]==mission_arguments(mission),(cmd,mission)
 app.on_exit()
 out.write_text(json.dumps(dict(status='PASS_LAUNCHER_UI_COMMANDS',scope='Real Tk widgets and launch callback; child process spawn intercepted, actual missions tested separately via RTMP',commands=commands,labels=labels,preferences_unchanged=True),indent=2)+'\n')
 print(out)
