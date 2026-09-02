#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path

BASE_DIR = Path("/Users/robzomb/Documents/antigravity/cool-hopper")
STATUS_FILE = BASE_DIR / "generation_status.json"

def get_status_data():
    if not STATUS_FILE.exists():
        return {"active_task": None, "history": []}
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return {"active_task": None, "history": []}

def save_status_data(data):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_task_status(task_id, project_name, status, step_curr=0, step_total=8, denoise_s=0.0, vae_s=0.0, master_file="", error=""):
    data = get_status_data()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    task_entry = {
        "task_id": task_id,
        "project_name": project_name,
        "status": status, # "QUEUED", "RUNNING", "COMPLETED", "FAILED"
        "step_current": step_curr,
        "step_total": step_total,
        "progress_pct": round(100.0 * step_curr / step_total, 1) if step_total > 0 else 100.0,
        "denoise_sec": denoise_s,
        "vae_sec": vae_s,
        "gpu_total_sec": round(denoise_s + vae_s, 2),
        "master_file": str(master_file),
        "error": error,
        "updated_at": now_str
    }
    
    # Update active task or history
    if status == "RUNNING" or status == "QUEUED":
        data["active_task"] = task_entry
    else:
        if data.get("active_task") and data["active_task"].get("task_id") == task_id:
            data["active_task"] = None
        # Add to history (deduplicate)
        data["history"] = [h for h in data.get("history", []) if h.get("task_id") != task_id]
        data["history"].insert(0, task_entry)
        data["history"] = data["history"][:20] # Keep last 20 runs
        
    save_status_data(data)

def render_cli():
    data = get_status_data()
    active = data.get("active_task")
    history = data.get("history", [])
    
    os.system("clear" if os.name == "posix" else "cls")
    print("\033[1;36m" + "=" * 100 + "\033[0m")
    print("\033[1;37m 🎬 MINIMAX H3-MAX / N-GRAM ENGINE · DASHBOARD STATO GENERAZIONE CLI\033[0m")
    print("\033[1;36m" + "=" * 100 + "\033[0m")
    
    if active:
        pct = active.get('progress_pct', 0)
        bar_len = 30
        filled = int(bar_len * pct / 100.0)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n \033[1;33m🔄 TASK ATTIVO IN ESECUZIONE:\033[0m")
        print(f"   • Progetto:      \033[1m{active.get('project_name')}\033[0m (ID: {active.get('task_id')})")
        print(f"   • Stato:         \033[1;32m{active.get('status')}\033[0m | Step: {active.get('step_current')}/{active.get('step_total')}")
        print(f"   • Progresso:     [\033[1;32m{bar}\033[0m] {pct}%")
        print(f"   • Ultimo Update: {active.get('updated_at')}")
    else:
        print("\n \033[1;32m💤 NESSUN TASK IN ESECUZIONE AL MOMENTO (GPU IDLE)\033[0m")
        
    print("\n" + "\033[1;34m" + "-" * 100 + "\033[0m")
    print("\033[1;37m 📋 STORICO DEGLI ULTIMI LAVORI COMPLETATI:\033[0m")
    print("\033[1;34m" + "-" * 100 + "\033[0m")
    
    if not history:
        print("   Nessun lavoro registrato.")
    else:
        header = f" {'STATO':<12} | {'PROGETTO':<32} | {'DENOISE':<10} | {'VAE':<8} | {'GPU TOT':<10} | {'COMPLETATO IL':<19}"
        print("\033[1;30;47m" + header + "\033[0m")
        for h in history[:8]:
            st = h.get('status', 'DONE')
            st_color = "\033[1;32m" if st == "COMPLETED" else ("\033[1;31m" if st == "FAILED" else "\033[1;33m")
            st_str = f"{st_color}{st:<12}\033[0m"
            pname = h.get('project_name', 'N/A')[:30]
            den = f"{h.get('denoise_sec', 0.0):.2f}s"
            vae = f"{h.get('vae_sec', 0.0):.2f}s"
            gpu = f"{h.get('gpu_total_sec', 0.0):.2f}s"
            upd = h.get('updated_at', 'N/A')
            print(f" {st_str} | {pname:<32} | {den:<10} | {vae:<8} | {gpu:<10} | {upd:<19}")
            if h.get('master_file'):
                print(f"   \033[0;36m↳ Master:\033[0m \033[4;37m{h.get('master_file')}\033[0m")
                
    print("\n\033[1;36m" + "=" * 100 + "\033[0m")
    print(" \033[0;37mComandi rapidi: \033[1m./h3_status.sh\033[0m (istantaneo) | \033[1m./h3_status.sh --watch\033[0m (live refresh senza AI token)\033[0m\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        try:
            while True:
                render_cli()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nChiusura monitor live.")
    else:
        render_cli()
