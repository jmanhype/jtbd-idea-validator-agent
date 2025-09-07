import requests, os

QC_URL = "https://quickchart.io/chart"
HEADERS = {"User-Agent": "jtbd-agent-dspy/1.0"}

def _save_chart(cfg: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.post(QC_URL, json={"backgroundColor":"white","width":900,"height":600,"format":"png","version":"4","chart":cfg}, headers=HEADERS, timeout=25)
    r.raise_for_status()
    with open(out_path, "wb") as f: f.write(r.content)

def radar(initial_vals, final_vals, labels, out_path: str):
    cfg = {
      "type":"radar",
      "data":{"labels":labels,"datasets":[{"label":"Initial","data":initial_vals},{"label":"Final","data":final_vals}]},
      "options":{"responsive":False,"plugins":{"legend":{"position":"bottom"}}}
    }
    _save_chart(cfg, out_path)

def waterfall(parts, out_path: str):
    labels = [p[0] for p in parts]; data=[]; sumv=0
    for _,d in parts:
        sumv += d; data.append(sumv)
    cfg = {
      "type":"bar",
      "data":{"labels":labels,"datasets":[{"label":"ΔScore cumulative","data":data}]},
      "options":{"responsive":False,"plugins":{"legend":{"display":False}}}
    }
    _save_chart(cfg, out_path)

def forces(push, pull, anxiety, inertia, out_path: str):
    text = f"Push: {len(push)}  Pull: {len(pull)}  Anxiety: {len(anxiety)}  Inertia: {len(inertia)}"
    cfg = {"type":"bar","data":{"labels":["Push","Pull","Anxiety","Inertia"],"datasets":[{"data":[len(push),len(pull),len(anxiety),len(inertia)]}]},"options":{"plugins":{"title":{"display":True,"text":text}}}}
    _save_chart(cfg, out_path)
