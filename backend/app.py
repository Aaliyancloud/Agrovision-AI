import io, os, json, threading
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MODEL_ID = os.getenv('AGROVISION_MODEL_ID', 'rodynaemad/plant-disease-dinov2-small')
MODEL_REV = os.getenv('AGROVISION_MODEL_REV', '06ebf5e308e100fd000d738c42b059ebb5f58a16')
MODEL_FILE = os.getenv('AGROVISION_MODEL_FILE', 'onnx/model_with_features.onnx')
MAX_BYTES = 15 * 1024 * 1024
app = FastAPI(title='AgroVision AI API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['GET','POST','OPTIONS'], allow_headers=['*'])
_session = None
_id2label = None
_load_lock = threading.Lock()

TOMATO_MAP = {
 'early blight':'early', 'late blight':'late', 'bacterial spot':'bacterial',
 'leaf mold':'leaf_mold', 'septoria leaf spot':'septoria',
 'spider mites':'spider_mites', 'two-spotted spider mite':'spider_mites',
 'target spot':'target', 'tomato yellow leaf curl virus':'curl',
 'yellow leaf curl virus':'curl', 'tomato mosaic virus':'mosaic', 'healthy':'healthy'
}

def _load_model():
    global _session, _id2label
    if _session is not None: return
    with _load_lock:
        if _session is not None: return
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=MODEL_ID, filename='config.json', revision=MODEL_REV)
        with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
        labels = config.get('id2label') or config.get('label2id')
        if not labels: raise RuntimeError('Model config has no class labels.')
        if 'id2label' in config:
            _id2label = {int(k): str(v) for k,v in config['id2label'].items()}
        else:
            _id2label = {int(v): str(k) for k,v in config['label2id'].items()}
        model_path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE, revision=MODEL_REV)
        opts = ort.SessionOptions(); opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2)); opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])

def _preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert('RGB')
    w,h = image.size
    if min(w,h) < 80: raise ValueError('Image is too small. Use a clearer photo.')
    # DINOv2 processor recipe: resize shortest edge to 256, center crop 224, ImageNet normalization.
    scale = 256.0 / min(w,h)
    nw,nh = max(256, round(w*scale)), max(256, round(h*scale))
    image = image.resize((nw,nh), Image.Resampling.BICUBIC)
    left,top=(nw-224)//2,(nh-224)//2
    image=image.crop((left,top,left+224,top+224))
    arr=np.asarray(image,dtype=np.float32)/255.0
    mean=np.array([0.485,0.456,0.406],dtype=np.float32)
    std=np.array([0.229,0.224,0.225],dtype=np.float32)
    arr=(arr-mean)/std
    return np.transpose(arr,(2,0,1))[None,...].astype(np.float32)

def _softmax(x):
    x=x.astype(np.float64); x=x-np.max(x); e=np.exp(x); return e/e.sum()

def _predict(image: Image.Image) -> dict[str, Any]:
    _load_model()
    tensor=_preprocess(image)
    input_name=_session.get_inputs()[0].name
    outputs=_session.run(None,{input_name:tensor})
    # The published ONNX export returns logits and a feature vector.
    logits=outputs[0].reshape(-1)
    probs=_softmax(logits)
    ranked=np.argsort(probs)[::-1]
    best_i=int(ranked[0]); best_label=_id2label.get(best_i, f'class_{best_i}')
    top=[]
    for i in ranked[:5]:
        label=_id2label.get(int(i),f'class_{int(i)}')
        top.append({'label':label.replace('Tomato___','').replace('___',' / ').replace('_',' '),'score':float(probs[int(i)])})
    # This model is closed-set. A non-tomato top class is a useful crop rejection,
    # but it is not a universal non-leaf detector. Do not invent an OOD guarantee.
    if not best_label.lower().startswith('tomato___'):
        return {'status':'not_tomato','key':'invalid','label':best_label,'score':float(probs[best_i]),'crop':best_label.split('___')[0],'top3':top}
    condition=best_label.split('___',1)[-1].replace('_',' ').lower()
    key='uncertain'
    for fragment,mapped in TOMATO_MAP.items():
        if fragment in condition: key=mapped; break
    score=float(probs[best_i])
    if score < 0.35: status='uncertain'
    else: status='ok'
    return {'status':status,'key':key,'label':best_label.split('___',1)[-1].replace('_',' '),'score':score,'crop':'Tomato','top3':top}

@app.get('/api/health')
def health():
    return {'ok':True,'service':'AgroVision AI','model_loaded':_session is not None,'model_id':MODEL_ID,'model_revision':MODEL_REV}

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(415,'Upload a JPG, PNG or WebP image.')
    raw=await file.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES: raise HTTPException(413,'Image is larger than 15 MB.')
    try:
        image=Image.open(io.BytesIO(raw))
        if image.width*image.height > 25_000_000: raise HTTPException(413,'Image dimensions are too large.')
        result=_predict(image)
        return {'ok':True,'result':result}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f'Prediction could not run: {str(e)[:220]}')

@app.get('/')
def home():
    return FileResponse(PROJECT/'index.html')

# Serve the existing one-page frontend and its assets when running locally.
@app.get('/js/{asset_path:path}')
def js_assets(asset_path: str):
    p=(PROJECT/'js'/asset_path).resolve()
    if not str(p).startswith(str((PROJECT/'js').resolve())) or not p.is_file(): raise HTTPException(404,'Script not found')
    return FileResponse(p, media_type='application/javascript')

@app.get('/assets/{asset_path:path}')
def assets(asset_path: str):
    p=(PROJECT/'assets'/asset_path).resolve()
    if not str(p).startswith(str((PROJECT/'assets').resolve())) or not p.is_file(): raise HTTPException(404,'Asset not found')
    return FileResponse(p)
