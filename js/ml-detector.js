/* AgroVision AI browser inference.
   FP32 ONNX export of rodynaemad/plant-disease-dinov2-small.
   Images are processed in the browser in public mode. The OOD threshold is
   intentionally not used: it previously rejected valid leaf photos too often.
   A non-tomato top class is reported as "not classified as tomato", not as
   proof that the image contains no leaf.
*/
(function () {
  "use strict";
  const REV = "06ebf5e308e100fd000d738c42b059ebb5f58a16";
  const BASE = "https://huggingface.co/rodynaemad/plant-disease-dinov2-small/resolve/" + REV + "/";
  const ORT_VERSION = "1.20.1";
  let loading = null, session = null, labels = null, ort = null;

  async function json(url) {
    const r = await fetch(url, { mode: "cors", cache: "force-cache" });
    if (!r.ok) throw new Error("Could not load model labels (" + r.status + ").");
    return r.json();
  }

  async function load() {
    if (session) return true;
    if (loading) return loading;
    loading = (async () => {
      const mod = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/ort.min.mjs");
      ort = mod;
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";
      const config = await json(BASE + "config.json");
      labels = Object.keys(config.id2label).sort((a,b)=>Number(a)-Number(b)).map(k=>config.id2label[k]);
      session = await ort.InferenceSession.create(BASE + "onnx/model_with_features.onnx", {
        executionProviders: ["wasm"], graphOptimizationLevel: "all"
      });
      if (!session.inputNames.length || !session.outputNames.includes("logits")) {
        throw new Error("The model file has an unexpected input/output format.");
      }
      return true;
    })();
    try { return await loading; } catch (e) { loading = null; session = null; throw e; }
  }

  function tensorFrom(source) {
    const width = source.naturalWidth || source.videoWidth || source.width;
    const height = source.naturalHeight || source.videoHeight || source.height;
    if (!width || !height) throw new Error("The image could not be read.");
    const scale = 256 / Math.min(width, height);
    const rw = Math.max(256, Math.round(width * scale));
    const rh = Math.max(256, Math.round(height * scale));
    const temp = document.createElement("canvas"); temp.width = rw; temp.height = rh;
    const tc = temp.getContext("2d"); tc.imageSmoothingEnabled = true; tc.imageSmoothingQuality = "high";
    tc.drawImage(source, 0, 0, rw, rh);
    const canvas = document.createElement("canvas"); canvas.width = 224; canvas.height = 224;
    const c = canvas.getContext("2d", { willReadFrequently: true });
    c.imageSmoothingEnabled = true; c.imageSmoothingQuality = "high";
    c.drawImage(temp, Math.floor((rw-224)/2), Math.floor((rh-224)/2), 224, 224, 0, 0, 224, 224);
    const px = c.getImageData(0,0,224,224).data;
    const plane = 224*224, data = new Float32Array(3*plane);
    const mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
    for(let i=0;i<plane;i++){
      data[i]=((px[i*4]/255)-mean[0])/std[0];
      data[plane+i]=((px[i*4+1]/255)-mean[1])/std[1];
      data[2*plane+i]=((px[i*4+2]/255)-mean[2])/std[2];
    }
    return new ort.Tensor("float32",data,[1,3,224,224]);
  }

  function softmax(values){let max=-Infinity;for(const x of values)max=Math.max(max,x);const e=Array.from(values,x=>Math.exp(x-max));const sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum)}
  function mapClass(label){
    if(!/^Tomato___/i.test(label))return {status:"not_tomato",key:"invalid",crop:(label.split("___")[0]||"Other")};
    const c=label.split("___").slice(1).join(" ").replace(/_/g," ").toLowerCase();
    const pairs=[["early blight","early"],["late blight","late"],["bacterial spot","bacterial"],["leaf mold","leaf_mold"],["septoria leaf spot","septoria"],["spider mites","spider_mites"],["two-spotted spider mite","spider_mites"],["target spot","target"],["yellow leaf curl virus","curl"],["mosaic virus","mosaic"],["healthy","healthy"]];
    const hit=pairs.find(([part])=>c.includes(part));
    return {status:"ok",key:hit?hit[1]:"uncertain",crop:"Tomato"};
  }

  async function predict(source){
    await load();
    const feeds={}; feeds[session.inputNames[0]]=tensorFrom(source);
    const output=await session.run(feeds);
    const probs=softmax(output.logits.data);
    const order=probs.map((score,index)=>({score,index})).sort((a,b)=>b.score-a.score).slice(0,5);
    const best=order[0], label=labels[best.index]||"Unknown";
    const mapped=mapClass(label);
    const top3=order.slice(0,3).map(x=>({label:(labels[x.index]||"Unknown").replace("Tomato___","").replace(/___/g," / ").replace(/_/g," "),score:x.score}));
    const status=mapped.status==="not_tomato"?"not_tomato":(best.score<0.35?"uncertain":"ok");
    return {status,key:status==="uncertain"?"uncertain":mapped.key,label:label.replace("Tomato___","").replace(/_/g," "),score:best.score,crop:mapped.crop,top3};
  }
  window.AgroML={load,predict};
})();
